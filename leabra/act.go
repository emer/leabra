// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/chewxy/math32"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/chans"
	"github.com/emer/leabra/knadapt"
	"github.com/emer/leabra/nxx1"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

///////////////////////////////////////////////////////////////////////
//  act.go contains the activation params and functions for leabra

// leabra.ActParams contains all the activation computation params and functions
// for basic Leabra, at the neuron level .
// This is included in leabra.Layer to drive the computation.
type ActParams struct {
	XX1        nxx1.Params     `view:"inline" desc:"Noisy X/X+1 rate code activation function parameters"`
	OptThresh  OptThreshParams `view:"inline" desc:"optimization thresholds for faster processing"`
	Init       ActInitParams   `view:"inline" desc:"initial values for key network state variables -- initialized at start of trial with InitActs or DecayActs"`
	Dt         DtParams        `view:"inline" desc:"time and rate constants for temporal derivatives / updating of activation state"`
	Gbar       chans.Chans     `view:"inline" desc:"[Defaults: 1, .2, 1, 1] maximal conductances levels for channels"`
	Erev       chans.Chans     `view:"inline" desc:"[Defaults: 1, .3, .25, .1] reversal potentials for each channel"`
	Clamp      ClampParams     `view:"inline" desc:"how external inputs drive neural activations"`
	Noise      ActNoiseParams  `view:"inline" desc:"how, where, when, and how much noise to add to activations"`
	VmRange    minmax.F32      `view:"inline" desc:"range for Vm membrane potential -- [0, 2.0] by default"`
	KNa        knadapt.Params  `view:"no-inline" desc:"sodium-gated potassium channel adaptation parameters -- activates an inhibitory leak-like current as a function of neural activity (firing = Na influx) at three different time-scales (M-type = fast, Slick = medium, Slack = slow)"`
	ErevSubThr chans.Chans     `inactive:"+" view:"-" json:"-" xml:"-" desc:"Erev - Act.Thr for each channel -- used in computing GeThrFmG among others"`
	ThrSubErev chans.Chans     `inactive:"+" view:"-" json:"-" xml:"-" desc:"Act.Thr - Erev for each channel -- used in computing GeThrFmG among others"`
}

func (ac *ActParams) Defaults() {
	ac.XX1.Defaults()
	ac.OptThresh.Defaults()
	ac.Init.Defaults()
	ac.Dt.Defaults()
	ac.Gbar.SetAll(1.0, 0.1, 1.0, 1.0)
	ac.Erev.SetAll(1.0, 0.3, 0.25, 0.25)
	ac.Clamp.Defaults()
	ac.VmRange.Max = 2.0
	ac.KNa.Defaults()
	ac.KNa.On = false
	ac.Noise.Defaults()
	ac.Update()
}

// Update must be called after any changes to parameters
func (ac *ActParams) Update() {
	ac.ErevSubThr.SetFmOtherMinus(ac.Erev, ac.XX1.Thr)
	ac.ThrSubErev.SetFmMinusOther(ac.XX1.Thr, ac.Erev)

	ac.XX1.Update()
	ac.OptThresh.Update()
	ac.Init.Update()
	ac.Dt.Update()
	ac.Clamp.Update()
	ac.Noise.Update()
	ac.KNa.Update()
}

///////////////////////////////////////////////////////////////////////
//  Init

// InitGinc initializes the Ge excitatory and Gi inhibitory conductance accumulation states
// including ActSent and G*Raw values.
// called at start of trial always, and can be called optionally
// when delta-based Ge computation needs to be updated (e.g., weights
// might have changed strength)
func (ac *ActParams) InitGInc(nrn *Neuron) {
	nrn.ActSent = 0
	nrn.GeRaw = 0
	nrn.GeInc = 0
	nrn.GiRaw = 0
	nrn.GiInc = 0
}

// DecayState decays the activation state toward initial values in proportion to given decay parameter
// Called with ac.Init.Decay by Layer during AlphaCycInit
func (ac *ActParams) DecayState(nrn *Neuron, decay float32) {
	if decay > 0 { // no-op for most, but not all..
		nrn.Act -= decay * (nrn.Act - ac.Init.Act)
		nrn.Ge -= decay * (nrn.Ge - ac.Init.Ge)
		nrn.Gi -= decay * nrn.Gi
		nrn.GiSelf -= decay * nrn.GiSelf
		nrn.Gk -= decay * nrn.Gk
		nrn.Vm -= decay * (nrn.Vm - ac.Init.Vm)
	}
	nrn.ActDel = 0
	nrn.Inet = 0
}

// InitActs initializes activation state in neuron -- called during InitWts but otherwise not
// automatically called (DecayState is used instead)
func (ac *ActParams) InitActs(nrn *Neuron) {
	nrn.Act = ac.Init.Act
	nrn.ActLrn = ac.Init.Act
	nrn.Ge = ac.Init.Ge
	nrn.Gi = 0
	nrn.Gk = 0
	nrn.GknaFast = 0
	nrn.GknaMed = 0
	nrn.GknaSlow = 0
	nrn.GiSelf = 0
	nrn.Inet = 0
	nrn.Vm = ac.Init.Vm
	nrn.Targ = 0
	nrn.Ext = 0
	nrn.ActDel = 0
	nrn.Spike = 0
	nrn.ISI = -1
	nrn.ISIAvg = -1

	ac.InitActQs(nrn)
	ac.InitGInc(nrn)
}

// InitActQs initializes quarter-based activation states in neuron (ActQ0-2, ActM, ActP, ActDif)
// Called from InitActs, which is called from InitWts, but otherwise not automatically called
// (DecayState is used instead)
func (ac *ActParams) InitActQs(nrn *Neuron) {
	nrn.ActQ0 = 0
	nrn.ActQ1 = 0
	nrn.ActQ2 = 0
	nrn.ActM = 0
	nrn.ActP = 0
	nrn.ActDif = 0
}

///////////////////////////////////////////////////////////////////////
//  Cycle

// GRawFmInc integrates G conductance from Inc delta-increment sent.
func (ac *ActParams) GRawFmInc(nrn *Neuron) {
	nrn.GeRaw += nrn.GeInc
	nrn.GeInc = 0

	nrn.GiRaw += nrn.GiInc
	nrn.GiInc = 0
}

// GeFmRaw integrates Ge excitatory conductance from GeRaw value
// (can add other terms to geRaw prior to calling this)
func (ac *ActParams) GeFmRaw(nrn *Neuron, geRaw float32) {
	if !ac.Clamp.Hard && nrn.HasFlag(NeurHasExt) {
		if ac.Clamp.Avg {
			geRaw = ac.Clamp.AvgGe(nrn.Ext, geRaw)
		} else {
			geRaw += nrn.Ext * ac.Clamp.Gain
		}
	}

	ac.Dt.GFmRaw(geRaw, &nrn.Ge)
	// first place noise is required -- generate here!
	if ac.Noise.Type != NoNoise && !ac.Noise.Fixed && ac.Noise.Dist != erand.Mean {
		nrn.Noise = float32(ac.Noise.Gen(-1))
	}
	if ac.Noise.Type == GeNoise {
		nrn.Ge += nrn.Noise
	}
}

// GiFmRaw integrates GiSyn inhibitory synaptic conductance from GiRaw value
// (can add other terms to geRaw prior to calling this)
func (ac *ActParams) GiFmRaw(nrn *Neuron, giRaw float32) {
	ac.Dt.GFmRaw(giRaw, &nrn.GiSyn)
	nrn.GiSyn = math32.Max(nrn.GiSyn, 0) // negative inhib G doesn't make any sense

}

// InetFmG computes net current from conductances and Vm
func (ac *ActParams) InetFmG(vm, ge, gi, gk float32) float32 {
	return ge*(ac.Erev.E-vm) + ac.Gbar.L*(ac.Erev.L-vm) + gi*(ac.Erev.I-vm) + gk*(ac.Erev.K-vm)
}

// VmFmG computes membrane potential Vm from conductances Ge, Gi, and Gk.
// The Vm value is only used in pure rate-code computation within the sub-threshold regime
// because firing rate is a direct function of excitatory conductance Ge.
func (ac *ActParams) VmFmG(nrn *Neuron) {
	ge := nrn.Ge * ac.Gbar.E
	gi := nrn.Gi * ac.Gbar.I
	gk := nrn.Gk * ac.Gbar.K
	nrn.Inet = ac.InetFmG(nrn.Vm, ge, gi, gk)
	nwVm := nrn.Vm + ac.Dt.VmDt*nrn.Inet

	if ac.Noise.Type == VmNoise {
		nwVm += nrn.Noise
	}
	nrn.Vm = ac.VmRange.ClipVal(nwVm)
}

// GeThrFmG computes the threshold for Ge based on all other conductances,
// including Gk.  This is used for computing the adapted Act value.
func (ac *ActParams) GeThrFmG(nrn *Neuron) float32 {
	return ((ac.Gbar.I*nrn.Gi*ac.ErevSubThr.I + ac.Gbar.L*ac.ErevSubThr.L + ac.Gbar.K*nrn.Gk*ac.ErevSubThr.K) / ac.ThrSubErev.E)
}

// GeThrFmGnoK computes the threshold for Ge based on other conductances,
// excluding Gk.  This is used for computing the non-adapted ActLrn value.
func (ac *ActParams) GeThrFmGnoK(nrn *Neuron) float32 {
	return ((ac.Gbar.I*nrn.Gi*ac.ErevSubThr.I + ac.Gbar.L*ac.ErevSubThr.L) / ac.ThrSubErev.E)
}

// ActFmG computes rate-coded activation Act from conductances Ge, Gi, Gk
func (ac *ActParams) ActFmG(nrn *Neuron) {
	if ac.HasHardClamp(nrn) {
		ac.HardClamp(nrn)
		return
	}
	var nwAct, nwActLrn float32
	if nrn.Act < ac.XX1.VmActThr && nrn.Vm <= ac.XX1.Thr {
		// note: this is quite important -- if you directly use the gelin
		// the whole time, then units are active right away -- need Vm dynamics to
		// drive subthreshold activation behavior
		nwAct = ac.XX1.NoisyXX1(nrn.Vm - ac.XX1.Thr)
		nwActLrn = nwAct
	} else {
		ge := nrn.Ge * ac.Gbar.E
		geThr := ac.GeThrFmG(nrn)
		nwAct = ac.XX1.NoisyXX1(ge - geThr)
		geThr = ac.GeThrFmGnoK(nrn)
		nwActLrn = ac.XX1.NoisyXX1(ge - geThr) // learning is non-adapted
	}
	curAct := nrn.Act
	nwAct = curAct + ac.Dt.VmDt*(nwAct-curAct)
	nrn.ActDel = nwAct - curAct

	if ac.Noise.Type == ActNoise {
		nwAct += nrn.Noise
	}
	nrn.Act = nwAct

	nwActLrn = nrn.ActLrn + ac.Dt.VmDt*(nwActLrn-nrn.ActLrn)
	nrn.ActLrn = nwActLrn

	if ac.KNa.On {
		ac.KNa.GcFmRate(&nrn.GknaFast, &nrn.GknaMed, &nrn.GknaSlow, nrn.Act)
		nrn.Gk = nrn.GknaFast + nrn.GknaMed + nrn.GknaSlow
	}
}

// HasHardClamp returns true if this neuron has external input that should be hard clamped
func (ac *ActParams) HasHardClamp(nrn *Neuron) bool {
	return ac.Clamp.Hard && nrn.HasFlag(NeurHasExt)
}

// HardClamp clamps activation from external input -- just does it -- use HasHardClamp to check
// if it should do it.  Also adds any Noise *if* noise is set to ActNoise.
func (ac *ActParams) HardClamp(nrn *Neuron) {
	ext := nrn.Ext
	if ac.Noise.Type == ActNoise {
		ext += nrn.Noise
	}
	clmp := ac.Clamp.Range.ClipVal(ext)
	nrn.Act = clmp + nrn.Noise
	nrn.ActLrn = clmp
	nrn.Vm = ac.XX1.Thr + nrn.Act/ac.XX1.Gain
	nrn.ActDel = 0
	nrn.Inet = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  OptThreshParams

// OptThreshParams provides optimization thresholds for faster processing
type OptThreshParams struct {
	Send  float32 `def:"0.1" desc:"don't send activation when act <= send -- greatly speeds processing"`
	Delta float32 `def:"0.005" desc:"don't send activation changes until they exceed this threshold: only for when LeabraNetwork::send_delta is on!"`
}

func (ot *OptThreshParams) Update() {
}

func (ot *OptThreshParams) Defaults() {
	ot.Send = .1
	ot.Delta = 0.005
}

//////////////////////////////////////////////////////////////////////////////////////
//  ActInitParams

// ActInitParams are initial values for key network state variables.
// Initialized at start of trial with Init_Acts or DecayState.
type ActInitParams struct {
	Decay float32 `def:"0,1" max:"1" min:"0" desc:"proportion to decay activation state toward initial values at start of every trial"`
	Vm    float32 `def:"0.4" desc:"initial membrane potential -- see e_rev.l for the resting potential (typically .3) -- often works better to have a somewhat elevated initial membrane potential relative to that"`
	Act   float32 `def:"0" desc:"initial activation value -- typically 0"`
	Ge    float32 `def:"0" desc:"baseline level of excitatory conductance (net input) -- Ge is initialized to this value, and it is added in as a constant background level of excitatory input -- captures all the other inputs not represented in the model, and intrinsic excitability, etc"`
}

func (ai *ActInitParams) Update() {
}

func (ai *ActInitParams) Defaults() {
	ai.Decay = 1
	ai.Vm = 0.4
	ai.Act = 0
	ai.Ge = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  DtParams

// DtParams are time and rate constants for temporal derivatives in Leabra (Vm, net input)
type DtParams struct {
	Integ  float32 `def:"1,0.5" min:"0" desc:"overall rate constant for numerical integration, for all equations at the unit level -- all time constants are specified in millisecond units, with one cycle = 1 msec -- if you instead want to make one cycle = 2 msec, you can do this globally by setting this integ value to 2 (etc).  However, stability issues will likely arise if you go too high.  For improved numerical stability, you may even need to reduce this value to 0.5 or possibly even lower (typically however this is not necessary).  MUST also coordinate this with network.time_inc variable to ensure that global network.time reflects simulated time accurately"`
	VmTau  float32 `def:"3.3" min:"1" desc:"membrane potential and rate-code activation time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) -- reflects the capacitance of the neuron in principle -- biological default for AdEx spiking model C = 281 pF = 2.81 normalized -- for rate-code activation, this also determines how fast to integrate computed activation values over time"`
	GTau   float32 `def:"1.4,3,5" min:"1" desc:"time constant for integrating synaptic conductances, in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) -- this is important for damping oscillations -- generally reflects time constants associated with synaptic channels which are not modeled in the most abstract rate code models (set to 1 for detailed spiking models with more realistic synaptic currents) -- larger values (e.g., 3) can be important for models with higher conductances that otherwise might be more prone to oscillation."`
	AvgTau float32 `def:"200" desc:"for integrating activation average (ActAvg), time constant in trials (roughly, how long it takes for value to change significantly) -- used mostly for visualization and tracking *hog* units"`

	VmDt  float32 `view:"-" json:"-" xml:"-" desc:"nominal rate = Integ / tau"`
	GDt   float32 `view:"-" json:"-" xml:"-" desc:"rate = Integ / tau"`
	AvgDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (dp *DtParams) Update() {
	dp.VmDt = dp.Integ / dp.VmTau
	dp.GDt = dp.Integ / dp.GTau
	dp.AvgDt = 1 / dp.AvgTau
}

func (dp *DtParams) Defaults() {
	dp.Integ = 1
	dp.VmTau = 3.3
	dp.GTau = 1.4
	dp.AvgTau = 200
	dp.Update()
}

func (dp *DtParams) GFmRaw(geRaw float32, ge *float32) {
	*ge += dp.GDt * (geRaw - *ge)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Noise

// ActNoiseType are different types / locations of random noise for activations
type ActNoiseType int

//go:generate stringer -type=ActNoiseType

var KiT_ActNoiseType = kit.Enums.AddEnum(ActNoiseTypeN, kit.NotBitFlag, nil)

func (ev ActNoiseType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *ActNoiseType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The activation noise types
const (
	// NoNoise means no noise added
	NoNoise ActNoiseType = iota

	// VmNoise means noise is added to the membrane potential.
	// IMPORTANT: this should NOT be used for rate-code (NXX1) activations,
	// because they do not depend directly on the vm -- this then has no effect
	VmNoise

	// GeNoise means noise is added to the excitatory conductance (Ge).
	// This should be used for rate coded activations (NXX1)
	GeNoise

	// ActNoise means noise is added to the final rate code activation
	ActNoise

	// GeMultNoise means that noise is multiplicative on the Ge excitatory conductance values
	GeMultNoise

	ActNoiseTypeN
)

// ActNoiseParams contains parameters for activation-level noise
type ActNoiseParams struct {
	erand.RndParams
	Type  ActNoiseType `desc:"where and how to add processing noise"`
	Fixed bool         `desc:"keep the same noise value over the entire alpha cycle -- prevents noise from being washed out and produces a stable effect that can be better used for learning -- this is strongly recommended for most learning situations"`
}

func (an *ActNoiseParams) Update() {
}

func (an *ActNoiseParams) Defaults() {
	an.Fixed = true
}

//////////////////////////////////////////////////////////////////////////////////////
//  ClampParams

// ClampParams are for specifying how external inputs are clamped onto network activation values
type ClampParams struct {
	Hard    bool       `def:"true" desc:"whether to hard clamp inputs where activation is directly set to external input value (Act = Ext) or do soft clamping where Ext is added into Ge excitatory current (Ge += Gain * Ext)"`
	Range   minmax.F32 `viewif:"Hard" desc:"range of external input activation values allowed -- Max is .95 by default due to saturating nature of rate code activation function"`
	Gain    float32    `viewif:"!Hard" def:"0.02:0.5" desc:"soft clamp gain factor (Ge += Gain * Ext)"`
	Avg     bool       `viewif:"!Hard" desc:"compute soft clamp as the average of current and target netins, not the sum -- prevents some of the main effect problems associated with adding external inputs"`
	AvgGain float32    `viewif:"!Hard && Avg" def:"0.2" desc:"gain factor for averaging the Ge -- clamp value Ext contributes with AvgGain and current Ge as (1-AvgGain)"`
}

func (cp *ClampParams) Update() {
}

func (cp *ClampParams) Defaults() {
	cp.Hard = true
	cp.Range.Max = 0.95
	cp.Gain = 0.2
	cp.Avg = false
	cp.AvgGain = 0.2
}

// AvgGe computes Avg-based Ge clamping value if using that option.
func (cp *ClampParams) AvgGe(ext, ge float32) float32 {
	return cp.AvgGain*cp.Gain*ext + (1-cp.AvgGain)*ge
}

//////////////////////////////////////////////////////////////////////////////////////
//  WtInitParams

// WtInitParams are weight initialization parameters -- basically the
// random distribution parameters but also Symmetry flag
type WtInitParams struct {
	erand.RndParams
	Sym bool `desc:"symmetrize the weight values with those in reciprocal projection -- typically true for bidirectional excitatory connections"`
}

func (wp *WtInitParams) Defaults() {
	wp.Mean = 0.5
	wp.Var = 0.25
	wp.Dist = erand.Uniform
	wp.Sym = true
}

//////////////////////////////////////////////////////////////////////////////////////
//  WtScaleParams

/// WtScaleParams are weight scaling parameters: modulates overall strength of projection,
// using both absolute and relative factors
type WtScaleParams struct {
	Abs float32 `def:"1" min:"0" desc:"absolute scaling, which is not subject to normalization: directly multiplies weight values"`
	Rel float32 `min:"0" desc:"[Default: 1] relative scaling that shifts balance between different projections -- this is subject to normalization across all other projections into unit"`
}

func (ws *WtScaleParams) Defaults() {
	ws.Abs = 1
	ws.Rel = 1
}

func (ws *WtScaleParams) Update() {
}

// SLayActScale computes scaling factor based on sending layer activity level (savg), number of units
// in sending layer (snu), and number of recv connections (ncon).
// Uses a fixed sem_extra standard-error-of-the-mean (SEM) extra value of 2
// to add to the average expected number of active connections to receive,
// for purposes of computing scaling factors with partial connectivity
// For 25% layer activity, binomial SEM = sqrt(p(1-p)) = .43, so 3x = 1.3 so 2 is a reasonable default.
func (ws *WtScaleParams) SLayActScale(savg, snu, ncon float32) float32 {
	ncon = math32.Max(ncon, 1) // prjn Avg can be < 1 in some cases
	semExtra := 2
	slayActN := int(mat32.Round(savg * snu)) // sending layer actual # active
	slayActN = ints.MaxInt(slayActN, 1)
	var sc float32
	if ncon == snu {
		sc = 1 / float32(slayActN)
	} else {
		rMaxActN := int(math32.Min(ncon, float32(slayActN))) // max number we could get
		rAvgActN := int(mat32.Round(savg * ncon))            // recv average actual # active if uniform
		rAvgActN = ints.MaxInt(rAvgActN, 1)
		rExpActN := rAvgActN + semExtra // expected
		rExpActN = ints.MinInt(rExpActN, rMaxActN)
		sc = 1 / float32(rExpActN)
	}
	return sc
}

// FullScale returns full scaling factor, which is product of Abs * Rel * SLayActScale
func (ws *WtScaleParams) FullScale(savg, snu, ncon float32) float32 {
	return ws.Abs * ws.Rel * ws.SLayActScale(savg, snu, ncon)
}
