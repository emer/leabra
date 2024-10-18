// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"

	"cogentcore.org/core/math32"
)

// MatrixParams has parameters for Dorsal Striatum Matrix computation.
// These are the main Go / NoGo gating units in BG driving updating of PFC WM in PBWM.
type MatrixParams struct {

	// Quarter(s) when learning takes place, typically Q2 and Q4, corresponding to the PFC GateQtr. Note: this is a bitflag and must be accessed using bitflag.Set / Has etc routines, 32 bit versions.
	LearnQtr Quarters

	// how much the patch shunt activation multiplies the dopamine values -- 0 = complete shunting, 1 = no shunting -- should be a factor < 1.0
	PatchShunt float32 `default:"0.2,0.5" min:"0" max:"1"`

	// also shunt the ACh value driven from CIN units -- this prevents clearing of MSNConSpec traces -- more plausibly the patch units directly interfere with the effects of CIN's rather than through ach, but it is easier to implement with ach shunting here.
	ShuntACh bool `default:"true"`

	// how much does the LACK of ACh from the CIN units drive extra inhibition to output-gating Matrix units -- gi += out_ach_inhib * (1-ach) -- provides a bias for output gating on reward trials -- do NOT apply to NoGo, only Go -- this is a key param -- between 0.1-0.3 usu good -- see how much output gating happening and change accordingly
	OutAChInhib float32 `default:"0,0.3"`

	// multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)
	BurstGain float32 `default:"1"`

	// multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)
	DipGain float32 `default:"1"`
}

func (mp *MatrixParams) Defaults() {
	mp.LearnQtr.SetFlag(true, Q2)
	mp.LearnQtr.SetFlag(true, Q4)
	mp.PatchShunt = 0.2
	mp.ShuntACh = true
	mp.OutAChInhib = 0.3
	mp.BurstGain = 1
	mp.DipGain = 1
}

func (mp *MatrixParams) Update() {
}

func (ly *Layer) MatrixDefaults() {
	// special inhib params
	ly.PBWM.Type = MaintOut
	ly.Inhib.Layer.Gi = 1.9
	ly.Inhib.Layer.FB = 0.5
	ly.Inhib.Pool.On = true
	ly.Inhib.Pool.Gi = 1.9
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.3
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 0.2
}

// DALrnFromDA returns effective learning dopamine value from given raw DA value
// applying Burst and Dip Gain factors, and then reversing sign for D2R.
func (ly *Layer) DALrnFromDA(da float32) float32 {
	if da > 0 {
		da *= ly.Matrix.BurstGain
	} else {
		da *= ly.Matrix.DipGain
	}
	if ly.PBWM.DaR == D2R {
		da *= -1
	}
	return da
}

// MatrixOutAChInhib applies OutAChInhib to bias output gating on reward trials.
func (ly *Layer) MatrixOutAChInhib(ctx *Context) {
	if ly.Matrix.OutAChInhib == 0 {
		return
	}

	ypN := ly.Shape.DimSize(0)
	xpN := ly.Shape.DimSize(1)
	ynN := ly.Shape.DimSize(2)
	xnN := ly.Shape.DimSize(3)
	maintN := ly.PBWM.MaintN
	layAch := ly.NeuroMod.ACh // ACh comes from CIN neurons, represents reward time
	for yp := 0; yp < ypN; yp++ {
		for xp := maintN; xp < xpN; xp++ {
			for yn := 0; yn < ynN; yn++ {
				for xn := 0; xn < xnN; xn++ {
					ni := ly.Shape.Offset([]int{yp, xp, yn, xn})
					nrn := &ly.Neurons[ni]
					if nrn.IsOff() {
						continue
					}
					ach := layAch
					if ly.Matrix.ShuntACh && nrn.Shunt > 0 {
						ach *= ly.Matrix.PatchShunt
					}
					achI := ly.Matrix.OutAChInhib * (1 - ach)
					nrn.Gi += achI
				}
			}
		}
	}
}

// DaAChFromLay computes Da and ACh from layer and Shunt received from PatchLayer units
func (ly *Layer) DaAChFromLay(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		da := ly.NeuroMod.DA
		if nrn.Shunt > 0 { // note: treating Shunt as binary variable -- could multiply
			da *= ly.Matrix.PatchShunt
		}
		nrn.DALrn = ly.DALrnFromDA(da)
	}
}

// RecGateAct records the gating activation from current activation, when gating occcurs
// based on GateState.Now
func (ly *Layer) RecGateAct(ctx *Context) {
	for pi := range ly.Pools {
		if pi == 0 {
			continue
		}
		pl := &ly.Pools[pi]
		if !pl.Gate.Now { // not gating now
			continue
		}
		for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			nrn.ActG = nrn.Act
		}
	}
}

// GateTypes for region of striatum
type GateTypes int32 //enums:enum

const (
	// Maint is maintenance gating -- toggles active maintenance in PFC.
	Maint GateTypes = iota

	// Out is output gating -- drives deep layer activation.
	Out

	// MaintOut for maint and output gating.
	MaintOut
)

// SendToMatrixPFC adds standard SendTo layers for PBWM: MatrixGo, NoGo, PFCmntD, PFCoutD
// with optional prefix -- excludes mnt, out cases if corresp shape = 0
func (ly *Layer) SendToMatrixPFC(prefix string) {
	pfcprefix := "PFC"
	if prefix != "" {
		pfcprefix = prefix
	}
	std := []string{prefix + "MatrixGo", prefix + "MatrixNoGo", pfcprefix + "mntD", pfcprefix + "outD"}
	ly.SendTo = make([]string, 2)
	for i, s := range std {
		nm := s
		switch {
		case i < 2:
			ly.SendTo[i] = nm
		case i == 2:
			if ly.PBWM.MaintX > 0 {
				ly.SendTo = append(ly.SendTo, nm)
			}
		case i == 3:
			if ly.PBWM.OutX > 0 {
				ly.SendTo = append(ly.SendTo, nm)
			}
		}
	}
}

// SendPBWMParams send PBWMParams info to all SendTo layers -- convenient config-time
// way to ensure all are consistent -- also checks validity of SendTo's
func (ly *Layer) SendPBWMParams() error {
	var lasterr error
	for _, lnm := range ly.SendTo {
		tly := ly.Network.LayerByName(lnm)
		tly.PBWM.CopyGeomFrom(&ly.PBWM)
	}
	return lasterr
}

// MatrixPaths returns the recv paths from Go and NoGo MatrixLayer pathways -- error if not
// found or if paths are not of the GPiThalPath type
func (ly *Layer) MatrixPaths() (goPath, nogoPath *Path, err error) {
	for _, p := range ly.RecvPaths {
		if p.Off {
			continue
		}
		slay := p.Send
		if slay.Type == MatrixLayer {
			if ly.PBWM.DaR == D1R {
				goPath = p
			} else {
				nogoPath = p
			}
		} else {
			nogoPath = p
		}
	}
	if goPath == nil || nogoPath == nil {
		err = fmt.Errorf("GPiThalLayer must have RecvPath's from a MatrixLayer D1R (Go) and another NoGo layer")
	}
	return
}

// PBWMParams defines the shape of the outer pool dimensions of gating layers,
// organized into Maint and Out subsets which are arrayed along the X axis
// with Maint first (to the left) then Out.  Individual layers may only
// represent Maint or Out subsets of this overall shape, but all need
// to have this coordinated shape information to be able to share gating
// state information.  Each layer represents gate state information in
// their native geometry -- FullIndex1D provides access from a subset
// to full set.
type PBWMParams struct {
	// Type of gating layer
	Type GateTypes

	// dominant type of dopamine receptor -- D1R for Go pathway, D2R for NoGo
	DaR DaReceptors

	// overall shape dimensions for the full set of gating pools,
	// e.g., as present in the Matrix and GPiThal levels
	Y int

	// how many pools in the X dimension are Maint gating pools -- rest are Out
	MaintX int

	// how many pools in the X dimension are Out gating pools -- comes after Maint
	OutX int

	// For the Matrix layers, this is the number of Maint Pools in X outer
	// dimension of 4D shape -- Out gating after that. Note: it is unclear
	// how this relates to MaintX, but it is different in SIR model.
	MaintN int
}

func (pp *PBWMParams) Defaults() {

}

func (pp *PBWMParams) Update() {

}

// Set sets the shape parameters: number of Y dimension pools, and
// numbers of maint and out pools along X axis
func (pp *PBWMParams) Set(nY, maintX, outX int) {
	pp.Y = nY
	pp.MaintX = maintX
	pp.OutX = outX
}

// TotX returns the total number of X-axis pools (Maint + Out)
func (pp *PBWMParams) TotX() int {
	return pp.MaintX + pp.OutX
}

func (pp *PBWMParams) CopyGeomFrom(src *PBWMParams) {
	pp.Set(src.Y, src.MaintX, src.OutX)
	pp.Type = src.Type
}

// Index returns the index into GateStates for given 2D pool coords
// for given GateType.  Each type stores gate info in its "native" 2D format.
func (pp *PBWMParams) Index(pY, pX int, typ GateTypes) int {
	switch typ {
	case Maint:
		if pp.MaintX == 0 {
			return 0
		}
		return pY*pp.MaintX + pX
	case Out:
		if pp.OutX == 0 {
			return 0
		}
		return pY*pp.OutX + pX
	case MaintOut:
		return pY*pp.TotX() + pX
	}
	return 0
}

// FullIndex1D returns the index into full MaintOut GateStates
// for given 1D pool idx (0-based) *from given GateType*.
func (pp *PBWMParams) FullIndex1D(idx int, fmTyp GateTypes) int {
	switch fmTyp {
	case Maint:
		if pp.MaintX == 0 {
			return 0
		}
		// convert to 2D and use that
		pY := idx / pp.MaintX
		pX := idx % pp.MaintX
		return pp.Index(pY, pX, MaintOut)
	case Out:
		if pp.OutX == 0 {
			return 0
		}
		// convert to 2D and use that
		pY := idx / pp.OutX
		pX := idx%pp.OutX + pp.MaintX
		return pp.Index(pY, pX, MaintOut)
	case MaintOut:
		return idx
	}
	return 0
}

//////// GateState

// GateState is gating state values stored in layers that receive thalamic gating signals
// including MatrixLayer, PFCLayer, GPiThal layer, etc -- use GateLayer as base layer to include.
type GateState struct {

	// gating activation value, reflecting thalamic gating layer activation at time of gating (when Now = true) -- will be 0 if gating below threshold for this pool, and prior to first Now for AlphaCycle
	Act float32

	// gating timing signal -- true if this is the moment when gating takes place
	Now bool

	// unique to each layer -- not copied.  Generally is a counter for interval between gating signals -- starts at -1, goes to 0 at first gating, counts up from there for subsequent gating.  Can be reset back to -1 when gate is reset (e.g., output gating) and counts down from -1 while not gating.
	Cnt int `copy:"-"`
}

// Init initializes the values -- call during InitActs()
func (gs *GateState) Init() {
	gs.Act = 0
	gs.Now = false
	gs.Cnt = -1
}

// CopyFrom copies from another GateState -- only the Act and Now signals are copied
func (gs *GateState) CopyFrom(fm *GateState) {
	gs.Act = fm.Act
	gs.Now = fm.Now
}

// GateType returns type of gating for this layer
func (ly *Layer) GateType() GateTypes {
	switch ly.Type {
	case GPiThalLayer, MatrixLayer:
		return MaintOut
	case PFCDeepLayer:
		if ly.PFCGate.OutGate {
			return Out
		}
		return Maint
	}
	return MaintOut
}

// SetGateStates sets the GateStates from given source states, of given gating type
func (ly *Layer) SetGateStates(src *Layer, typ GateTypes) {
	myt := ly.GateType()
	if myt < MaintOut && typ < MaintOut && myt != typ { // mismatch
		return
	}
	switch {
	case myt == typ:
		mx := min(len(src.Pools), len(ly.Pools))
		for i := 1; i < mx; i++ {
			ly.Pool(i).Gate.CopyFrom(&src.Pool(i).Gate)
		}
	default: // typ == MaintOut, myt = Maint or Out
		mx := len(ly.Pools)
		for i := 1; i < mx; i++ {
			gs := &ly.Pool(i).Gate
			si := 1 + ly.PBWM.FullIndex1D(i-1, myt)
			sgs := &src.Pool(si).Gate
			gs.CopyFrom(sgs)
		}
	}
}

//////// GPiThalLayer

// GPiGateParams has gating parameters for gating in GPiThal layer, including threshold.
type GPiGateParams struct {
	// GateQtr is the Quarter(s) when gating takes place, typically Q1 and Q3,
	// which is the quarter prior to the PFC GateQtr when deep layer updating
	// takes place. Note: this is a bitflag and must be accessed using bitflag.
	// Set / Has etc routines, 32 bit versions.
	GateQtr Quarters

	// Cycle within Qtr to determine if activation over threshold for gating.
	// We send GateState updates on this cycle either way.
	Cycle int `default:"18"`

	// extra netinput gain factor to compensate for reduction in Ge from subtracting away NoGo -- this is *IN ADDITION* to adding the NoGo factor as an extra gain: Ge = (GeGain + NoGo) * (GoIn - NoGo * NoGoIn)
	GeGain float32 `default:"3"`

	// how much to weight NoGo inputs relative to Go inputs (which have an implied weight of 1 -- this also up-scales overall Ge to compensate for subtraction
	NoGo float32 `min:"0" default:"1,0.1"`

	// threshold for gating, applied to activation -- when any GPiThal unit activation gets above this threshold, it counts as having gated, driving updating of GateState which is broadcast to other layers that use the gating signal
	Thr float32 `default:"0.2"`

	// Act value of GPiThal unit reflects gating threshold: if below threshold, it is zeroed -- see ActLrn for underlying non-thresholded activation
	ThrAct bool `default:"true"`
}

func (gp *GPiGateParams) Defaults() {
	gp.GateQtr.SetFlag(true, Q1)
	gp.GateQtr.SetFlag(true, Q3)
	gp.Cycle = 18
	gp.GeGain = 3
	gp.NoGo = 1
	gp.Thr = 0.2
	gp.ThrAct = true
}

func (gp *GPiGateParams) Update() {
}

// GeRaw returns the net GeRaw from go, nogo specific values
func (gp *GPiGateParams) GeRaw(goRaw, nogoRaw float32) float32 {
	return (gp.GeGain + gp.NoGo) * (goRaw - gp.NoGo*nogoRaw)
}

func (ly *Layer) GPiThalDefaults() {
	ly.PBWM.Type = MaintOut
	ly.Inhib.Layer.Gi = 1.8
	ly.Inhib.Layer.FB = 0.2
	ly.Inhib.Pool.On = false
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 1
}

// GPiGFromInc integrates new synaptic conductances from increments
// sent during last SendGDelta.
func (ly *Layer) GPiGFromInc(ctx *Context) {
	goPath, nogoPath, _ := ly.MatrixPaths()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		goRaw := goPath.GeRaw[ni]
		nogoRaw := nogoPath.GeRaw[ni]
		nrn.GeRaw = ly.GPiGate.GeRaw(goRaw, nogoRaw)
		ly.Act.GeFromRaw(nrn, nrn.GeRaw)
		ly.Act.GiFromRaw(nrn, nrn.GiRaw)
	}
}

// GPiGateSend updates gating state and sends it along to other layers
func (ly *Layer) GPiGateSend(ctx *Context) {
	ly.GPiGateFromAct(ctx)
	ly.GPiSendGateStates()
}

// GPiGateFromAct updates GateState from current activations, at time of gating
func (ly *Layer) GPiGateFromAct(ctx *Context) {
	gateQtr := ly.GPiGate.GateQtr.HasFlag(ctx.Quarter)
	qtrCyc := ctx.QuarterCycle()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		gs := &ly.Pool(int(nrn.SubPool)).Gate
		if ctx.Quarter == 0 && qtrCyc == 0 {
			gs.Act = 0 // reset at start
		}
		if gateQtr && qtrCyc == ly.GPiGate.Cycle { // gating
			gs.Now = true
			if nrn.Act < ly.GPiGate.Thr { // didn't gate
				gs.Act = 0 // not over thr
				if ly.GPiGate.ThrAct {
					gs.Act = 0
				}
				if gs.Cnt >= 0 {
					gs.Cnt++
				} else if gs.Cnt < 0 {
					gs.Cnt--
				}
			} else { // did gate
				gs.Cnt = 0
				gs.Act = nrn.Act
			}
		} else {
			gs.Now = false
		}
	}
}

// GPiSendGateStates sends GateStates to other layers
func (ly *Layer) GPiSendGateStates() {
	myt := MaintOut // always
	for _, lnm := range ly.SendTo {
		gl := ly.Network.LayerByName(lnm)
		gl.SetGateStates(ly, myt)
	}
}

//////// CINLayer

// CINParams (cholinergic interneuron) reads reward signals from named source layer(s)
// and sends the Max absolute value of that activity as the positively rectified
// non-prediction-discounted reward signal computed by CINs, and sent as
// an acetylcholine (ACh) signal.
// To handle positive-only reward signals, need to include both a reward prediction
// and reward outcome layer.
type CINParams struct {
	// RewThr is the threshold on reward values from RewLays,
	// to count as a significant reward event, which then drives maximal ACh.
	// Set to 0 to disable this nonlinear behavior.
	RewThr float32 `default:"0.1"`

	// Reward-representing layer(s) from which this computes ACh as Max absolute value
	RewLays LayerNames
}

func (ly *CINParams) Defaults() {
	ly.RewThr = 0.1
}

func (ly *CINParams) Update() {
}

// CINMaxAbsRew returns the maximum absolute value of reward layer activations.
func (ly *Layer) CINMaxAbsRew() float32 {
	mx := float32(0)
	for _, nm := range ly.CIN.RewLays {
		ly := ly.Network.LayerByName(nm)
		if ly == nil {
			continue
		}
		act := math32.Abs(ly.Pools[0].Inhib.Act.Max)
		mx = math32.Max(mx, act)
	}
	return mx
}

func (ly *Layer) ActFromGCIN(ctx *Context) {
	ract := ly.CINMaxAbsRew()
	if ly.CIN.RewThr > 0 {
		if ract > ly.CIN.RewThr {
			ract = 1
		}
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = ract
		ly.Learn.AvgsFromAct(nrn)
	}
}

// SendAChFromAct sends ACh from neural activity in first unit.
func (ly *Layer) SendAChFromAct(ctx *Context) {
	act := ly.Neurons[0].Act
	ly.NeuroMod.ACh = act
	ly.SendACh(act)
}

//////// PFC

// PFCGateParams has parameters for PFC gating
type PFCGateParams struct {

	// Quarter(s) that the effect of gating on updating Deep from Super occurs -- this is typically 1 quarter after the GPiThal GateQtr
	GateQtr Quarters

	// if true, this PFC layer is an output gate layer, which means that it only has transient activation during gating
	OutGate bool

	// for output gating, only compute gating in first quarter -- do not compute in 3rd quarter -- this is typically true, and GateQtr is typically set to only Q1 as well -- does Burst updating immediately after first quarter gating signal -- allows gating signals time to influence performance within a single trial
	OutQ1Only bool `viewif:"OutGate" default:"true"`
}

func (gp *PFCGateParams) Defaults() {
	gp.GateQtr.SetFlag(true, Q2)
	gp.GateQtr.SetFlag(true, Q4)
	gp.OutQ1Only = true
}

func (gp *PFCGateParams) Update() {
}

// PFCMaintParams for PFC maintenance functions
type PFCMaintParams struct {

	// use fixed dynamics for updating deep_ctxt activations -- defined in dyn_table -- this also preserves the initial gating deep_ctxt value in Maint neuron val (view as Cust1) -- otherwise it is up to the recurrent loops between super and deep for maintenance
	UseDyn bool

	// multiplier on maint current
	MaintGain float32 `min:"0" default:"0.8"`

	// on output gating, clear corresponding maint pool.  theoretically this should be on, but actually it works better off in most cases..
	OutClearMaint bool `default:"false"`

	// how much to clear out (decay) super activations when the stripe itself gates and was previously maintaining something, or for maint pfc stripes, when output go fires and clears.
	Clear    float32 `min:"0" max:"1" default:"0"`
	MaxMaint int     `"min:"1" default:"1:100" maximum duration of maintenance for any stripe -- beyond this limit, the maintenance is just automatically cleared -- typically 1 for output gating and 100 for maintenance gating"`
}

func (mp *PFCMaintParams) Defaults() {
	mp.MaintGain = 0.8
	mp.OutClearMaint = false // theoretically should be true, but actually was false due to bug
	mp.Clear = 0
	mp.MaxMaint = 100
}

func (mp *PFCMaintParams) Update() {
}

// PFC dynamic behavior element -- defines the dynamic behavior of deep layer PFC units
type PFCDyn struct {

	// initial value at point when gating starts -- MUST be > 0 when used.
	Init float32

	// time constant for linear rise in maintenance activation (per quarter when deep is updated) -- use integers -- if both rise and decay then rise comes first
	RiseTau float32

	// time constant for linear decay in maintenance activation (per quarter when deep is updated) -- use integers -- if both rise and decay then rise comes first
	DecayTau float32

	// description of this factor
	Desc string
}

func (pd *PFCDyn) Defaults() {
	pd.Init = 1
}

func (pd *PFCDyn) Set(init, rise, decay float32, desc string) {
	pd.Init = init
	pd.RiseTau = rise
	pd.DecayTau = decay
	pd.Desc = desc
}

// Value returns dynamic value at given time point
func (pd *PFCDyn) Value(time float32) float32 {
	val := pd.Init
	if time <= 0 {
		return val
	}
	if pd.RiseTau > 0 && pd.DecayTau > 0 {
		if time >= pd.RiseTau {
			val = 1 - ((time - pd.RiseTau) / pd.DecayTau)
		} else {
			val = pd.Init + (1-pd.Init)*(time/pd.RiseTau)
		}
	} else if pd.RiseTau > 0 {
		val = pd.Init + (1-pd.Init)*(time/pd.RiseTau)
	} else if pd.DecayTau > 0 {
		val = pd.Init - pd.Init*(time/pd.DecayTau)
	}
	if val > 1 {
		val = 1
	}
	if val < 0.001 {
		val = 0.001
	}
	return val
}

//////////////////////////////////////////////////////////////////////////////
//  PFCDyns

// PFCDyns is a slice of dyns. Provides deterministic control over PFC
// maintenance dynamics -- the rows of PFC units (along Y axis) behave
// according to corresponding index of Dyns.
// ensure layer Y dim has even multiple of len(Dyns).
type PFCDyns []*PFCDyn

// SetDyn sets given dynamic maint element to given parameters (must be allocated in list first)
func (pd *PFCDyns) SetDyn(dyn int, init, rise, decay float32, desc string) *PFCDyn {
	dy := &PFCDyn{}
	dy.Set(init, rise, decay, desc)
	(*pd)[dyn] = dy
	return dy
}

// MaintOnly creates basic default maintenance dynamic configuration -- every
// unit just maintains over time.
// This should be used for Output gating layer.
func (pd *PFCDyns) MaintOnly() {
	*pd = make([]*PFCDyn, 1)
	pd.SetDyn(0, 1, 0, 0, "maintain stable act")
}

// FullDyn creates full dynamic Dyn configuration, with 5 different
// dynamic profiles: stable maint, phasic, rising maint, decaying maint,
// and up / down maint.  tau is the rise / decay base time constant.
func (pd *PFCDyns) FullDyn(tau float32) {
	ndyn := 5
	*pd = make([]*PFCDyn, ndyn)

	pd.SetDyn(0, 1, 0, 0, "maintain stable act")
	pd.SetDyn(1, 1, 0, 1, "immediate phasic response")
	pd.SetDyn(2, .1, tau, 0, "maintained, rising value over time")
	pd.SetDyn(3, 1, 0, tau, "maintained, decaying value over time")
	pd.SetDyn(4, .1, .5*tau, tau, "maintained, rising then falling over time")
}

// Value returns value for given dyn item at given time step
func (pd *PFCDyns) Value(dyn int, time float32) float32 {
	sz := len(*pd)
	if sz == 0 {
		return 1
	}
	dy := (*pd)[dyn%sz]
	return dy.Value(time)
}

func (ly *Layer) PFCDeepDefaults() {
	if ly.PFCGate.OutGate && ly.PFCGate.OutQ1Only {
		ly.PFCMaint.MaxMaint = 1
		ly.PFCGate.GateQtr = 0
		ly.PFCGate.GateQtr.SetFlag(true, Q1)
	}
	if len(ly.PFCDyns) > 0 {
		ly.PFCMaint.UseDyn = true
	} else {
		ly.PFCMaint.UseDyn = false
	}
}

// MaintPFC returns corresponding PFCDeep maintenance layer
// with same name but outD -> mntD; could be nil
func (ly *Layer) MaintPFC() *Layer {
	sz := len(ly.Name)
	mnm := ly.Name[:sz-4] + "mntD"
	li := ly.Network.LayerByName(mnm)
	return li
}

// SuperPFC returns corresponding PFC super layer with same name without D
// should not be nil.  Super can be any layer type.
func (ly *Layer) SuperPFC() *Layer {
	dnm := ly.Name[:len(ly.Name)-1]
	li := ly.Network.LayerByName(dnm)
	return li
}

// MaintGInc increments Ge from MaintGe, for PFCDeepLayer.
func (ly *Layer) MaintGInc(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		geRaw := nrn.GeRaw + nrn.MaintGe
		ly.Act.GeFromRaw(nrn, geRaw)
		ly.Act.GiFromRaw(nrn, nrn.GiRaw)
	}
}

// PFCDeepGating updates PFC Gating state.
func (ly *Layer) PFCDeepGating(ctx *Context) {
	if ly.PFCGate.OutGate && ly.PFCGate.OutQ1Only {
		if ctx.Quarter > 1 {
			return
		}
	}

	for pi := range ly.Pools {
		if pi == 0 {
			continue
		}
		gs := &ly.Pools[pi].Gate
		if !gs.Now { // not gating now
			continue
		}
		if gs.Act > 0 { // use GPiThal threshold, so anything > 0
			gs.Cnt = 0              // this is the "just gated" signal
			if ly.PFCGate.OutGate { // time to clear out maint
				if ly.PFCMaint.OutClearMaint {
					fmt.Println("clear maint")
					ly.ClearMaint(pi)
				}
			} else {
				pfcs := ly.SuperPFC()
				pfcs.DecayStatePool(pi, ly.PFCMaint.Clear)
			}
		}
		// test for over-duration maintenance -- allow for active gating to override
		if gs.Cnt >= ly.PFCMaint.MaxMaint {
			gs.Cnt = -1
		}
	}
}

// ClearMaint resets maintenance in corresponding pool (0 based) in maintenance layer
func (ly *Layer) ClearMaint(pool int) {
	pfcm := ly.MaintPFC()
	if pfcm == nil {
		return
	}
	gs := &pfcm.Pools[pool].Gate
	if gs.Cnt >= 1 { // important: only for established maint, not just gated..
		gs.Cnt = -1 // reset
		pfcs := pfcm.SuperPFC()
		pfcs.DecayStatePool(pool, pfcm.PFCMaint.Clear)
	}
}

// DeepMaint updates deep maintenance activations
func (ly *Layer) DeepMaint(ctx *Context) {
	if !ly.PFCGate.GateQtr.HasFlag(ctx.Quarter) {
		return
	}
	sly := ly.SuperPFC()
	if sly == nil {
		return
	}
	yN := ly.Shape.DimSize(2)
	xN := ly.Shape.DimSize(3)

	nn := yN * xN

	syN := sly.Shape.DimSize(2)
	sxN := sly.Shape.DimSize(3)
	snn := syN * sxN

	dper := yN / syN  // dyn per sender -- should be len(Dyns)
	dtyp := yN / dper // dyn type

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ui := ni % nn
		pi := ni / nn
		uy := ui / xN
		ux := ui % xN

		gs := &ly.Pool(int(nrn.SubPool)).Gate
		if gs.Cnt < 0 {
			nrn.Maint = 0
			nrn.MaintGe = 0
		} else if gs.Cnt <= 1 { // first gating, save first gating value
			sy := uy % syN // inner loop is s
			si := pi*snn + sy*sxN + ux
			snr := &sly.Neurons[si]
			nrn.Maint = ly.PFCMaint.MaintGain * snr.Act
		}
		if ly.PFCMaint.UseDyn {
			nrn.MaintGe = nrn.Maint * ly.PFCDyns.Value(dtyp, float32(gs.Cnt-1))
		} else {
			nrn.MaintGe = nrn.Maint
		}
	}
}

// UpdateGateCnt updates the gate counter
func (ly *Layer) UpdateGateCnt(ctx *Context) {
	if !ly.PFCGate.GateQtr.HasFlag(ctx.Quarter) {
		return
	}
	for pi := range ly.Pools {
		if pi == 0 {
			continue
		}
		gs := &ly.Pools[pi].Gate
		if gs.Cnt < 0 {
			// ly.ClearCtxtPool(gi)
			gs.Cnt--
		} else {
			gs.Cnt++
		}
	}
}
