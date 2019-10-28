// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/chewxy/math32"
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
)

// PFCGateParams has parameters for PFC gating
type PFCGateParams struct {
	OutGate   bool    `desc:"if true, this PFC layer is an output gate layer, which means that it only has transient activation during gating"`
	OutQ1Only bool    `viewif:"OutGate" def:"true" desc:"for output gating, only compute gating in first quarter -- do not compute in 3rd quarter -- this is typically true, and BurstQtr is typically set to only Q1 as well -- does Burst updating immediately after first quarter gating signal -- allows gating signals time to influence performance within a single trial"`
	MntThal   float32 `def:"1" desc:"effective Thal activation to use in computing the Burst activation sent from Super to Deep layers, for continued maintenance beyond the initial Thal signal provided by the BG -- also sets an effective minimum Thal value regardless of the actual gating thal value"`
	GateThr   float32 `def:0.2" desc:"threshold on thalamic gating signal to drive gating -- when using GpiInvUnitSpec gpi, this parameter ususally doesn't matter!  set the gpi.gate_thr value instead -- the only constraint is that this value be <= gpi.min_thal as that determines the minimum thalamic value for gated stripes"`
}

func (gp *PFCGateParams) Defaults() {
	gp.OutQ1Only = true
	gp.MntThal = 1
	gp.GateThr = 0.2
}

// PFCMaintParams for PFC maintenance functions
type PFCMaintParams struct {
	SMnt     minmax.F32 `desc:"default 0.3..0.5 -- for superficial neurons, how much of DeepLrn to add into Ge input to support maintenance, from deep maintenance signal -- 0.25 is generally minimum to support maintenance"`
	MntGeMax float32    `def:"0.5" desc:"maximum GeRaw.Max value required to drive the minimum sMnt.Min maintenance current from deep -- anything above this drives the same SMnt.Min value -- below this value scales the effective mnt current between SMnt.Min to .Max in reverse proportion to GeRaw.Max value"`
	Clear    float32    `"min:"0" max:"1" def:"0.5" desc:"how much to clear out (decay) super activations when the stripe itself gates and was previously maintaining something, or for maint pfc stripes, when output go fires and clears"`
	UseDyn   bool       `desc:"use fixed dynamics for updating deep_ctxt activations -- defined in dyn_table -- this also preserves the initial gating deep_ctxt value in misc_1 -- otherwise it is up to the recurrent loops between super and deep for maintenance"`
	MaxMaint int        `"min:"1" def:"1:100" maximum duration of maintenance for any stripe -- beyond this limit, the maintenance is just automatically cleared -- typically 1 for output gating and 100 for maintenance gating"`
}

func (mp *PFCMaintParams) Defaults() {
	mp.SMnt.Set(0.3, 0.5)
	mp.MntGeMax = 0.5
	mp.Clear = 0.5
	mp.UseDyn = true
	mp.MaxMaint = 100
}

// PFC dynamic behavior element -- defines the dynamic behavior of deep layer PFC units
type PFCDyn struct {
	Init     float32 `desc:"initial value at point when gating starts -- MUST be > 0 when used."`
	RiseTau  float32 `desc:"time constant for linear rise in maintenance activation (per quarter when deep is updated) -- use integers -- if both rise and decay then rise comes first"`
	DecayTau float32 `desc:"time constant for linear decay in maintenance activation (per quarter when deep is updated) -- use integers -- if both rise and decay then rise comes first"`
	Desc     string  `desc:"description of this factor"`
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

///////////////////////////////////////////////////////////////////
// PFCLayer

// PFCNeuron contains extra variables for PFCLayer neurons -- stored separately
type PFCNeuron struct {
	ActG float32 `desc:"gating activation -- the activity value when gating occurred in this pool"`
}

// PFCLayer is a Prefrontal Cortex BG-gated working memory layer
type PFCLayer struct {
	GateLayer
	Gate     PFCGateParams  `desc:"PFC Gating parameters"`
	Maint    PFCMaintParams `desc:"PFC Maintenance parameters"`
	Dyns     []*PFCDyn      `desc:"PFC dynamic behavior parameters -- provides deterministic control over PFC maintenance dynamics -- the rows of PFC units (along Y axis) behave according to corresponding index of Dyns -- grouped together -- ensure Y dim has even multiple of len(Dyns)"`
	PFCNeurs []PFCNeuron    `desc:"slice of PFCNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values."`
}

// UnitValByIdx returns value of given variable by variable index
// and flat neuron index (from layer or neuron-specific one).
// First indexes are ModNeuronVars
func (ly *PFCLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	if vidx != ActG {
		return ly.GateLayer.UnitValByIdx(vidx, idx)
	}
	pnrn := &ly.PFCNeurs[idx]
	return pnrn.ActG
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *PFCLayer) Build() error {
	err := ly.GateLayer.Build()
	if err != nil {
		return err
	}
	ly.PFCNeurs = make([]PFCNeuron, len(ly.Neurons))
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *PFCLayer) InitActs() {
	ly.GateLayer.InitActs()
	for ni := range ly.PFCNeurs {
		pnr := &ly.PFCNeurs[ni]
		pnr.ActG = 0
	}
}

// SetDyn sets given dynamic maint element to given parameters (must exist)
func (ly *PFCLayer) SetDyn(dyn int, init, rise, decay float32, desc string) *PFCDyn {
	dy := &PFCDyn{}
	dy.Set(init, rise, decay, desc)
	ly.Dyns[dyn] = dy
	return dy
}

// MaintDyn creates basic default maintenance dynamic configuration -- every
// unit just maintains over time.
// This should be used for Output gating layer.
func (ly *PFCLayer) MaintDyn() {
	ly.Dyns = make([]*PFCDyn, 1)
	ly.SetDyn(0, 1, 0, 0, "maintain stable act")
}

// FullDyn creates full dynamic Dyn configuration, with 5 different
// dynamic profiles: stable maint, phasic, rising maint, decaying maint,
// and up / down maint.  tau is the rise / decay base time constant.
func (ly *PFCLayer) FullDyn(tau float32) {
	ndyn := 5
	ly.Dyns = make([]*PFCDyn, ndyn)

	ly.SetDyn(0, 1, 0, 0, "maintain stable act")
	ly.SetDyn(1, 1, 0, 1, "immediate phasic response")
	ly.SetDyn(2, .1, tau, 0, "maintained, rising value over time")
	ly.SetDyn(3, 1, 0, tau, "maintained, decaying value over time")
	ly.SetDyn(4, .1, .5*tau, tau, "maintained, rising then falling over time")
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// todo: compute GeRaw min / max
//

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *PFCLayer) GFmInc(ltime *leabra.Time) {
	if ly.Typ != deep.TRC && ly.Typ != deep.Deep { // Super -- note PFC should not be TRC anyway
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			tg := &ly.GateStates[nrn.SubPool]
			if tg.Cnt < 0 {
				continue
			} else if tg.Cnt == 0 { // just gated
				dnr := &ly.DeepNeurs[ni]
				geMax := math32.Min(tg.GeRaw.Max, ly.Maint.MntGeMax)
				geFact := 1 - (geMax / ly.Maint.MntGeMax)
				geMnt := ly.Maint.SMnt.ProjVal(geFact)
				geRaw := geMnt * dnr.DeepLrn
				_ = geRaw
				ly.Act.GRawFmInc(nrn)
				// geRaw := ly.DeepTRC.BurstGe(dnr.TRCBurstGe) // only use trcburst
				// ly.Act.GeFmRaw(nrn, geRaw)
				// ly.Act.GiFmRaw(nrn, nrn.GiRaw)
			}
		}
	} else {
		ly.GateLayer.GFmInc(ltime) // use deep version
	}
}

// Gating computes PFC Gating state
func (ly *PFCLayer) Gating(ltime *leabra.Time) {
}
