// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/chewxy/math32"
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// PFCGateParams has parameters for PFC gating
type PFCGateParams struct {
	OutGate   bool    `desc:"if true, this PFC layer is an output gate layer, which means that it only has transient activation during gating"`
	OutQ1Only bool    `viewif:"OutGate" def:"true" desc:"for output gating, only compute gating in first quarter -- do not compute in 3rd quarter -- this is typically true, and BurstQtr is typically set to only Q1 as well -- does Burst updating immediately after first quarter gating signal -- allows gating signals time to influence performance within a single trial"`
	MntThal   float32 `def:"1" desc:"effective Thal activation to use in computing the Burst activation sent from Super to Deep layers, for continued maintenance beyond the initial Thal signal provided by the BG -- also sets an effective minimum Thal value regardless of the actual gating thal value"`
}

func (gp *PFCGateParams) Defaults() {
	gp.OutQ1Only = true
	gp.MntThal = 1
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

var KiT_PFCLayer = kit.Types.AddType(&PFCLayer{}, deep.LayerProps)

func (ly *PFCLayer) Defaults() {
	ly.GateLayer.Defaults()
	ly.Gate.Defaults()
	ly.Maint.Defaults()
	if ly.Gate.OutGate && ly.Gate.OutQ1Only {
		ly.DeepBurst.BurstQtr = 0
		ly.DeepBurst.SetBurstQtr(leabra.Q1)
	} else {
		ly.DeepBurst.BurstQtr = 0
		ly.DeepBurst.SetBurstQtr(leabra.Q2)
		ly.DeepBurst.SetBurstQtr(leabra.Q4)
	}
}

func (ly *PFCLayer) GateType() GateTypes {
	if ly.Gate.OutGate {
		return Out
	} else {
		return Maint
	}
}

// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
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

// MaintPFC returns corresponding PFC maintenance layer with same name but out -> mnt
// could be nil
func (ly *PFCLayer) MaintPFC() *PFCLayer {
	sz := len(ly.Nm)
	mnm := ly.Nm[:sz-3] + "mnt"
	li := ly.Network.LayerByName(mnm)
	if li == nil {
		return nil
	}
	return li.(*PFCLayer)
}

// DeepPFC returns corresponding PFC deep layer with same name + D
// could be nil
func (ly *PFCLayer) DeepPFC() *PFCLayer {
	dnm := ly.Nm + "D"
	li := ly.Network.LayerByName(dnm)
	if li == nil {
		return nil
	}
	return li.(*PFCLayer)
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

// DecayStatePool decays activation state by given proportion in given pool index (0 based)
func (ly *PFCLayer) DecayStatePool(pool int, decay float32) {
	pi := int32(pool + 1) // 1 based
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() || nrn.SubPool != pi {
			continue
		}
		ly.Act.DecayState(nrn, decay)
	}
	pl := &ly.Pools[pi]
	pl.Inhib.Decay(decay)
}

// ClearCtxtPool clears CtxtGe in given pool index (0 based)
func (ly *PFCLayer) ClearCtxtPool(pool int) {
	pi := int32(pool + 1) // 1 based
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() || nrn.SubPool != pi {
			continue
		}
		dnr := &ly.DeepNeurs[ni]
		dnr.Burst = 0
		dnr.CtxtGe = 0
	}
}

// ClearMaint resets maintenance in corresponding pool (0 based) in maintenance layer
func (ly *PFCLayer) ClearMaint(pool int) {
	pfcm := ly.MaintPFC()
	if pfcm == nil {
		return
	}
	pfcm.DecayStatePool(pool, pfcm.Maint.Clear)
	gs := &ly.GateStates[pool] // 0 based
	gs.Cnt = -1                // reset
}

//////////////////////////////////////////////////////////////////////////////////////
//  Dyns

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

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (ly *PFCLayer) AvgMaxGe(ltime *leabra.Time) {
	ly.GateLayer.AvgMaxGe(ltime)
	ly.AvgMaxGeRaw(ltime) // defined in GateLayer
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *PFCLayer) GFmInc(ltime *leabra.Time) {
	if !ly.IsSuper() {
		ly.GateLayer.GFmInc(ltime) // use deep version
		return
	}
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		dnr := &ly.DeepNeurs[ni]
		gs := &ly.GateStates[int(nrn.SubPool)-1]
		ly.Act.GRawFmInc(nrn)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
		geRaw := nrn.GeRaw
		if gs.Cnt == 0 { // just gated -- only maint if nothing else going on
			if gs.GeRaw.Max < 0.05 {
				geRaw += ly.Maint.SMnt.Max * dnr.DeepLrn
			}
		} else if gs.Cnt > 0 { // maintaining
			geMax := math32.Min(gs.GeRaw.Max, ly.Maint.MntGeMax)
			geFact := 1 - (geMax / ly.Maint.MntGeMax)
			geMnt := ly.Maint.SMnt.ProjVal(geFact)
			geRaw += geMnt * dnr.DeepLrn
		}
		ly.Act.GeFmRaw(nrn, geRaw)
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act.
// PFC extends to call Gating.
func (ly *PFCLayer) ActFmG(ltime *leabra.Time) {
	ly.GateLayer.ActFmG(ltime)
	ly.Gating(ltime)
}

// GateSend computes PFC Gating state
func (ly *PFCLayer) Gating(ltime *leabra.Time) {
	if !ly.IsSuper() {
		return // only SUPER
	}
	if ly.Gate.OutGate && ly.Gate.OutQ1Only {
		if ltime.Quarter > 1 {
			return
		}
	}

	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		if !gs.Now { // not gating now
			continue
		}
		if gs.Act > 0 { // use GPiThal threshold, so anything > 0
			if gs.Cnt >= 1 { // already maintaining
				if ly.Maint.Clear > 0 {
					ly.DecayStatePool(gi, ly.Maint.Clear)
				}
			}
			gs.Cnt = 0           // this is the "just gated" signal
			if ly.Gate.OutGate { // time to clear out maint
				ly.ClearMaint(gi)
			}
		}
		// test for over-duration maintenance -- allow for active gating to override
		if gs.Cnt >= ly.Maint.MaxMaint {
			gs.Cnt = -1
		}
	}
}

// QuarterFinal does updating after end of a quarter
func (ly *PFCLayer) QuarterFinal(ltime *leabra.Time) {
	ly.GateLayer.QuarterFinal(ltime)
	if ly.IsSuper() {
		ly.GateStateToDeep()
	}
}

// GateStateToDeep copies superficial gate state to corresponding deep layer
func (ly *PFCLayer) GateStateToDeep() {
	pfcd := ly.DeepPFC()
	if pfcd == nil {
		return
	}
	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		dgs := &pfcd.GateStates[gi]
		dgs.Cnt = gs.Cnt // just the count
	}
}

// SendCtxtGe sends full Burst activation over BurstCtxt projections to integrate
// CtxtGe excitatory conductance on deep layers.
// This must be called at the end of the DeepBurst quarter for this layer.
func (ly *PFCLayer) SendCtxtGe(ltime *leabra.Time) {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		return
	}
	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		if gs.Cnt < 0 {
			ly.ClearCtxtPool(gi)
			gs.Cnt--
		} else {
			gs.Cnt++
		}
	}

	// todo: could optimize to not send if not maint

	ly.GateLayer.SendCtxtGe(ltime)
}

// BurstFmAct updates Burst layer 5 IB bursting value from current Act (superficial activation)
// Subject to thresholding.
func (ly *PFCLayer) BurstFmAct(ltime *leabra.Time) {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		return
	}
	if !ly.IsSuper() { // rest is special for super
		ly.GateLayer.BurstFmAct(ltime)
		return
	}
	lpl := &ly.DeepPools[0]
	actMax := lpl.ActNoAttn.Max
	actAvg := lpl.ActNoAttn.Avg
	thr := actAvg + ly.DeepBurst.ThrRel*(actMax-actAvg)
	thr = math32.Max(thr, ly.DeepBurst.ThrAbs)

	for ni := range ly.DeepNeurs {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		dnr := &ly.DeepNeurs[ni]
		var burst float32
		if dnr.ActNoAttn > thr {
			burst = dnr.ActNoAttn
			// only PFC-specific gated part here:
			gs := ly.GateStates[int(nrn.SubPool)-1]
			if gs.Cnt < 0 { // not gated or maintaining
				burst = 0
			} else {
				burst *= math32.Max(ly.Gate.MntThal, gs.Act)
			}
		}
		dnr.Burst = burst
	}
}

// RecGateAct records the gating activation from current activation, when gating occcurs
// based on GateState.Now
func (ly *PFCLayer) RecGateAct(ltime *leabra.Time) {
	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		if !gs.Now { // not gating now
			continue
		}
		pl := ly.Pools[1+gi]
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pnr := &ly.PFCNeurs[ni]
			pnr.ActG = nrn.Act
		}
	}
}

// DoQuarter2DWt indicates whether to do optional Q2 DWt
func (ly *PFCLayer) DoQuarter2DWt() bool {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(1) {
		return false
	}
	return true
}

// todo: Quarter_Init_Deep called "Compute_DeepStateUpdt":

// void STATE_CLASS(PFCUnitSpec)::Compute_DeepStateUpdt
//   (LEABRA_UNIT_STATE* u, LEABRA_NETWORK_STATE* net, int thr_no) {
//
//   if(!deep.on || !Quarter_DeepRawPrevQtr(net->quarter)) return;
//
//   if(maint.use_dyn && deep.IsDeep() && u->thal_cnt >= 0) { // update dynamics!
//     LEABRA_LAYER_STATE* lay = u->GetOwnLayer(net);
//     int unidx = u->ungp_un_idx;
//     int dyn_row = unidx % n_dyns;
//     if(u->thal_cnt <= 1.0f) { // first gating -- should only ever be 1.0 here..
//       u->misc_1 = u->deep_ctxt; // record gating ctxt
//       u->deep_ctxt *= InitDynVal(dyn_row);
//     }
//     else {
//       u->deep_ctxt = u->misc_1 * UpdtDynVal(dyn_row, (u->thal_cnt-1.0f));
//     }
//   }
//
//   inherited::Compute_DeepStateUpdt(u, net, thr_no);
// }
