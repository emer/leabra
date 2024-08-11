// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/leabra/v2/leabra"
)

// PFCGateParams has parameters for PFC gating
type PFCGateParams struct {

	// Quarter(s) that the effect of gating on updating Deep from Super occurs -- this is typically 1 quarter after the GPiThal GateQtr
	GateQtr leabra.Quarters

	// if true, this PFC layer is an output gate layer, which means that it only has transient activation during gating
	OutGate bool

	// for output gating, only compute gating in first quarter -- do not compute in 3rd quarter -- this is typically true, and GateQtr is typically set to only Q1 as well -- does Burst updating immediately after first quarter gating signal -- allows gating signals time to influence performance within a single trial
	OutQ1Only bool `viewif:"OutGate" def:"true"`
}

func (gp *PFCGateParams) Defaults() {
	gp.GateQtr.SetFlag(true, leabra.Q2)
	gp.GateQtr.SetFlag(true, leabra.Q4)
	gp.OutQ1Only = true
}

// PFCMaintParams for PFC maintenance functions
type PFCMaintParams struct {

	// use fixed dynamics for updating deep_ctxt activations -- defined in dyn_table -- this also preserves the initial gating deep_ctxt value in Maint neuron val (view as Cust1) -- otherwise it is up to the recurrent loops between super and deep for maintenance
	UseDyn bool

	// multiplier on maint current
	MaintGain float32 `min:"0" def:"0.8"`

	// on output gating, clear corresponding maint pool.  theoretically this should be on, but actually it works better off in most cases..
	OutClearMaint bool `def:"false"`

	// how much to clear out (decay) super activations when the stripe itself gates and was previously maintaining something, or for maint pfc stripes, when output go fires and clears.
	Clear    float32 `min:"0" max:"1" def:"0"`
	MaxMaint int     `"min:"1" def:"1:100" maximum duration of maintenance for any stripe -- beyond this limit, the maintenance is just automatically cleared -- typically 1 for output gating and 100 for maintenance gating"`
}

func (mp *PFCMaintParams) Defaults() {
	mp.MaintGain = 0.8
	mp.OutClearMaint = false // theoretically should be true, but actually was false due to bug
	mp.Clear = 0
	mp.MaxMaint = 100
}

// PFCNeuron contains extra variables for PFCLayer neurons -- stored separately
type PFCNeuron struct {

	// gating activation -- the activity value when gating occurred in this pool
	ActG float32

	// maintenance value for Deep layers = sending act at time of gating
	Maint float32

	// maintenance excitatory conductance value for Deep layers
	MaintGe float32
}

///////////////////////////////////////////////////////////////////
// PFCDeepLayer

// PFCDeepLayer is a Prefrontal Cortex BG-gated deep working memory layer.
// This handles all of the PFC-specific functionality, looking for a corresponding
// Super layer with the same name except no final D.
// If Dyns are used, they are represented in extra Y-axis neurons, with the inner-loop
// being the basic Super Y axis values for each Dyn type, and outer-loop the Dyn types.
type PFCDeepLayer struct {
	GateLayer

	// PFC Gating parameters
	Gate PFCGateParams `view:"inline"`

	// PFC Maintenance parameters
	Maint PFCMaintParams `view:"inline"`

	// PFC dynamic behavior parameters -- provides deterministic control over PFC maintenance dynamics -- the rows of PFC units (along Y axis) behave according to corresponding index of Dyns (inner loop is Super Y axis, outer is Dyn types) -- ensure Y dim has even multiple of len(Dyns)
	Dyns PFCDyns

	// slice of PFCNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values.
	PFCNeurs []PFCNeuron
}

func (ly *PFCDeepLayer) Defaults() {
	ly.GateLayer.Defaults()
	ly.Gate.Defaults()
	ly.Maint.Defaults()
	if ly.Gate.OutGate && ly.Gate.OutQ1Only {
		ly.Maint.MaxMaint = 1
		ly.Gate.GateQtr = 0
		ly.Gate.GateQtr.SetFlag(true, leabra.Q1)
	}
	if len(ly.Dyns) > 0 {
		ly.Maint.UseDyn = true
	} else {
		ly.Maint.UseDyn = false
	}
}

func (ly *PFCDeepLayer) GateType() GateTypes {
	if ly.Gate.OutGate {
		return Out
	} else {
		return Maint
	}
}

// UnitValueByIndex returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *PFCDeepLayer) UnitValueByIndex(vidx NeurVars, idx int) float32 {
	pnrn := &ly.PFCNeurs[idx]
	switch vidx {
	case ActG:
		return pnrn.ActG
	case NrnMaint:
		return pnrn.Maint
	case MaintGe:
		return pnrn.MaintGe
	default:
		return ly.GateLayer.UnitValueByIndex(vidx, idx)
	}
}

// Build constructs the layer state, including calling Build on the pathways.
func (ly *PFCDeepLayer) Build() error {
	err := ly.GateLayer.Build()
	if err != nil {
		return err
	}
	ly.PFCNeurs = make([]PFCNeuron, len(ly.Neurons))
	return nil
}

// MaintPFC returns corresponding PFCDeep maintenance layer with same name but outD -> mntD
// could be nil
func (ly *PFCDeepLayer) MaintPFC() *PFCDeepLayer {
	sz := len(ly.Name)
	mnm := ly.Name[:sz-4] + "mntD"
	li := ly.Network.LayerByName(mnm)
	if li == nil {
		return nil
	}
	return li.(*PFCDeepLayer)
}

// SuperPFC returns corresponding PFC super layer with same name without D
// should not be nil.  Super can be any layer type.
func (ly *PFCDeepLayer) SuperPFC() leabra.LeabraLayer {
	dnm := ly.Name[:len(ly.Name)-1]
	li := ly.Network.LayerByName(dnm)
	if li == nil {
		return nil
	}
	return li.(leabra.LeabraLayer)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *PFCDeepLayer) InitActs() {
	ly.GateLayer.InitActs()
	for ni := range ly.PFCNeurs {
		pnr := &ly.PFCNeurs[ni]
		pnr.ActG = 0
		pnr.Maint = 0
		pnr.MaintGe = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *PFCDeepLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		pnr := &ly.PFCNeurs[ni]
		geRaw := nrn.GeRaw + pnr.MaintGe
		ly.Act.GeFmRaw(nrn, geRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act.
// PFC extends to call Gating.
func (ly *PFCDeepLayer) ActFmG(ltime *leabra.Time) {
	ly.GateLayer.ActFmG(ltime)
	ly.Gating(ltime)
}

// Gating updates PFC Gating state
func (ly *PFCDeepLayer) Gating(ltime *leabra.Time) {
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
			gs.Cnt = 0           // this is the "just gated" signal
			if ly.Gate.OutGate { // time to clear out maint
				if ly.Maint.OutClearMaint {
					ly.ClearMaint(gi)
				}
			} else {
				pfcs := ly.SuperPFC().AsLeabra()
				pfcs.DecayStatePool(gi, ly.Maint.Clear)
			}
		}
		// test for over-duration maintenance -- allow for active gating to override
		if gs.Cnt >= ly.Maint.MaxMaint {
			gs.Cnt = -1
		}
	}
}

// ClearMaint resets maintenance in corresponding pool (0 based) in maintenance layer
func (ly *PFCDeepLayer) ClearMaint(pool int) {
	pfcm := ly.MaintPFC()
	if pfcm == nil {
		return
	}
	gs := &pfcm.GateStates[pool] // 0 based
	if gs.Cnt >= 1 {             // important: only for established maint, not just gated..
		gs.Cnt = -1 // reset
		pfcs := pfcm.SuperPFC().AsLeabra()
		pfcs.DecayStatePool(pool, pfcm.Maint.Clear)
	}
}

// QuarterFinal does updating after end of a quarter
func (ly *PFCDeepLayer) QuarterFinal(ltime *leabra.Time) {
	ly.GateLayer.QuarterFinal(ltime)
	ly.UpdateGateCnt(ltime)
	ly.DeepMaint(ltime)
}

// DeepMaint updates deep maintenance activations
func (ly *PFCDeepLayer) DeepMaint(ltime *leabra.Time) {
	if !ly.Gate.GateQtr.HasFlag(ltime.Quarter) {
		return
	}
	slyi := ly.SuperPFC()
	if slyi == nil {
		return
	}
	sly := slyi.AsLeabra()
	yN := ly.Shp.Dim(2)
	xN := ly.Shp.Dim(3)

	nn := yN * xN

	syN := sly.Shp.Dim(2)
	sxN := sly.Shp.Dim(3)
	snn := syN * sxN

	dper := yN / syN  // dyn per sender -- should be len(Dyns)
	dtyp := yN / dper // dyn type

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ui := ni % nn
		pi := ni / nn
		uy := ui / xN
		ux := ui % xN

		pnr := &ly.PFCNeurs[ni]
		gs := &ly.GateStates[nrn.SubPool-1]
		if gs.Cnt < 0 {
			pnr.Maint = 0
			pnr.MaintGe = 0
		} else if gs.Cnt <= 1 { // first gating, save first gating value
			sy := uy % syN // inner loop is s
			si := pi*snn + sy*sxN + ux
			snr := &sly.Neurons[si]
			pnr.Maint = ly.Maint.MaintGain * snr.Act
		}
		if ly.Maint.UseDyn {
			pnr.MaintGe = pnr.Maint * ly.Dyns.Value(dtyp, float32(gs.Cnt-1))
		} else {
			pnr.MaintGe = pnr.Maint
		}
	}
}

// UpdateGateCnt updates the gate counter
func (ly *PFCDeepLayer) UpdateGateCnt(ltime *leabra.Time) {
	if !ly.Gate.GateQtr.HasFlag(ltime.Quarter) {
		return
	}
	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		if gs.Cnt < 0 {
			// ly.ClearCtxtPool(gi)
			gs.Cnt--
		} else {
			gs.Cnt++
		}
	}
}

// RecGateAct records the gating activation from current activation,
// when gating occcurs based on GateState.Now
func (ly *PFCDeepLayer) RecGateAct(ltime *leabra.Time) {
	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		if !gs.Now { // not gating now
			continue
		}
		pl := &ly.Pools[1+gi]
		for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.Off {
				continue
			}
			pnr := &ly.PFCNeurs[ni]
			pnr.ActG = nrn.Act
		}
	}
}

// DoQuarter2DWt indicates whether to do optional Q2 DWt
func (ly *PFCDeepLayer) DoQuarter2DWt() bool {
	if !ly.Gate.GateQtr.HasFlag(leabra.Q2) {
		return false
	}
	return true
}
