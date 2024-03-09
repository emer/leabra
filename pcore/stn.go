// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"strings"

	"cogentcore.org/core/kit"
	"cogentcore.org/core/mat32"
	"github.com/emer/leabra/v2/leabra"
)

// CaParams control the calcium dynamics in STN neurons.
// Gillies & Willshaw, 2006 provide a biophysically detailed simulation,
// and we use their logistic function for computing KCa conductance based on Ca,
// but we use a simpler approximation with burst and act threshold.
// KCa are Calcium-gated potassium channels that drive the long
// afterhyperpolarization of STN neurons.  Auto reset at each AlphaCycle.
// The conductance is applied to KNa channels to take advantage
// of the existing infrastructure.
type CaParams struct {

	// activation threshold for bursting that drives strong influx of Ca to turn on KCa channels -- there is a complex de-inactivation dynamic involving the volley of excitation and inhibition from GPe, but we can just use a threshold
	BurstThr float32 `def:"0.9"`

	// activation threshold for increment in activation above baseline that drives lower influx of Ca
	ActThr float32 `def:"0.7"`

	// Ca level for burst level activation
	BurstCa float32 `def:"1"`

	// Ca increment from regular sub-burst activation -- drives slower inhibition of firing over time -- for stop-type STN dynamics that initially put hold on GPi and then decay
	ActCa float32 `def:"0.2"`

	// maximal KCa conductance (actual conductance is applied to KNa channels)
	GbarKCa float32 `def:"10"`

	// KCa conductance time constant -- 40 from Gillies & Willshaw, 2006, but sped up here to fit in AlphaCyc
	KCaTau float32 `def:"20"`

	// Ca time constant of decay to baseline -- 185.7 from Gillies & Willshaw, 2006, but sped up here to fit in AlphaCyc
	CaTau float32 `def:"50"`

	// initialize Ca, KCa values at start of every AlphaCycle
	AlphaInit bool
}

func (kc *CaParams) Defaults() {
	kc.BurstThr = 0.9
	kc.ActThr = 0.7
	kc.BurstCa = 1 // just long enough for 100 msec alpha trial window
	kc.ActCa = 0.2
	kc.GbarKCa = 10 // 20
	kc.KCaTau = 20  // 20
	kc.CaTau = 50   // 185.7
}

// KCaGFmCa returns the driving conductance for KCa channels based on given Ca level.
// This equation comes from Gillies & Willshaw, 2006.
func (kc *CaParams) KCaGFmCa(ca float32) float32 {
	return 0.81 / (1 + mat32.Exp(-(mat32.Log(ca)+0.3))/0.46)
}

///////////////////////////////////////////////////////////////////////////
// STNLayer

// STNLayer represents STN neurons, with two subtypes:
// STNp are more strongly driven and get over bursting threshold, driving strong,
// rapid activation of the KCa channels, causing a long pause in firing, which
// creates a window during which GPe dynamics resolve Go vs. No balance.
// STNs are more weakly driven and thus more slowly activate KCa, resulting in
// a longer period of activation, during which the GPi is inhibited to prevent
// premature gating based only MtxGo inhibition -- gating only occurs when
// GPeIn signal has had a chance to integrate its MtxNo inputs.
type STNLayer struct {
	Layer

	// parameters for calcium and calcium-gated potassium channels that drive the afterhyperpolarization that open the gating window in STN neurons (Hallworth et al., 2003)
	Ca CaParams `view:"inline"`

	// slice of extra STNNeuron state for this layer -- flat list of len = Shape.Len(). You must iterate over index and use pointer to modify values.
	STNNeurs []STNNeuron
}

var KiT_STNLayer = kit.Types.AddType(&STNLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "STNLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.56",
// 		"Layer.Act.Init.Act":  "0.57",
// 		"Layer.Act.Erev.L":    "0.8",
// 		"Layer.Act.Gbar.L":    "0.4",
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3",
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *STNLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Ca.Defaults()
	ly.DA = 0

	// STN is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Vm = 0.56
	ly.Act.Init.Act = 0.63
	ly.Act.Erev.L = 0.8
	ly.Act.Gbar.L = 0.4
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.Self.Tau = 3.0
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 0.25
	ly.Act.XX1.Gain = 20 // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3
	ly.Act.Dt.GTau = 3 // fastest
	ly.Act.Init.Decay = 0

	if strings.HasSuffix(ly.Nm, "STNp") {
		ly.Act.Init.Act = 0.48
	}

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		pj.Learn.Learn = false
		pj.Learn.Norm.On = false
		pj.Learn.Momentum.On = false
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.9
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		if strings.HasSuffix(ly.Nm, "STNp") {
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToSTNp
				pj.WtScale.Abs = 0.1
			}
		} else { // STNs
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToSTNs
				pj.WtScale.Abs = 0.1 // note: not currently used -- interferes with threshold-based Ca self-inhib dynamics
			} else {
				pj.WtScale.Abs = 0.2 // weaker inputs
			}
		}
	}

	ly.UpdateParams()
}

// DALayer interface:

func (ly *STNLayer) GetDA() float32   { return ly.DA }
func (ly *STNLayer) SetDA(da float32) { ly.DA = da }

func (ly *STNLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.STNNeurs {
		nrn := &ly.STNNeurs[ni]
		nrn.Ca = 0
		nrn.KCa = 0
	}
}

func (ly *STNLayer) AlphaCycInit(updtActAvg bool) {
	ly.Layer.AlphaCycInit(updtActAvg)
	if !ly.Ca.AlphaInit {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Gk = 0
		snr := &ly.STNNeurs[ni]
		snr.Ca = 0
		snr.KCa = 0
	}
}

func (ly *STNLayer) ActFmG(ltime *leabra.Time) {
	for ni := range ly.Neurons { // note: copied from leabra ActFmG, not calling it..
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)

		snr := &ly.STNNeurs[ni]
		snr.KCa += (ly.Ca.KCaGFmCa(snr.Ca) - snr.KCa) / ly.Ca.KCaTau
		dCa := -snr.Ca / ly.Ca.CaTau
		if nrn.Act >= ly.Ca.BurstThr {
			dCa += ly.Ca.BurstCa
			snr.KCa = 1 // burst this too
		} else if nrn.Act >= ly.Ca.ActThr {
			dCa += (nrn.Act - ly.Ca.ActThr) * ly.Ca.ActCa
		}
		snr.Ca += dCa
		nrn.Gk = ly.Ca.GbarKCa * snr.KCa

		ly.Learn.AvgsFmAct(nrn)
	}
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *STNLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.STNNeurs = make([]STNNeuron, len(ly.Neurons))
	return nil
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *STNLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = STNNeuronVarIdxByName(varNm)
	if err != nil {
		return -1, err
	}
	nn := ly.Layer.UnitVarNum()
	return nn + vidx, err
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *STNLayer) UnitVal1D(varIdx int, idx int, di int) float32 {
	if varIdx < 0 {
		return mat32.NaN()
	}
	nn := ly.Layer.UnitVarNum()
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx, di)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	varIdx -= nn
	if varIdx > len(STNNeuronVars) {
		return mat32.NaN()
	}
	snr := &ly.STNNeurs[idx]
	return snr.VarByIndex(varIdx)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *STNLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + len(STNNeuronVars)
}
