// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// KCaParams are Calcium-gated potassium channels that drive the long
// afterhyperpolarization of STN neurons.  Auto reset at each AlphaCycle.
// The conductance is applied to KNa channels ta take advantage
// of the existing infrastructure.
type KCaParams struct {
	ActThr float32 `def:"0.7" desc:"threshold for activation to turn on KCa channels -- assuming that there is a sufficiently nonlinear increase in Ca influx above a critical firing rate, so as to make these channels threshold-like"`
	GKCa   float32 `desc:"kCa conductance after threshold is hit -- actual conductance is applied to KNa channels"`
}

func (kc *KCaParams) Defaults() {
	kc.ActThr = 0.7
	kc.GKCa = 20
}

// STNpLayer represents the pausing subtype of STN neurons.
// These open the gating window.
type STNpLayer struct {
	leabra.Layer
	KCa KCaParams `desc:"parameters for the calcium-gated potassium channels that drive the afterhyperpolarization that open the gating window in STN neurons (Hallworth et al., 2003)"`
	DA  float32   `inactive:"+" desc:"dopamine value for this layer"`
}

var KiT_STNpLayer = kit.Types.AddType(&STNpLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "STNpLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.9",
// 		"Layer.Act.Init.Act":  "0.5",
// 		"Layer.Act.Erev.L":    "0.8",
// 		"Layer.Act.Gbar.L":    "0.3",
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3",
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *STNpLayer) Defaults() {
	ly.Layer.Defaults()
	ly.KCa.Defaults()
	ly.DA = 0

	// STN is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Vm = 0.9
	ly.Act.Init.Act = 0.5
	ly.Act.Erev.L = 0.8
	ly.Act.Gbar.L = 0.3
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.ActAvg.Init = 0.25
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.Self.Tau = 3.0
	ly.Act.XX1.Gain = 20 // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3
	ly.Act.Dt.GTau = 3 // fastest
	ly.Act.Init.Decay = 0

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		pj.Learn.Learn = false
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.9
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		if _, ok := pj.Send.(*GPLayer); ok { // GPeIn -- others are PFC, 1.5 in orig
			pj.WtScale.Abs = 0.5
		}
	}

	ly.UpdateParams()
}

// DALayer interface:

func (ly *STNpLayer) GetDA() float32   { return ly.DA }
func (ly *STNpLayer) SetDA(da float32) { ly.DA = da }

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *STNpLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

func (ly *STNpLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
// should already have presented the external input to the network at this point.
func (ly *STNpLayer) AlphaCycInit() {
	ly.Layer.AlphaCycInit()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Gk = 0
	}
}

func (ly *STNpLayer) ActFmG(ltime *leabra.Time) {
	for ni := range ly.Neurons { // note: copied from leabra ActFmG, not calling it..
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		if nrn.Act >= ly.KCa.ActThr {
			nrn.Gk = ly.KCa.GKCa
		}
		ly.Learn.AvgsFmAct(nrn)
	}
}

//////////////////////////////////////////////////////////////////////
// STNsLayer

// STNsLayer represents the pausing subtype of STN neurons.
// These open the gating window.
type STNsLayer struct {
	leabra.Layer
	DA float32 `inactive:"+" desc:"dopamine value for this layer"`
}

var KiT_STNsLayer = kit.Types.AddType(&STNsLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "STNsLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.9",
// 		"Layer.Act.Init.Act":  "0.5",
// 		"Layer.Act.Erev.L":    "0.8",
// 		"Layer.Act.Gbar.L":    "0.3",
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3",
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *STNsLayer) Defaults() {
	ly.Layer.Defaults()
	ly.DA = 0

	// STN is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Vm = 0.9
	ly.Act.Init.Act = 0.5
	ly.Act.Erev.L = 0.8
	ly.Act.Gbar.L = 0.3
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.ActAvg.Init = 0.25
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.Self.Tau = 3.0
	ly.Act.XX1.Gain = 20 // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3
	ly.Act.Dt.GTau = 3 // fastest
	ly.Act.Init.Decay = 0

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		pj.Learn.Learn = false
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.9
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
	}

	ly.UpdateParams()
}

// DALayer interface:

func (ly *STNsLayer) GetDA() float32   { return ly.DA }
func (ly *STNsLayer) SetDA(da float32) { ly.DA = da }

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *STNsLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

func (ly *STNsLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}
