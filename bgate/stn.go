// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// KCaParams are Calcium-gated potassium channels that drive the long
// afterhyperpolarization of STN neurons.
// The conductance is applied to KNa channels ta take advantage
// of the existing infrastructure.
type KCaParams struct {
	ActThr float32 `desc:"threshold for activation to turn on KCa channels -- assuming that there is a sufficiently nonlinear increase in Ca influx above a critical firing rate, so as to make these channels threshold-like"`
	GKCa   float32 `desc:"kCa conductance after threshold is hit -- actual conductance is applied to KNa channels"`
}

// STNLayer represents the dorsal matrisome MSN's that are the main
// Go / NoGo gating units in BG.  D1R = Go, D2R = NoGo.
type STNLayer struct {
	leabra.Layer
	KCa KCaParams `desc:"parameters for the calcium-gated potassium channels that drive the afterhyperpolarization that open the gating window in STN neurons (Hallworth et al., 2003)"`
	DA  float32   `inactive:"+" desc:"dopamine value for this layer"`
}

var KiT_STNLayer = kit.Types.AddType(&STNLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "STNLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.9",
// 		"Layer.Act.Init.Act":  "0.5",
// 		"Layer.Act.Erev.L":    "0.9",
// 		"Layer.Act.Gbar.L":    "0.2", // stronger here makes them more robust to inputs -- .2 is best
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "4",
// 		"Layer.Act.Dt.GTau":        "5", // 5 also works but less smooth RT dist
// 		"Layer.Act.Gbar.L":         "0.1",
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *STNLayer) Defaults() {
	ly.Layer.Defaults()
	ly.DA = 0

	// STN is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Vm = 0.9
	ly.Act.Init.Act = 0.5
	ly.Act.Erev.L = 0.9
	ly.Act.Gbar.L = 0.2
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.ActAvg.Init = 0.25
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.Self.Tau = 3.0
	ly.Act.XX1.Gain = 20 // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 4
	ly.Act.Dt.GTau = 5 // 5 also works but less smooth RT dist
	ly.Act.Gbar.L = 0.1
	ly.Act.Init.Decay = 0

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		if _, ok := pj.Send.(*GPLayer); ok { // GPeIn -- others are PFC, 1.5 in orig
			pj.WtScale.Rel = 0.5
		}
	}
}

// DALayer interface:

func (ly *STNLayer) GetDA() float32   { return ly.DA }
func (ly *STNLayer) SetDA(da float32) { ly.DA = da }

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *STNLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

func (ly *STNLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}

func (ly *STNLayer) ActFmG(ltime *leabra.Time) {
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