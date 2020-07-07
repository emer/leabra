// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"strings"

	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// VThalLayer represents the Ventral thalamus: VA / VM / VL which receives BG gating.
type VThalLayer struct {
	leabra.Layer
	DA float32 `inactive:"+" desc:"dopamine value for this layer"`
}

var KiT_VThalLayer = kit.Types.AddType(&VThalLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "VThalLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3", // fastest
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *VThalLayer) Defaults() {
	ly.Layer.Defaults()
	ly.DA = 0

	// Thal is tonically self-active and has no FFFB inhibition
	// TODO: experiment with this

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
		if strings.HasSuffix(pj.Send.Name(), "GPi") {
			pj.WtScale.Abs = 2.0
		}
	}

	ly.UpdateParams()
}

// DALayer interface:

func (ly *VThalLayer) GetDA() float32   { return ly.DA }
func (ly *VThalLayer) SetDA(da float32) { ly.DA = da }

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *VThalLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

func (ly *VThalLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}
