// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"strings"

	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// MatrixParams has parameters for Dorsal Striatum Matrix computation
// These are the main Go / NoGo gating units in BG driving updating of PFC WM in PBWM
type MatrixParams struct {
	BurstGain float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
	DipGain   float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
}

func (mp *MatrixParams) Defaults() {
	mp.BurstGain = 1
	mp.DipGain = 1
}

// MatrixLayer represents the dorsal matrisome MSN's that are the main
// Go / NoGo gating units in BG.  D1R = Go, D2R = NoGo.
type MatrixLayer struct {
	leabra.Layer
	DaR    DaReceptors  `desc:"dominant type of dopamine receptor -- D1R for Go pathway, D2R for NoGo"`
	Matrix MatrixParams `view:"inline" desc:"matrix parameters"`
	DA     float32      `inactive:"+" desc:"dopamine value for this layer"`
	DALrn  float32      `inactive:"+" desc:"effective learning dopamine value for this layer: reflects DaR and Gains"`
}

var KiT_MatrixLayer = kit.Types.AddType(&MatrixLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "MatrixLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.On":     "true",
// 		"Layer.Inhib.Layer.Gi":     "1.5",
// 		"Layer.Inhib.Layer.FB":     "0.0",
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.3", // 0.6 in localist -- expt
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3",
// 		"Layer.Act.Init.Decay":     "0",
// 	}}

func (ly *MatrixLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Matrix.Defaults()

	// special inhib params
	ly.Inhib.Layer.On = true
	ly.Inhib.Layer.Gi = 1.5
	ly.Inhib.Layer.FB = 0
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.3 // 0.6 in localist one
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 0.25
	ly.Act.XX1.Gain = 20  // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3 // fastest
	ly.Act.Dt.GTau = 3
	ly.Act.Init.Decay = 0

	// todo: important -- need to adjust wt scale of some PFC inputs vs others:
	// drivers vs. modulators

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		if _, ok := pj.Send.(*GPLayer); ok { // From GPe TA or In
			pj.WtScale.Abs = 3
			pj.Learn.Learn = false
			pj.Learn.WtSig.Gain = 1
			pj.WtInit.Mean = 0.9
			pj.WtInit.Var = 0
			pj.WtInit.Sym = false
			if strings.HasSuffix(pj.Send.Name(), "GPeIn") {
				pj.WtScale.Abs = 1 // weaker
			}
		}
	}

	ly.UpdateParams()
}

// DALayer interface:

func (ly *MatrixLayer) GetDA() float32   { return ly.DA }
func (ly *MatrixLayer) SetDA(da float32) { ly.DA = da }

// DALrnFmDA returns effective learning dopamine value from given raw DA value
// applying Burst and Dip Gain factors, and then reversing sign for D2R.
func (ly *MatrixLayer) DALrnFmDA(da float32) float32 {
	if da > 0 {
		da *= ly.Matrix.BurstGain
	} else {
		da *= ly.Matrix.DipGain
	}
	if ly.DaR == D2R {
		da *= -1
	}
	return da
}

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *MatrixLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

func (ly *MatrixLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act.
// Matrix extends to call DALrnFmDA
func (ly *MatrixLayer) ActFmG(ltime *leabra.Time) {
	ly.DALrn = ly.DALrnFmDA(ly.DA)
	ly.Layer.ActFmG(ltime)
}
