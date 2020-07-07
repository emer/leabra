// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"strings"

	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// GPLayer represents the dorsal matrisome MSN's that are the main
// Go / NoGo gating units in BG.  D1R = Go, D2R = NoGo.
type GPLayer struct {
	leabra.Layer
	DA float32 `inactive:"+" desc:"dopamine value for this layer"`
}

var KiT_GPLayer = kit.Types.AddType(&GPLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "GPLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.9",
// 		"Layer.Act.Init.Act":  "0.5",
// 		"Layer.Act.Erev.L":    "0.8",
// 		"Layer.Act.Gbar.L":    "0.3", // 0.2 orig
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3", // 5 orig
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *GPLayer) Defaults() {
	ly.Layer.Defaults()
	ly.DA = 0

	// GP is tonically self-active and has no FFFB inhibition

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
	ly.Act.XX1.Gain = 20  // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3 // fastest
	ly.Act.Dt.GTau = 3
	ly.Act.Init.Decay = 0

	for _, pjii := range ly.RcvPrjns {
		pji := pjii.(leabra.LeabraPrjn)
		pj := pji.AsLeabra()
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.9
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		if _, ok := pji.(*GPeInPrjn); ok {
			if ml, ok := pj.Send.(*MatrixLayer); ok { // MtxGoToGPeIn is key off-beam WTA pathway
				if ml.DaR == D1R {
					pj.WtScale.Abs = .2 // .2 is lowest for effective WTA -- .1 allows more stripes to gate -- but lower is better for less complex dynamics.
				} else { // MtxNoToGPeIn -- primary NoGo pathway
					pj.WtScale.Abs = 1.5 // 2.5 orig -- 1.5 is about as low as you can go
				}
			} else if _, ok := pj.Send.(*GPLayer); ok { // GPeOutToGPeIn
				pj.WtScale.Abs = 1.5
			}
			continue
		}
		if _, ok := pji.(*GPiPrjn); ok {
			continue
		}
		pj.Learn.Learn = false

		if _, ok := pj.Send.(*MatrixLayer); ok {
			pj.WtScale.Abs = 0.5
		} else if _, ok := pj.Send.(*STNpLayer); ok {
			pj.WtScale.Abs = 0.1 // 0.1 orig
		}

		switch {
		case strings.HasSuffix(ly.Nm, "GPeOut"):
			if _, ok := pj.Send.(*MatrixLayer); ok { // MtxGoToGPeOut
				pj.WtScale.Abs = 0.5 // Go firing threshold
			}
		case strings.HasSuffix(ly.Nm, "GPeIn"):
			if _, ok := pj.Send.(*STNpLayer); ok { // STNToGPeIn
				pj.WtScale.Abs = 0.5 // todo: 0.1 default
			}
		case strings.HasSuffix(ly.Nm, "GPeTA"):
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToGPeTA
				pj.WtScale.Abs = 1 // 1 by default
			}
		}
	}

	ly.UpdateParams()
}

// DALayer interface:

func (ly *GPLayer) GetDA() float32   { return ly.DA }
func (ly *GPLayer) SetDA(da float32) { ly.DA = da }

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *GPLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

func (ly *GPLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}

//////////////////////////////////////////////////////////////////////
// GPeInPrjn

// GPeInPrjn must be used with GPLayer.
// Learns from DA and gating status.
type GPeInPrjn struct {
	leabra.Prjn
}

var KiT_GPeInPrjn = kit.Types.AddType(&GPeInPrjn{}, leabra.PrjnProps)

func (pj *GPeInPrjn) Defaults() {
	pj.Prjn.Defaults()
	// no additional factors
	pj.WtInit.Mean = 0.9
	pj.WtInit.Var = 0
	pj.WtInit.Sym = false
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
	pj.Learn.Lrate = 0.01
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *GPeInPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(*GPLayer)

	da := rlay.DA

	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]

			dwt := da * rn.Act * sn.Act

			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFmAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pj.Learn.Momentum.On {
				dwt = norm * pj.Learn.Momentum.MomentFmDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pj.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pj.Learn.Norm.On {
			maxNorm := float32(0)
			for ci := range syns {
				sy := &syns[ci]
				if sy.Norm > maxNorm {
					maxNorm = sy.Norm
				}
			}
			for ci := range syns {
				sy := &syns[ci]
				sy.Norm = maxNorm
			}
		}
	}
}
