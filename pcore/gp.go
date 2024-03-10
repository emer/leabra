// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/leabra/v2/leabra"
)

// GPLayer represents a globus pallidus layer, including:
// GPeOut, GPeIn, GPeTA (arkypallidal), and GPi (see GPLay for type).
// Typically just a single unit per Pool representing a given stripe.
type GPLayer struct {
	Layer

	// type of GP layer
	GPLay GPLays
}

// Defaults in param.Sheet format
// Sel: "GPLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.57",
// 		"Layer.Act.Init.Act":  "0.65",
// 		"Layer.Act.Erev.L":    "0.8",
// 		"Layer.Act.Gbar.L":    "0.3",
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3", // 5 orig
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *GPLayer) Defaults() {
	ly.Layer.Defaults()
	ly.DA = 0

	// GP is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Vm = 0.57
	ly.Act.Init.Act = 0.67
	ly.Act.Erev.L = 0.8
	ly.Act.Gbar.L = 0.3
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.Self.Tau = 3.0
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 0.25
	ly.Act.XX1.Gain = 20  // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3 // fastest
	ly.Act.Dt.GTau = 3
	ly.Act.Init.Decay = 0

	switch ly.GPLay {
	case GPeIn:
		ly.Act.Init.Act = 0.81
		ly.Act.Init.Vm = 0.60
	case GPeTA:
		ly.Act.Init.Act = 0.26
		ly.Act.Init.Vm = 0.50
	}

	for _, pjii := range ly.RcvPrjns {
		pji := pjii.(leabra.LeabraPrjn)
		pj := pji.AsLeabra()
		pj.Learn.Learn = false
		pj.Learn.Norm.On = false
		pj.Learn.Momentum.On = false
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.9
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		if _, ok := pj.Send.(*MatrixLayer); ok {
			pj.WtScale.Abs = 0.5
		} else if _, ok := pj.Send.(*STNLayer); ok {
			pj.WtScale.Abs = 0.1 // default level for GPeOut and GPeTA -- weaker to not oppose GPeIn surge
		}
		switch ly.GPLay {
		case GPeIn:
			if _, ok := pj.Send.(*MatrixLayer); ok { // MtxNoToGPeIn -- primary NoGo pathway
				pj.WtScale.Abs = 1
			} else if _, ok := pj.Send.(*GPLayer); ok { // GPeOutToGPeIn
				pj.WtScale.Abs = 0.5
			}
			if _, ok := pj.Send.(*STNLayer); ok { // STNpToGPeIn -- stronger to drive burst of activity
				pj.WtScale.Abs = 0.5
			}
		case GPeOut:
		case GPeTA:
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToGPeTA
				pj.WtScale.Abs = 0.9 // just enough to knock down to near-zero at baseline
			}
		}
	}

	ly.UpdateParams()
}

//////////////////////////////////////////////////////////////////////
//  GPLays

// GPLays for GPLayer type
type GPLays int //enums:enum

const (
	// GPeOut is Outer layer of GPe neurons, receiving inhibition from MtxGo
	GPeOut GPLays = iota

	// GPeIn is Inner layer of GPe neurons, receiving inhibition from GPeOut and MtxNo
	GPeIn

	// GPeTA is arkypallidal layer of GPe neurons, receiving inhibition from GPeIn
	// and projecting inhibition to Mtx
	GPeTA

	// GPi is the inner globus pallidus, functionally equivalent to SNr,
	// receiving from MtxGo and GPeIn, and sending inhibition to VThal
	GPi

	GPLaysN
)
