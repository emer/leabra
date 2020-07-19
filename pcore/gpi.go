// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"strings"

	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// GPiLayer represents the GPi / SNr output nucleus of the BG.
// It gets inhibited by the MtxGo and GPeIn layers, and its minimum
// activation during this inhibition is recorded in ActLrn, for learning.
// Typically just a single unit per Pool representing a given stripe.
type GPiLayer struct {
	GPLayer
}

var KiT_GPiLayer = kit.Types.AddType(&GPiLayer{}, leabra.LayerProps)

func (ly *GPiLayer) Defaults() {
	ly.GPLayer.Defaults()
	ly.GPLay = GPi

	// note: GPLayer took care of STN input prjns

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.5
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		pj.Learn.Learn = false
		pj.Learn.Norm.On = false
		pj.Learn.Momentum.On = false
		if _, ok := pj.Send.(*MatrixLayer); ok { // MtxGoToGPi
			pj.WtScale.Abs = 0.8 // slightly weaker than GPeIn
		} else if _, ok := pj.Send.(*GPLayer); ok { // GPeInToGPi
			pj.WtScale.Abs = 1 // stronger because integrated signal, also act can be weaker
		} else if strings.HasSuffix(pj.Send.Name(), "STNp") { // STNpToGPi
			pj.WtScale.Abs = 1
		} else if strings.HasSuffix(pj.Send.Name(), "STNs") { // STNsToGPi
			pj.WtScale.Abs = 0.2
		}
	}

	ly.UpdateParams()
}
