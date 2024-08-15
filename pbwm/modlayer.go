// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/leabra/v2/leabra"
)

// ModLayer provides DA modulated learning to basic Leabra layers.
type ModLayer struct {
	Layer

	// dopamine modulation effects, typically affecting Ge or gain -- a phase-based difference in modulation will result in learning effects through standard error-driven learning.
	DaMod DaModParams
}

// GFromInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *ModLayer) GFromInc(ltime *leabra.Time) {
	if !ly.DaMod.GeModOn() {
		ly.Layer.GFromInc(ltime)
		return
	}
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		geRaw := ly.DaMod.Ge(ly.DA, nrn.GeRaw, ltime.PlusPhase)
		ly.Act.GeFromRaw(nrn, geRaw)
		ly.Act.GiFromRaw(nrn, nrn.GiRaw)
	}
}

// ActFromG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *ModLayer) ActFromG(ltime *leabra.Time) {
	if !ly.DaMod.GainModOn() {
		ly.Layer.ActFromG(ltime)
		return
	}
	curGain := ly.Act.XX1.Gain
	ly.Act.XX1.Gain = ly.DaMod.Gain(ly.DA, curGain, ltime.PlusPhase)
	ly.Act.XX1.Update()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFromG(nrn)
		ly.Act.ActFromG(nrn)
		ly.Learn.AvgsFromAct(nrn)
	}
	ly.Act.XX1.Gain = curGain
	ly.Act.XX1.Update()
}
