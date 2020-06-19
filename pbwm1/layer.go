// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// pbwm.Layer is the default layer type for PBWM framework, based on the ModLayer
// with dopamine modulation -- can be used for basic DA-modulated learning.
type Layer struct {
	ModLayer
	DaMod DaModParams `desc:"dopamine modulation effects, typically affecting Ge or gain -- a phase-based difference in modulation will result in learning effects through standard error-driven learning."`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, deep.LayerProps)

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *Layer) GFmInc(ltime *leabra.Time) {
	if !ly.DaMod.GeModOn() {
		ly.ModLayer.GFmInc(ltime)
		return
	}
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.GRawFmInc(nrn)
		geRaw := ly.DaMod.Ge(ly.DA, nrn.GeRaw, ltime.PlusPhase)
		ly.Act.GeFmRaw(nrn, geRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
	ly.LeabraLay.(PBWMLayer).AttnGeInc(ltime)
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *Layer) ActFmG(ltime *leabra.Time) {
	if !ly.DaMod.GainModOn() {
		ly.ModLayer.ActFmG(ltime)
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
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		ly.Learn.AvgsFmAct(nrn)
	}
	ly.Act.XX1.Gain = curGain
	ly.Act.XX1.Update()
}
