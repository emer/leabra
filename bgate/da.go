// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"github.com/goki/ki/kit"
)

//////////////////////////////////////////////////////////////////////////////////////
//  DaMod

// Params for effects of dopamine (Da) based modulation, typically adding
// a Da-based term to the Ge excitatory synaptic input.
// Plus-phase = learning effects relative to minus-phase "performance" dopamine effects
type DaModParams struct {
	On      bool    `desc:"whether to use dopamine modulation"`
	ModGain bool    `viewif:"On" desc:"modulate gain instead of Ge excitatory synaptic input"`
	Minus   float32 `viewif:"On" desc:"how much to multiply Da in the minus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`
	Plus    float32 `viewif:"On" desc:"how much to multiply Da in the plus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`
	NegGain float32 `viewif:"On&&ModGain" desc:"for negative dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * NegNain) -- da is multiplied by minus or plus depending on phase"`
	PosGain float32 `viewif:"On&&ModGain" desc:"for positive dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * PosGain) -- da is multiplied by minus or plus depending on phase"`
}

func (dm *DaModParams) Defaults() {
	dm.Minus = 0
	dm.Plus = 0.01
	dm.NegGain = 0.1
	dm.PosGain = 0.1
}

// GeModOn returns true if modulating Ge
func (dm *DaModParams) GeModOn() bool {
	return dm.On && !dm.ModGain
}

// GainModOn returns true if modulating Gain
func (dm *DaModParams) GainModOn() bool {
	return dm.On && dm.ModGain
}

// Ge returns da-modulated ge value
func (dm *DaModParams) Ge(da, ge float32, plusPhase bool) float32 {
	if plusPhase {
		return dm.Plus * da * ge
	} else {
		return dm.Minus * da * ge
	}
}

// Gain returns da-modulated gain value
func (dm *DaModParams) Gain(da, gain float32, plusPhase bool) float32 {
	if plusPhase {
		da *= dm.Plus
	} else {
		da *= dm.Minus
	}
	if da < 0 {
		return gain * (1 + da*dm.NegGain)
	} else {
		return gain * (1 + da*dm.PosGain)
	}
}

//////////////////////////////////////////////////////////////////////
// Enums

// DaReceptors for D1R and D2R dopamine receptors
type DaReceptors int

//go:generate stringer -type=DaReceptors

var KiT_DaReceptors = kit.Enums.AddEnum(DaReceptorsN, kit.NotBitFlag, nil)

func (ev DaReceptors) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *DaReceptors) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// D1R primarily expresses Dopamine D1 Receptors -- dopamine is excitatory and bursts of dopamine lead to increases in synaptic weight, while dips lead to decreases -- direct pathway in dorsal striatum
	D1R DaReceptors = iota

	// D2R primarily expresses Dopamine D2 Receptors -- dopamine is inhibitory and bursts of dopamine lead to decreases in synaptic weight, while dips lead to increases -- indirect pathway in dorsal striatum
	D2R

	DaReceptorsN
)
