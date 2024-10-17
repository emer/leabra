// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

// SendDA sends dopamine to SendTo list of layers.
func (ly *Layer) SendDA(da float32) {
	for _, lnm := range ly.SendTo {
		tly := ly.Network.LayerByName(lnm)
		if tly != nil {
			tly.DA = da
		}
	}
}

// SendACh sends ACh to SendTo list of layers.
func (ly *Layer) SendACh(ach float32) {
	for _, lnm := range ly.SendTo {
		tly := ly.Network.LayerByName(lnm)
		if tly != nil {
			tly.ACh = ach
		}
	}
}

////////  ClampDaLayer

// AddClampDaLayer adds a ClampDaLayer of given name
func (nt *Network) AddClampDaLayer(name string) *Layer {
	return nt.AddLayer2D(name, 1, 1, ClampDaLayer)
}

func (ly *Layer) ClampDaDefaults() {
	ly.Act.Clamp.Range.Set(-1, 1)
}

// SendDaFromAct is called in SendMods to send activity as DA.
func (ly *Layer) SendDaFromAct(ctx *Context) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA(act)
}

// Params for effects of dopamine (Da) based modulation, typically adding
// a Da-based term to the Ge excitatory synaptic input.
// Plus-phase = learning effects relative to minus-phase "performance" dopamine effects
type DaModParams struct {

	// whether to use dopamine modulation
	On bool

	// modulate gain instead of Ge excitatory synaptic input
	ModGain bool

	// how much to multiply Da in the minus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons
	Minus float32

	// how much to multiply Da in the plus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons
	Plus float32

	// for negative dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * NegNain) -- da is multiplied by minus or plus depending on phase
	NegGain float32

	// for positive dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * PosGain) -- da is multiplied by minus or plus depending on phase
	PosGain float32
}

func (dm *DaModParams) Defaults() {
	dm.Minus = 0
	dm.Plus = 0.01
	dm.NegGain = 0.1
	dm.PosGain = 0.1
}

func (dm *DaModParams) Update() {
}

func (dm *DaModParams) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	case "NegGain", "PosGain":
		return dm.On && dm.ModGain
	default:
		return dm.On
	}
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

//////// Enums

// DaReceptors for D1R and D2R dopamine receptors
type DaReceptors int32 //enums:enum

const (
	// D1R primarily expresses Dopamine D1 Receptors -- dopamine is excitatory and bursts of dopamine lead to increases in synaptic weight, while dips lead to decreases -- direct pathway in dorsal striatum
	D1R DaReceptors = iota

	// D2R primarily expresses Dopamine D2 Receptors -- dopamine is inhibitory and bursts of dopamine lead to decreases in synaptic weight, while dips lead to increases -- indirect pathway in dorsal striatum
	D2R
)

// Valences for Appetitive and Aversive valence coding
type Valences int32 //enums:enum

const (
	// Appetititve is a positive valence US (food, water, etc)
	Appetitive Valences = iota

	// Aversive is a negative valence US (shock, threat etc)
	Aversive
)
