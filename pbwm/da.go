// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"log"

	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// DaSrcLayer is the basic type of layer that sends dopamine to other layers.
// Uses a list of layer names to send to -- not using Prjn infrastructure
// as it is global broadcast modulator -- individual neurons
// can use it in their own special way.
type DaSrcLayer struct {
	ModLayer
	SendTo []string `desc:"list of layers to send dopamine to"`
}

var KiT_DaSrcLayer = kit.Types.AddType(&DaSrcLayer{}, deep.LayerProps)

// SendToCheck is called during Build to ensure that SendTo layers are valid
func (ly *DaSrcLayer) SendToCheck() error {
	var lasterr error
	for _, lnm := range ly.SendTo {
		ly, err := ly.Network.LayerByNameTry(lnm)
		if err != nil {
			log.Printf("DaSrcLayer %s SendToCheck: %v\n", ly.Name(), err)
			lasterr = err
		}
	}
	return lasterr
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *DaSrcLayer) Build() error {
	err := ly.ModLayer.Build()
	if err != nil {
		return err
	}
	err = ly.SendToCheck()
	return err
}

// SendDA sends dopamine to SendTo list of layers
func (ly *DaSrcLayer) SendDA(da float32) {
	for _, lnm := range ly.SendTo {
		ml := ly.Network.LayerByName(lnm).(PBWMLayer).AsMod()
		ml.DA = da
	}
}

// AddSendTo adds given layer name to list of those to send DA to
func (ly *DaSrcLayer) AddSendTo(laynm string) {
	ly.SendTo = append(ly.SendTo, laynm)
}

// SendToAllBut adds all layers in network except those in list to the SendTo
// list of layers to send to -- this layer is automatically excluded as well.
func (ly *DaSrcLayer) SendToAllBut(excl []string) {
	exmap := make(map[string]struct{})
	exmap[ly.Nm] = struct{}{}
	for _, ex := range excl {
		exmap[ex] = struct{}{}
	}
	ly.SendTo = nil
	nl := ly.Network.NLayers()
	for li := 0; li < nl; li++ {
		aly := ly.Network.Layer(li)
		nm := aly.Name()
		if _, on := exmap[nm]; on {
			continue
		}
		ly.SendTo = append(ly.SendTo, nm)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  ClampDaLayer

// ClampDaLayer is an Input layer that just sends its activity as the dopamine signal
type ClampDaLayer struct {
	DaSrcLayer
}

var KiT_ClampDaLayer = kit.Types.AddType(&ClampDaLayer{}, deep.LayerProps)

// SendMods is called at end of Cycle to send modulator signals (DA, etc)
// which will then be active for the next cycle of processing
func (ly *ClampDaLayer) SendMods(ltime *leabra.Time) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA(act)
}

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

// Valences for Appetitive and Aversive valence coding
type Valences int

//go:generate stringer -type=Valences

var KiT_Valences = kit.Enums.AddEnum(ValencesN, kit.NotBitFlag, nil)

func (ev Valences) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Valences) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// Appetititve is a positive valence US (food, water, etc)
	Appetitive Valences = iota

	// Aversive is a negative valence US (shock, threat etc)
	Aversive

	ValencesN
)
