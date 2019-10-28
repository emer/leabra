// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"log"

	"github.com/goki/ki/kit"
)

// DaSrcLayer is the basic type of layer that sends dopamine to other layers.
// Uses a list of layer names to send to -- not use Prjn infrastructure
// as it is global broadcast modulator -- individual neurons
// can use it in their own special way.
type DaSrcLayer struct {
	ModLayer
	SendTo []string `desc:"list of layers to send dopamine to"`
}

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
		ml := ly.Network.LayerByName(lnm).(*ModLayer)
		ml.DA = da
	}
}

// AddSendTo adds given layer name to list of those to send DA to
func (ly *DaSrcLayer) AddSendTo(laynm string) {
	ly.SendTo = append(ly.SendTo, laynm)
}

// Params for effects of dopamine (Da) based modulation, typically adding
// a Da-based term to the Ge excitatory synaptic input.
// Plus-phase = learning effects relative to minus-phase "performance" dopamine effects
type DaModParams struct {
	On    bool    `desc:"whether to use dopamine modulation"`
	Minus float32 `viewif:"On" desc:"how much to multiply Da in the minus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`
	Plus  float32 `viewif:"On" desc:"how much to multiply Da in the plus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`
}

//////////////////////////////////////////////////////////////////////
// Enums

// DaReceptors for D1R and D2R dopamine receptors
type DaReceptors int

//go:generate stringer -type=DaReceptors

var KiT_DaReceptors = kit.Enums.AddEnum(DaReceptorsN, false, nil)

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

var KiT_Valences = kit.Enums.AddEnum(ValencesN, false, nil)

func (ev Valences) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Valences) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// Appetititve is a positive valence US (food, water, etc)
	Appetitive Valences = iota

	// Aversive is a negative valence US (shock, threat etc)
	Aversive

	ValencesN
)
