// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"errors"
	"fmt"
)

// LayerNames is a list of layer names, with methods to add and validate.
type LayerNames []string

// Add adds given layer name(s) to list.
func (ln *LayerNames) Add(laynm ...string) {
	*ln = append(*ln, laynm...)
}

// AddAllBut adds all layers in network except those in exclude list.
func (ln *LayerNames) AddAllBut(net *Network, excl ...string) {
	*ln = nil
	for _, l := range net.Layers {
		lnm := l.Name
		exl := false
		for _, ex := range excl {
			if lnm == ex {
				exl = true
				break
			}
		}
		if exl {
			continue
		}
		ln.Add(lnm)
	}
}

// Validate ensures that layer names are valid.
func (ln *LayerNames) Validate(net *Network) error {
	var errs []error
	for _, lnm := range *ln {
		tly := net.LayerByName(lnm)
		if tly == nil {
			err := fmt.Errorf("Validate: Layer name found %s", lnm)
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// SendDA sends dopamine to SendTo list of layers.
func (ly *Layer) SendDA(da float32) {
	for _, lnm := range ly.SendTo {
		tly := ly.Network.LayerByName(lnm)
		if tly != nil {
			tly.NeuroMod.DA = da
		}
	}
}

// SendACh sends ACh to SendTo list of layers.
func (ly *Layer) SendACh(ach float32) {
	for _, lnm := range ly.SendTo {
		tly := ly.Network.LayerByName(lnm)
		if tly != nil {
			tly.NeuroMod.ACh = ach
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
	ly.NeuroMod.DA = act
	ly.SendDA(act)
}

// NeuroMod are the neuromodulatory neurotransmitters, at the layer level.
type NeuroMod struct {

	// DA is dopamine, which primarily modulates learning, and also excitability,
	// and reflects the reward prediction error (RPE).
	DA float32

	// ACh is acetylcholine, which modulates excitability and also learning,
	// and reflects salience, i.e., reward (without discount by prediction) and
	// learned CS onset.
	ACh float32

	// SE is serotonin, which is a longer timescale neuromodulator with many
	// different effects. Currently not implemented, but here for future expansion.
	SE float32
}

func (nm *NeuroMod) Init() {
	nm.DA = 0
	nm.ACh = 0
	nm.SE = 0
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
