// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"fmt"
	"log"

	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/rl"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// CINLayer (cholinergic interneuron) reads reward signals from a named source layer
// and sends the absolute value of that activity as the positively-rectified
// non-prediction-discounted reward signal computed by CINs, and sent as
// an acetylcholine (ACh) signal.
type CINLayer struct {
	leabra.Layer
	RewLay  string     `desc:"name of Reward-representing layer from which this computes ACh as absolute value"`
	SendACh rl.SendACh `desc:"list of layers to send acetylcholine to"`
	ACh     float32    `desc:"acetylcholine value for this layer"`
}

var KiT_CINLayer = kit.Types.AddType(&CINLayer{}, leabra.LayerProps)

func (ly *CINLayer) Defaults() {
	ly.Layer.Defaults()
	if ly.RewLay == "" {
		ly.RewLay = "Rew"
	}
}

// AChLayer interface:

func (ly *CINLayer) GetACh() float32    { return ly.ACh }
func (ly *CINLayer) SetACh(ach float32) { ly.ACh = ach }

// RewLayer returns the reward layer based on name
func (ly *CINLayer) RewLayer() (*leabra.Layer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewLay)
	if err != nil {
		log.Printf("CINLayer %s, RewLay: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(leabra.LeabraLayer).AsLeabra(), nil
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *CINLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendACh.Validate(ly.Network, ly.Name()+" SendTo list")
	_, err = ly.RewLayer()
	return err
}

func (ly *CINLayer) ActFmG(ltime *leabra.Time) {
	rly, _ := ly.RewLayer()
	if rly == nil {
		return
	}
	rnrn := &(rly.Neurons[0])
	ract := rnrn.Act
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = mat32.Abs(ract)
	}
}

// CyclePost is called at end of Cycle
// We use it to send ACh, which will then be active for the next cycle of processing.
func (ly *CINLayer) CyclePost(ltime *leabra.Time) {
	act := ly.Neurons[0].Act
	ly.ACh = act
	ly.SendACh.SendACh(ly.Network, act)
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *CINLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	if varNm != "ACh" {
		return -1, fmt.Errorf("bgate.NeuronVars: variable named: %s not found", varNm)
	}
	nn := len(leabra.NeuronVars)
	return nn, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *CINLayer) UnitVal1D(varIdx int, idx int) float32 {
	nn := len(leabra.NeuronVars)
	if varIdx < 0 || varIdx > nn { // nn = ACh
		return math32.NaN()
	}
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	if varIdx > nn {
		return math32.NaN()
	}
	return ly.ACh
}
