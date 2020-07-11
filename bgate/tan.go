// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"log"

	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/rl"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// TANLayer (tonically active neuron) reads reward signals from a named source layer
// and sends the absolute value of that activity as the positively-rectified
// non-prediction-discounted reward signal computed TANs, and sent as
// an acetylcholine signal.
type TANLayer struct {
	leabra.Layer
	RewLay  string     `desc:"name of Reward-representing layer from which this computes ACh as absolute value"`
	SendACh rl.SendACh `desc:"list of layers to send acetylcholine to"`
	ACh     float32    `desc:"acetylcholine value for this layer"`
}

var KiT_TANLayer = kit.Types.AddType(&TANLayer{}, leabra.LayerProps)

func (ly *TANLayer) Defaults() {
	ly.Layer.Defaults()
	ly.RewLay = "Rew"
}

// AChLayer interface:

func (ly *TANLayer) GetACh() float32    { return ly.ACh }
func (ly *TANLayer) SetACh(ach float32) { ly.ACh = ach }

// RewLayer returns the reward layer based on name
func (ly *TANLayer) RewLayer() (*leabra.Layer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewLay)
	if err != nil {
		log.Printf("TANLayer %s, RewLay: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(leabra.LeabraLayer).AsLeabra(), nil
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *TANLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendACh.Validate(ly.Network, ly.Name()+" SendTo list")
	_, err = ly.RewLayer()
	return err
}

func (ly *TANLayer) ActFmG(ltime *leabra.Time) {
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
func (ly *TANLayer) CyclePost(ltime *leabra.Time) {
	act := ly.Neurons[0].Act
	ly.ACh = act
	ly.SendACh.SendACh(ly.Network, act)
}
