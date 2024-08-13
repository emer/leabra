// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"fmt"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
	"github.com/emer/leabra/v2/rl"
)

// CINLayer (cholinergic interneuron) reads reward signals from named source layer(s)
// and sends the Max absolute value of that activity as the positively rectified
// non-prediction-discounted reward signal computed by CINs, and sent as
// an acetylcholine (ACh) signal.
// To handle positive-only reward signals, need to include both a reward prediction
// and reward outcome layer.
type CINLayer struct {
	leabra.Layer

	// threshold on reward values from RewLays, to count as a significant reward event, which then drives maximal ACh -- set to 0 to disable this nonlinear behavior
	RewThr float32

	// Reward-representing layer(s) from which this computes ACh as Max absolute value
	RewLays emer.LayNames

	// list of layers to send acetylcholine to
	SendACh rl.SendACh

	// acetylcholine value for this layer
	ACh float32
}

func (ly *CINLayer) Defaults() {
	ly.Layer.Defaults()
	ly.RewThr = 0.1
}

// AChLayer interface:

func (ly *CINLayer) GetACh() float32    { return ly.ACh }
func (ly *CINLayer) SetACh(ach float32) { ly.ACh = ach }

// Build constructs the layer state, including calling Build on the pathways.
func (ly *CINLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendACh.Validate(ly.Network, ly.Name()+" SendTo list")
	err = ly.RewLays.Validate(ly.Network, ly.Name()+" RewLays list")
	return err
}

// MaxAbsRew returns the maximum absolute value of reward layer activations
func (ly *CINLayer) MaxAbsRew() float32 {
	mx := float32(0)
	for _, nm := range ly.RewLays {
		lyi := ly.Network.LayerByName(nm)
		if lyi == nil {
			continue
		}
		ly := lyi.(leabra.LeabraLayer).AsLeabra()
		act := math32.Abs(ly.Pools[0].Inhib.Act.Max)
		mx = math32.Max(mx, act)
	}
	return mx
}

func (ly *CINLayer) ActFmG(ltime *leabra.Time) {
	ract := ly.MaxAbsRew()
	if ly.RewThr > 0 {
		if ract > ly.RewThr {
			ract = 1
		}
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		nrn.Act = ract
	}
}

// CyclePost is called at end of Cycle
// We use it to send ACh, which will then be active for the next cycle of processing.
func (ly *CINLayer) CyclePost(ltime *leabra.Time) {
	act := ly.Neurons[0].Act
	ly.ACh = act
	ly.SendACh.SendACh(ly.Network, act)
}

// UnitVarIndex returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *CINLayer) UnitVarIndex(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIndex(varNm)
	if err == nil {
		return vidx, err
	}
	if varNm != "ACh" {
		return -1, fmt.Errorf("pcore.CINLayer: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn, nil
}

// UnitValue1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *CINLayer) UnitValue1D(varIndex int, idx int, di int) float32 {
	nn := ly.Layer.UnitVarNum()
	if varIndex < 0 || varIndex > nn { // nn = ACh
		return math32.NaN()
	}
	if varIndex < nn {
		return ly.Layer.UnitValue1D(varIndex, idx, di)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	return ly.ACh
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *CINLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 1
}
