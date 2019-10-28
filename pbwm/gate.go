// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/etable/minmax"
)

// GateState is gating state values stored in layers that receive thalamic gating signals
// including MatrixLayer, PFCLayer, GPiThal layer, etc -- use GateLayer as base layer to include.
type GateState struct {
	Act   float32         `desc:"gating activation value, reflecting current thalamic gating layer activation and sent back to corresponding Matrix and PFC layers"`
	Now   bool            `desc:"discrete gating signal -- true if Gating just occurred"`
	Cnt   int             `desc:"counter for thalamic activation value -- increments for continued maintenance or non-maintenance"`
	GeRaw minmax.AvgMax32 `desc:"average and max Ge Raw excitatory conductance values -- before being influenced by gating signals"`
}

// Init initializes the values -- call during InitActs()
func (tg *GateState) Init() {
	tg.Act = 0
	tg.Now = false
	tg.Cnt = -1
	tg.GeRaw.Init()
}

// GateLayer is a layer that cares about thalamic (BG) gating signals, and has
// slice of GateState fields that a gating layer will update.
type GateLayer struct {
	ModLayer
	GateStates []GateState `desc:"slice of gating state values for this layer -- in one-to-one correspondence with Pools (0 = layer, 1 = first Pool, etc)."`
}

func (ly *GateLayer) AsGateLayer() *GateLayer {
	return ly
}

// SetGateState sets the GateState for given pool index (starting at 1) on this layer
func (ly *GateLayer) SetGateState(poolIdx int, state *GateState) {
	gs := &ly.GateStates[poolIdx]
	*gs = *state
}

// UnitValByIdx returns value of given variable by variable index
// and flat neuron index (from layer or neuron-specific one).
// First indexes are ModNeuronVars
func (ly *GateLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	nrn := &ly.Neurons[idx]
	switch vidx {
	case DA:
		return ly.DA
	case ACh:
		return ly.ACh
	case SE:
		return ly.SE
	case GateAct:
		return ly.GateStates[nrn.SubPool].Act
	case GateNow:
		if ly.GateStates[nrn.SubPool].Now {
			return 1
		}
		return 0
	case GateCnt:
		return float32(ly.GateStates[nrn.SubPool].Cnt)
	}
	return 0
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *GateLayer) Build() error {
	err := ly.ModLayer.Build()
	if err != nil {
		return err
	}
	ly.GateStates = make([]GateState, len(ly.Pools))
	return err
}

func (ly *GateLayer) InitActs() {
	ly.ModLayer.InitActs()
	for ti := range ly.GateStates {
		tg := &ly.GateStates[ti]
		tg.Init()
	}
}

// GateLayerer is an interface for GateLayer layers
type GateLayerer interface {
	// AsGateLayer returns the layer as a GateLayer layer, for direct access to fields
	AsGateLayer() *GateLayer

	// SetGateState sets the GateState for given pool index (starting at 1) on this layer
	SetGateState(poolIdx int, state *GateState)
}
