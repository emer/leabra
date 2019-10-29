// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
)

// PBWMLayer defines the essential algorithmic API for PBWM at the layer level.
// Builds upon the deep.DeepLayer API
type PBWMLayer interface {
	deep.DeepLayer

	// AsMod returns this layer as a pbwm.ModLayer (minimum layer in PBWM)
	AsMod() *ModLayer

	// AsGate returns this layer as a pbwm.GateLayer (gated layer type) -- nil if not impl
	AsGate() *GateLayer

	// UnitValByIdx returns value of given variable by variable index
	// and flat neuron index (from layer or neuron-specific one).
	// First indexes are ModNeuronVars
	UnitValByIdx(vidx NeuronVars, idx int) float32

	// GateSend updates gating state and sends it along to other layers.
	// Called after std Cycle methods.
	// Only implemented for gating layers.
	GateSend(ltime *leabra.Time)

	// RecGateAct records the gating activation from current activation, when gating occcurs
	// based on GateState.Now
	RecGateAct(ltime *leabra.Time)
}

// PBWMPrjn defines the essential algorithmic API for PBWM at the projection level.
// Builds upon the deep.DeepPrjn API
type PBWMPrjn interface {
	deep.DeepPrjn
}
