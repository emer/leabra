// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/leabra/deep"
)

// PBWMLayer defines the essential algorithmic API for PBWM at the layer level.
// Builds upon the deep.DeepLayer API
type PBWMLayer interface {
	deep.DeepLayer

	// AsMod returns this layer as a pbwm.ModLayer
	AsMod() *ModLayer

	// UnitValByIdx returns value of given variable by variable index
	// and flat neuron index (from layer or neuron-specific one).
	// First indexes are ModNeuronVars
	UnitValByIdx(vidx int, idx int) float32
}

// PBWMPrjn defines the essential algorithmic API for PBWM at the projection level.
// Builds upon the deep.DeepPrjn API
type PBWMPrjn interface {
	deep.DeepPrjn
}
