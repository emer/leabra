// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/leabra/v2/leabra"
)

// PBWMLayer defines the essential algorithmic API for PBWM at the layer level.
// Builds upon the leabra.LeabraLayer API
type PBWMLayer interface {
	leabra.LeabraLayer

	// AsPBWM returns this layer as a pbwm.Layer (base Layer in PBWM)
	AsPBWM() *Layer

	// AsGate returns this layer as a pbwm.GateLayer (gated layer type) -- nil if not impl
	AsGate() *GateLayer

	// UnitValueByIndex returns value of given PBWM-specific variable by variable index
	// and flat neuron index (from layer or neuron-specific one).
	UnitValueByIndex(vidx NeurVars, idx int) float32

	// GateSend updates gating state and sends it along to other layers.
	// Called after std Cycle methods.
	// Only implemented for gating layers.
	GateSend(ltime *leabra.Time)

	// RecGateAct records the gating activation from current activation, when gating occcurs
	// based on GateState.Now
	RecGateAct(ltime *leabra.Time)

	// SendMods is called at end of Cycle to send modulator signals (DA, etc)
	// which will then be active for the next cycle of processing
	SendMods(ltime *leabra.Time)

	// Quarter2DWt is optional Q2 DWt -- PFC and matrix layers can do this as appropriate
	Quarter2DWt()

	// DoQuarter2DWt returns true if this recv layer should have its weights updated
	DoQuarter2DWt() bool
}
