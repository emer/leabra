// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

//gosl:start pathtypes

// PathTypes enumerates all the different types of leabra pathways,
// for the different algorithm types supported.
// Class parameter styles automatically key off of these types.
type PathTypes int32 //enums:enum

// The pathway types
const (
	// Forward is a feedforward, bottom-up pathway from sensory inputs to higher layers
	ForwardPath PathTypes = iota

	// Back is a feedback, top-down pathway from higher layers back to lower layers
	BackPath

	// Lateral is a lateral pathway within the same layer / area
	LateralPath

	// Inhib is an inhibitory pathway that drives inhibitory
	// synaptic conductances instead of the default excitatory ones.
	InhibPath

	// CTCtxt are pathways from Superficial layers to CT layers that
	// send Burst activations drive updating of CtxtGe excitatory conductance,
	// at end of plus (51B Bursting) phase.  Biologically, this pathway
	// comes from the PT layer 5IB neurons, but it is simpler to use the
	// Super neurons directly, and PT are optional for most network types.
	// These pathways also use a special learning rule that
	// takes into account the temporal delays in the activation states.
	// Can also add self context from CT for deeper temporal context.
	CTCtxtPath
)
