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

	//	CHLPath implements Contrastive Hebbian Learning.
	CHLPath

	//	EcCa1Path implements special learning for EC <-> CA1 pathways
	// in the hippocampus to perform error-driven learning of this
	// encoder pathway according to the ThetaPhase algorithm.
	// uses Contrastive Hebbian Learning (CHL) on ActP - ActQ1
	// Q1: ECin -> CA1 -> ECout       : ActQ1 = minus phase for auto-encoder
	// Q2, 3: CA3 -> CA1 -> ECout     : ActM = minus phase for recall
	// Q4: ECin -> CA1, ECin -> ECout : ActP = plus phase for everything
	EcCa1Path

	//////// RL

	// RWPath does dopamine-modulated learning for reward prediction: Da * Send.Act
	// Use in RWPredLayer typically to generate reward predictions.
	// Has no weight bounds or limits on sign etc.
	RWPath

	// TDRewPredPath does dopamine-modulated learning for reward prediction:
	// DWt = Da * Send.ActQ0 (activity on *previous* timestep)
	// Use in TDRewPredLayer typically to generate reward predictions.
	// Has no weight bounds or limits on sign etc.
	TDRewPredPath

	//////// PBWM

	// MatrixPath does dopamine-modulated, gated trace learning,
	// for Matrix learning in PBWM context.
	MatrixPath

	// GPiThalPath accumulates per-path raw conductance that is needed for
	// separately weighting NoGo vs. Go inputs.
	GPiThalPath
)
