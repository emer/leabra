// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package chans provides standard neural conductance channels for computing
a point-neuron approximation based on the standard equivalent RC circuit
model of a neuron (i.e., basic Ohms law equations).
Includes excitatory, leak, inhibition, and dynamic potassium channels.
*/
package chans

//go:generate core generate -add-types

// Chans are ion channels used in computing point-neuron activation function
type Chans struct {

	// excitatory sodium (Na) AMPA channels activated by synaptic glutamate
	E float32

	// constant leak (potassium, K+) channels -- determines resting potential (typically higher than resting potential of K)
	L float32

	// inhibitory chloride (Cl-) channels activated by synaptic GABA
	I float32

	// gated / active potassium channels -- typically hyperpolarizing relative to leak / rest
	K float32
}

// SetAll sets all the values
func (ch *Chans) SetAll(e, l, i, k float32) {
	ch.E, ch.L, ch.I, ch.K = e, l, i, k
}

// SetFromOtherMinus sets all the values from other Chans minus given value
func (ch *Chans) SetFromOtherMinus(oth Chans, minus float32) {
	ch.E, ch.L, ch.I, ch.K = oth.E-minus, oth.L-minus, oth.I-minus, oth.K-minus
}

// SetFromMinusOther sets all the values from given value minus other Chans
func (ch *Chans) SetFromMinusOther(minus float32, oth Chans) {
	ch.E, ch.L, ch.I, ch.K = minus-oth.E, minus-oth.L, minus-oth.I, minus-oth.K
}
