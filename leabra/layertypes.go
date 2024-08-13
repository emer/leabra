// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

// LayerTypes enumerates all the different types of layers,
// for the different algorithm types supported.
// Class parameter styles automatically key off of these types.
type LayerTypes int32 //enums:enum

// note: we need to add the Layer extension to avoid naming
// conflicts between layer, pathway and other things.

// The layer types
const (
	// Super is a superficial cortical layer (lamina 2-3-4)
	// which does not receive direct input or targets.
	// In more generic models, it should be used as a Hidden layer,
	// and maps onto the Hidden type in emer.LayerType.
	SuperLayer LayerTypes = iota

	// Input is a layer that receives direct external input
	// in its Ext inputs.  Biologically, it can be a primary
	// sensory layer, or a thalamic layer.
	InputLayer

	// Target is a layer that receives direct external target inputs
	// used for driving plus-phase learning.
	// Simple target layers are generally not used in more biological
	// models, which instead use predictive learning via Pulvinar
	// or related mechanisms.
	TargetLayer

	// Compare is a layer that receives external comparison inputs,
	// which drive statistics but do NOT drive activation
	// or learning directly.  It is rarely used in axon.
	CompareLayer

	/////////////
	// Deep

	// CT are layer 6 corticothalamic projecting neurons,
	// which drive "top down" predictions in Pulvinar layers.
	// They maintain information over time via stronger NMDA
	// channels and use maintained prior state information to
	// generate predictions about current states forming on Super
	// layers that then drive PT (5IB) bursting activity, which
	// are the plus-phase drivers of Pulvinar activity.
	CTLayer

	// Pulvinar are thalamic relay cell neurons in the higher-order
	// Pulvinar nucleus of the thalamus, and functionally isomorphic
	// neurons in the MD thalamus, and potentially other areas.
	// These cells alternately reflect predictions driven by CT pathways,
	// and actual outcomes driven by 5IB Burst activity from corresponding
	// PT or Super layer neurons that provide strong driving inputs.
	PulvinarLayer

	// TRNLayer is thalamic reticular nucleus layer for inhibitory competition
	// within the thalamus.
	TRNLayer

	// PTMaintLayer implements the subset of pyramidal tract (PT)
	// layer 5 intrinsic bursting (5IB) deep neurons that exhibit
	// robust, stable maintenance of activity over the duration of a
	// goal engaged window, modulated by basal ganglia (BG) disinhibitory
	// gating, supported by strong MaintNMDA channels and recurrent excitation.
	// The lateral PTSelfMaint pathway uses MaintG to drive GMaintRaw input
	// that feeds into the stronger, longer MaintNMDA channels,
	// and the ThalToPT ModulatoryG pathway from BGThalamus multiplicatively
	// modulates the strength of other inputs, such that only at the time of
	// BG gating are these strong enough to drive sustained active maintenance.
	// Use Act.Dend.ModGain to parameterize.
	PTMaintLayer

	// PTPredLayer implements the subset of pyramidal tract (PT)
	// layer 5 intrinsic bursting (5IB) deep neurons that combine
	// modulatory input from PTMaintLayer sustained maintenance and
	// CTLayer dynamic predictive learning that helps to predict
	// state changes during the period of active goal maintenance.
	// This layer provides the primary input to VSPatch US-timing
	// prediction layers, and other layers that require predictive dynamic
	PTPredLayer
}
