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
	// and maps onto the Hidden type in LayerTypes.
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

	//////// Deep

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

	///////// Neuromodulation & RL

	// ClampDaLayer is an Input layer that just sends its activity
	// as the dopamine signal.
	ClampDaLayer

	// RWPredLayer computes reward prediction for a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the PVLV framework).
	// Activity is computed as linear function of excitatory conductance
	// (which can be negative -- there are no constraints).
	// Use with [RWPath] which does simple delta-rule learning on minus-plus.
	RWPredLayer

	// RWDaLayer computes a dopamine (DA) signal based on a simple Rescorla-Wagner
	// learning dynamic (i.e., PV learning in the PVLV framework).
	// It computes difference between r(t) and [RWPredLayer] values.
	// r(t) is accessed directly from a Rew layer -- if no external input then no
	// DA is computed -- critical for effective use of RW only for PV cases.
	// RWPred prediction is also accessed directly from Rew layer to avoid any issues.
	RWDaLayer

	// TDPredLayer is the temporal differences reward prediction layer.
	// It represents estimated value V(t) in the minus phase, and computes
	// estimated V(t+1) based on its learned weights in plus phase.
	// Use [TDPredPath] for DA modulated learning.
	TDPredLayer

	// TDIntegLayer is the temporal differences reward integration layer.
	// It represents estimated value V(t) in the minus phase, and
	// estimated V(t+1) + r(t) in the plus phase.
	// It computes r(t) from (typically fixed) weights from a reward layer,
	// and directly accesses values from [TDPredLayer].
	TDIntegLayer

	// TDDaLayer computes a dopamine (DA) signal as the temporal difference (TD)
	// between the [TDIntegLayer[] activations in the minus and plus phase.
	TDDaLayer

	///////// BG Basal Ganglia

	// MatrixLayer represents the dorsal matrisome MSN's that are the main
	// Go / NoGo gating units in BG driving updating of PFC WM in PBWM.
	// D1R = Go, D2R = NoGo, and outer 4D Pool X dimension determines GateTypes per MaintN
	// (Maint on the left up to MaintN, Out on the right after)
	MatrixLayer

	// GPeLayer is a Globus pallidus external layer, a key region of the basal ganglia.
	// It does not require any additional mechanisms beyond the SuperLayer.
	GPeLayer

	// GPiThalLayer represents the combined Winner-Take-All dynamic of GPi (SNr) and Thalamus.
	// It is the final arbiter of gating in the BG, weighing Go (direct) and NoGo (indirect)
	// inputs from MatrixLayers (indirectly via GPe layer in case of NoGo).
	// Use 4D structure for this so it matches 4D structure in Matrix layers
	GPiThalLayer

	// CINLayer (cholinergic interneuron) reads reward signals from named source layer(s)
	// and sends the Max absolute value of that activity as the positively rectified
	// non-prediction-discounted reward signal computed by CINs, and sent as
	// an acetylcholine (ACh) signal.
	// To handle positive-only reward signals, need to include both a reward prediction
	// and reward outcome layer.
	CINLayer

	///////// PFC Prefrontal Cortex

	// PFCLayer is a prefrontal cortex layer, either superficial or output.
	// See [PFCDeepLayer] for the deep maintenance layer.
	PFCLayer

	// PFCDeepLayer is a prefrontal cortex deep maintenance layer.
	PFCDeepLayer
)
