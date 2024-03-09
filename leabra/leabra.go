// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/etable/v2/etensor"
)

// LeabraNetwork defines the essential algorithmic API for Leabra, at the network level.
// These are the methods that the user calls in their Sim code:
// * AlphaCycInit
// * Cycle
// * QuarterFinal
// * DWt
// * WtFmDwt
// Because we don't want to have to force the user to use the interface cast in calling
// these methods, we provide Impl versions here that are the implementations
// which the user-facing method calls.
//
// Typically most changes in algorithm can be accomplished directly in the Layer
// or Prjn level, but sometimes (e.g., in deep) additional full-network passes
// are required.
//
// All of the structural API is in emer.Network, which this interface also inherits for
// convenience.
type LeabraNetwork interface {
	emer.Network

	// AsLeabra returns this network as a leabra.Network -- so that the
	// LeabraNetwork interface does not need to include accessors
	// to all the basic stuff
	AsLeabra() *Network

	// NewLayer creates a new concrete layer of appropriate type for this network
	NewLayer() emer.Layer

	// NewPrjn creates a new concrete projection of appropriate type for this network
	NewPrjn() emer.Prjn

	// AlphaCycInit handles all initialization at start of new input pattern.
	// Should already have presented the external input to the network at this point.
	// If updtActAvg is true, this includes updating the running-average
	// activations for each layer / pool, and the AvgL running average used
	// in BCM Hebbian learning.
	// The input scaling is updated  based on the layer-level running average acts,
	// and this can then change the behavior of the network,
	// so if you want 100% repeatable testing results, set this to false to
	// keep the existing scaling factors (e.g., can pass a train bool to
	// only update during training).  This flag also affects the AvgL learning
	// threshold
	AlphaCycInitImpl(updtActAvg bool)

	// CycleImpl runs one cycle of activation updating:
	// * Sends Ge increments from sending to receiving layers
	// * Average and Max Ge stats
	// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
	// * Activation from Ge, Gi, and Gl
	// * Average and Max Act stats
	// This basic version doesn't use the time info, but more specialized types do, and we
	// want to keep a consistent API for end-user code.
	CycleImpl(ltime *Time)

	// CyclePostImpl is called after the standard Cycle update, and calls CyclePost
	// on Layers -- this is reserved for any kind of special ad-hoc types that
	// need to do something special after Act is finally computed.
	// For example, sending a neuromodulatory signal such as dopamine.
	CyclePostImpl(ltime *Time)

	// QuarterFinalImpl does updating after end of a quarter
	QuarterFinalImpl(ltime *Time)

	// DWtImpl computes the weight change (learning) based on current
	// running-average activation values
	DWtImpl()

	// WtFmDWtImpl updates the weights from delta-weight changes.
	// Also calls WtBalFmWt every WtBalInterval times
	WtFmDWtImpl()
}

// LeabraLayer defines the essential algorithmic API for Leabra, at the layer level.
// These are the methods that the leabra.Network calls on its layers at each step
// of processing.  Other Layer types can selectively re-implement (override) these methods
// to modify the computation, while inheriting the basic behavior for non-overridden methods.
//
// All of the structural API is in emer.Layer, which this interface also inherits for
// convenience.
type LeabraLayer interface {
	emer.Layer

	// AsLeabra returns this layer as a leabra.Layer -- so that the LeabraLayer
	// interface does not need to include accessors to all the basic stuff
	AsLeabra() *Layer

	// SetThread sets the thread number for this layer to run on
	SetThread(thr int)

	// InitWts initializes the weight values in the network, i.e., resetting learning
	// Also calls InitActs
	InitWts()

	// InitActAvg initializes the running-average activation values that drive learning.
	InitActAvg()

	// InitActs fully initializes activation state -- only called automatically during InitWts
	InitActs()

	// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
	InitWtSym()

	// InitExt initializes external input state -- called prior to apply ext
	InitExt()

	// ApplyExt applies external input in the form of an etensor.Tensor
	// If the layer is a Target or Compare layer type, then it goes in Targ
	// otherwise it goes in Ext.
	ApplyExt(ext etensor.Tensor)

	// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
	// If the layer is a Target or Compare layer type, then it goes in Targ
	// otherwise it goes in Ext
	ApplyExt1D(ext []float64)

	// UpdateExtFlags updates the neuron flags for external input based on current
	// layer Type field -- call this if the Type has changed since the last
	// ApplyExt* method call.
	UpdateExtFlags()

	// RecvPrjns returns the slice of receiving projections for this layer
	RecvPrjns() *LeabraPrjns

	// SendPrjns returns the slice of sending projections for this layer
	SendPrjns() *LeabraPrjns

	// IsTarget returns true if this layer is a Target layer.
	// By default, returns true for layers of Type == emer.Target
	// Other Target layers include the TRCLayer in deep predictive learning.
	// This is used for turning off BCM hebbian learning,
	// in CosDiffFmActs to set the CosDiff.ModAvgLLrn value
	// for error-modulated level of hebbian learning.
	// It is also used in WtBal to not apply it to target layers.
	// In both cases, Target layers are purely error-driven.
	IsTarget() bool

	// AlphaCycInit handles all initialization at start of new input pattern.
	// Should already have presented the external input to the network at this point.
	// If updtActAvg is true, this includes updating the running-average
	// activations for each layer / pool, and the AvgL running average used
	// in BCM Hebbian learning.
	// The input scaling is updated  based on the layer-level running average acts,
	// and this can then change the behavior of the network,
	// so if you want 100% repeatable testing results, set this to false to
	// keep the existing scaling factors (e.g., can pass a train bool to
	// only update during training).  This flag also affects the AvgL learning
	// threshold
	AlphaCycInit(updtActAvg bool)

	// AvgLFmAvgM updates AvgL long-term running average activation that
	// drives BCM Hebbian learning
	AvgLFmAvgM()

	// GScaleFmAvgAct computes the scaling factor for synaptic conductance input
	// based on sending layer average activation.
	// This attempts to automatically adjust for overall differences in raw
	// activity coming into the units to achieve a general target
	// of around .5 to 1 for the integrated G values.
	GScaleFmAvgAct()

	// GenNoise generates random noise for all neurons
	GenNoise()

	// DecayState decays activation state by given proportion (default is on ly.Act.Init.Decay)
	DecayState(decay float32)

	// HardClamp hard-clamps the activations in the layer -- called during AlphaCycInit
	// for hard-clamped Input layers
	HardClamp()

	//////////////////////////////////////////////////////////////////////////////////////
	//  Cycle Methods

	// InitGInc initializes synaptic conductance increments -- optional
	InitGInc()

	// SendGDelta sends change in activation since last sent, to increment recv
	// synaptic conductances G, if above thresholds
	SendGDelta(ltime *Time)

	// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta
	GFmInc(ltime *Time)

	// AvgMaxGe computes the average and max Ge stats, used in inhibition
	AvgMaxGe(ltime *Time)

	// InhibiFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
	InhibFmGeAct(ltime *Time)

	// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
	// and updates learning running-average activations from that Act
	ActFmG(ltime *Time)

	// AvgMaxAct computes the average and max Act stats, used in inhibition
	AvgMaxAct(ltime *Time)

	// CyclePost is called after the standard Cycle update, as a separate
	// network layer loop.
	// This is reserved for any kind of special ad-hoc types that
	// need to do something special after Act is finally computed.
	// For example, sending a neuromodulatory signal such as dopamine.
	CyclePost(ltime *Time)

	//////////////////////////////////////////////////////////////////////////////////////
	//  Quarter Methods

	// QuarterFinal does updating after end of a quarter
	QuarterFinal(ltime *Time)

	// CosDiffFmActs computes the cosine difference in activation state
	// between minus and plus phases.
	// This is also used for modulating the amount of BCM hebbian learning
	CosDiffFmActs()

	// DWt computes the weight change (learning) -- calls DWt method on sending projections
	DWt()

	// WtFmDWt updates the weights from delta-weight changes -- on the sending projections
	WtFmDWt()

	// WtBalFmWt computes the Weight Balance factors based on average recv weights
	WtBalFmWt()

	// LrateMult sets the new Lrate parameter for Prjns to LrateInit * mult.
	// Useful for implementing learning rate schedules.
	LrateMult(mult float32)
}

// LeabraPrjn defines the essential algorithmic API for Leabra, at the projection level.
// These are the methods that the leabra.Layer calls on its prjns at each step
// of processing.  Other Prjn types can selectively re-implement (override) these methods
// to modify the computation, while inheriting the basic behavior for non-overridden methods.
//
// All of the structural API is in emer.Prjn, which this interface also inherits for
// convenience.
type LeabraPrjn interface {
	emer.Prjn

	// AsLeabra returns this prjn as a leabra.Prjn -- so that the LeabraPrjn
	// interface does not need to include accessors to all the basic stuff.
	AsLeabra() *Prjn

	// InitWts initializes weight values according to Learn.WtInit params
	InitWts()

	// InitWtSym initializes weight symmetry -- is given the reciprocal projection where
	// the Send and Recv layers are reversed.
	InitWtSym(rpj LeabraPrjn)

	// InitGInc initializes the per-projection synaptic conductance threadsafe increments.
	// This is not typically needed (called during InitWts only) but can be called when needed
	InitGInc()

	// SendGDelta sends the delta-activation from sending neuron index si,
	// to integrate synaptic conductances on receivers
	SendGDelta(si int, delta float32)

	// RecvGInc increments the receiver's synaptic conductances from those of all the projections.
	RecvGInc()

	// DWt computes the weight change (learning) -- on sending projections
	DWt()

	// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
	WtFmDWt()

	// WtBalFmWt computes the Weight Balance factors based on average recv weights
	WtBalFmWt()

	// LrateMult sets the new Lrate parameter for Prjns to LrateInit * mult.
	// Useful for implementing learning rate schedules.
	LrateMult(mult float32)
}

type LeabraPrjns []LeabraPrjn

func (pl *LeabraPrjns) Add(p LeabraPrjn) {
	(*pl) = append(*pl, p)
}
