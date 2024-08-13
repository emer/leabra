// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/core/views"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/etable/v2/etensor"
)

// Layer implements the Leabra algorithm at the layer level,
// managing neurons and pathways.
type Layer struct {
	emer.LayerBase

	// our parent network, in case we need to use it to
	// find other layers etc; set when added by network
	Network *Network `copier:"-" json:"-" xml:"-" display:"-"`

	// type of layer.
	Type LayerType

	// list of receiving pathways into this layer from other layers
	RecvPaths []*Path

	// list of sending pathways from this layer to other layers
	SendPaths []*Path

	// Activation parameters and methods for computing activations
	Act ActParams `view:"add-fields"`

	// Inhibition parameters and methods for computing layer-level inhibition
	Inhib InhibParams `view:"add-fields"`

	// Learning parameters and methods that operate at the neuron level
	Learn LearnNeurParams `view:"add-fields"`

	// slice of neurons for this layer, as a flat list of len = Shape.Len().
	// Must iterate over index and use pointer to modify values.
	Neurons []Neuron

	// inhibition and other pooled, aggregate state variables.
	// flat list has at least of 1 for layer, and one for each sub-pool
	// if shape supports that (4D).
	// Must iterate over index and use pointer to modify values.
	Pools []Pool

	// cosine difference between ActM, ActP stats
	CosDiff CosDiffStats
}

// emer.Layer interface methods

func (ls *LayerBase) TypeName() string           { return ly.Type.String() }
func (ls *LayerBase) NumRecvPaths() int          { return len(ls.RecvPaths) }
func (ls *LayerBase) RecvPath(idx int) emer.Path { return ls.RecvPaths[idx] }
func (ls *LayerBase) NumSendPaths() int          { return len(ls.SendPaths) }
func (ls *LayerBase) SendPath(idx int) emer.Path { return ls.SendPaths[idx] }

// RecipToSendPath finds the reciprocal pathway relative to the given sending pathway
// found within the SendPaths of this layer.  This is then a recv path within this layer:
//
//	S=A -> R=B recip: R=A <- S=B -- ly = A -- we are the sender of srj and recv of rpj.
//
// returns false if not found.
func (ls *LayerBase) RecipToSendPath(spj emer.Path) (emer.Path, bool) {
	for _, rpj := range ls.RecvPaths {
		if rpj.SendLay() == spj.RecvLay() {
			return rpj, true
		}
	}
	return nil, false
}

// ApplyParams applies given parameter style Sheet to this layer and its recv pathways.
// Calls UpdateParams on anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ls *LayerBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	app, err := pars.Apply(ls.LeabraLay, setMsg) // essential to go through LeabraPrj
	if app {
		ls.LeabraLay.UpdateParams()
		applied = true
	}
	if err != nil {
		rerr = err
	}
	for _, pj := range ls.RecvPaths {
		app, err = pj.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
}
