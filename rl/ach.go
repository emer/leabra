// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

// AChLayer is an interface for a layer with acetylcholine neuromodulator on it
type AChLayer interface {
	// GetACh returns the acetylcholine level for layer
	GetACh() float32

	// SetACh sets the acetylcholine level for layer
	SetACh(ach float32)
}

// SendACh is a list of layers to send acetylcholine to
type SendACh emer.LayNames

// SendACh sends acetylcholine to list of layers
func (sd *SendACh) SendACh(net emer.Network, ach float32) {
	for _, lnm := range *sd {
		ml, ok := net.LayerByName(lnm).(AChLayer)
		if ok {
			ml.SetACh(ach)
		}
	}
}

// Validate ensures that LayNames layers are valid.
// ctxt is string for error message to provide context.
func (sd *SendACh) Validate(net emer.Network, ctxt string) error {
	ln := (*emer.LayNames)(sd)
	return ln.Validate(net, ctxt)
}

// Add adds given layer name(s) to list
func (sd *SendACh) Add(laynm ...string) {
	*sd = append(*sd, laynm...)
}

// AddOne adds one layer name to list -- python version -- doesn't support varargs
func (sd *SendACh) AddOne(laynm string) {
	*sd = append(*sd, laynm)
}

// AddAllBut adds all layers in network except those in exlude list
func (sd *SendACh) AddAllBut(net emer.Network, excl ...string) {
	ln := (*emer.LayNames)(sd)
	ln.AddAllBut(net, excl...)
}

// Layers that use SendACh should include a Validate check in Build as follows:

// Build constructs the layer state, including calling Build on the pathways.
// func (ly *AChSrcLayer) Build() error {
// 	err := ly.Layer.Build()
// 	if err != nil {
// 		return err
// 	}
// 	err = ly.SendACh.Validate(ly.Network, ly.Name()+" SendTo list")
// 	return err
// }

//////////////////////////////////////////////////////////////////////////////////////
//  ClampAChLayer

// ClampAChLayer is an Input layer that just sends its activity as the acetylcholine signal
type ClampAChLayer struct {
	leabra.Layer

	// list of layers to send acetylcholine to
	SendACh SendACh

	// acetylcholine value for this layer
	ACh float32
}

// AChLayer interface:

func (ly *ClampAChLayer) GetACh() float32    { return ly.ACh }
func (ly *ClampAChLayer) SetACh(ach float32) { ly.ACh = ach }

// Build constructs the layer state, including calling Build on the pathways.
func (ly *ClampAChLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendACh.Validate(ly.Network, ly.Name()+" SendTo list")
	return err
}

// CyclePost is called at end of Cycle
// We use it to send ACh, which will then be active for the next cycle of processing.
func (ly *ClampAChLayer) CyclePost(ltime *leabra.Time) {
	act := ly.Neurons[0].Act
	ly.ACh = act
	ly.SendACh.SendACh(ly.Network, act)
}
