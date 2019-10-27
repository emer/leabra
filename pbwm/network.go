// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/deep"
	"github.com/goki/ki/kit"
)

// pbwm.Network has parameters for running a DeepLeabra network
type Network struct {
	deep.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = deep.NetworkProps

// NewLayer returns new layer of default ModLayer type
func (nt *Network) NewLayer() emer.Layer {
	return &ModLayer{}
}

// NewPrjn returns new prjn of default type
func (nt *Network) NewPrjn() emer.Prjn {
	return &deep.Prjn{}
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.Network.Defaults()
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	nt.Network.UpdateParams()
}

// todo: layer creation types

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods
