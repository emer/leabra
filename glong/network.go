// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/leabra/v2/leabra"
	"github.com/goki/ki/kit"
)

// glong.Network has methods for configuring specialized Glong network components.
type Network struct {
	leabra.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = leabra.NetworkProps

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.Network.Defaults()
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	nt.Network.UpdateParams()
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitVarProps returns properties for variables
func (nt *Network) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// NewLayer returns new layer of glong.Layer type -- this is default type for this network
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// ConnectNMDA adds a NMDAPrjn between given layers
func (nt *Network) ConnectNMDA(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return ConnectNMDA(&nt.Network, send, recv, pat)
}

////////////////////////////////////////////////////////////////////////
// Network functions available here as standalone functions
//         for mixing in to other models

// AddGlongLayer2D adds a glong.Layer using 2D shape
func AddGlongLayer2D(nt *leabra.Network, name string, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddGlongLayer4D adds a glong.Layer using 4D shape with pools
func AddGlongLayer4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// ConnectNMDA adds a NMDAPrjn between given layers
func ConnectNMDA(nt *leabra.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, NMDA, &NMDAPrjn{})
}
