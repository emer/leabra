// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attrn

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
)

////////////////////////////////////////////////////////////////////////
// Network functions available here as standalone functions
//         for mixing in to other models

// AddAttnLayer adds an attrn.AttnLayer, which must be a 4D, with pools (pool-level attention).
func AddAttnLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeursY, nNeursX int) *AttnLayer {
	ly := &AttnLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeursY, nNeursX}, emer.Hidden)
	return ly
}

// AddTRNLayer adds an attrn.TRNLayer -- the nPoolss correspond to pool dimensions
// of modulated layer.
func AddTRNLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX int) *TRNLayer {
	ly := &TRNLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX}, emer.Hidden)
	return ly
}

// AddAttnTRNLayer adds AttnLayer and corresponding TRNLayer, with "T" suffix.
// Uses the pool-level dimensions to set 2D dimensions for TRN layer,
// and connects the layers with default parameters.
// The TRN layer will automatically look for "CT" and "P" layer names
// and receive EPool inputs from those.
func AddAttnTRNLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeursY, nNeursX int) (*AttnLayer, *TRNLayer) {
	attn := AddAttnLayer(nt, name, nPoolsY, nPoolsX, nNeursY, nNeursX)
	trn := AddTRNLayer(nt, name+"T", nPoolsY, nPoolsX)
	_, err := nt.LayerByNameTry(name + "CT")
	if err == nil {
		trn.EPools.Add(name+"CT", 1)
	}
	_, err = nt.LayerByNameTry(name + "P")
	if err == nil {
		trn.EPools.Add(name+"P", .2)
	}
	trn.SendTo.Add(name)
	return attn, trn
}
