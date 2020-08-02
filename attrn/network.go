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

// AddAttnLayer adds an attrn.AttnLayer -- the nPools correspond to pool dimensions
// of modulated layer.
func AddAttnLayer(nt *leabra.Network, name string, nPoolY, nPoolX int) *AttnLayer {
	ly := &AttnLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolY, nPoolX}, emer.Hidden)
	return ly
}

// AddTRNLayer adds an attrn.TRNLayer -- the nPools correspond to pool dimensions
// of modulated layer.
func AddTRNLayer(nt *leabra.Network, name string, nPoolY, nPoolX int) *TRNLayer {
	ly := &TRNLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolY, nPoolX}, emer.Hidden)
	return ly
}

// AddAttnForLayer adds AttnLayer and TRNLayer attention layers
// for given source layer to be modulated (typically a deep.Super layer).
// Source Layer must have a 4D shape -- uses the pool-level dimensions
// to set 2D dimensions for Attn layers, and connects the layers with
// default parameters.
// The TRN layer will automatically look for "CT" and "P" layer names
// and receive EPool inputs from those.
func AddAttnForLayer(nt *leabra.Network, srcLay *leabra.Layer) (*AttnLayer, *TRNLayer) {
	nPoolY := srcLay.Shp.Dim(0)
	nPoolX := srcLay.Shp.Dim(1)
	attn := AddAttnLayer(nt, srcLay.Nm+"A", nPoolY, nPoolX)
	trn := AddTRNLayer(nt, srcLay.Nm+"T", nPoolY, nPoolX)
	_, err := nt.LayerByNameTry(srcLay.Nm + "CT")
	if err == nil {
		trn.EPools.Add(srcLay.Nm+"CT", 1)
	}
	_, err = nt.LayerByNameTry(srcLay.Nm + "P")
	if err == nil {
		trn.EPools.Add(srcLay.Nm+"P", .2)
	}
	trn.SendTo.Add(srcLay.Nm)
	return attn, trn
}
