// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"fmt"

	"github.com/emer/leabra/glong"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// Layer is the base layer type for PCore framework.
// Adds a dopamine variable to base Leabra layer type.
type Layer struct {
	glong.AlphaMaxLayer

	// dopamine value for this layer
	DA float32 `inactive:"+" desc:"dopamine value for this layer"`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, leabra.LayerProps)

// DALayer interface:

func (ly *Layer) GetDA() float32   { return ly.DA }
func (ly *Layer) SetDA(da float32) { ly.DA = da }

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.AlphaMaxLayer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	if varNm != "DA" {
		return -1, fmt.Errorf("pcore.Layer: variable named: %s not found", varNm)
	}
	nn := ly.AlphaMaxLayer.UnitVarNum()
	return nn, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
	nn := ly.AlphaMaxLayer.UnitVarNum()
	if varIdx < 0 || varIdx > nn { // nn = DA
		return mat32.NaN()
	}
	if varIdx < nn {
		return ly.AlphaMaxLayer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	return ly.DA
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return ly.AlphaMaxLayer.UnitVarNum() + 1
}

func (ly *Layer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}
