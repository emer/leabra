// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"fmt"

	"cogentcore.org/core/mat32"
	"github.com/emer/leabra/v2/leabra"
	"github.com/goki/ki/kit"
)

// AlphaMaxLayer computes the maximum activation per neuron over the alpha cycle.
// Needed for recording activations on layers with transient dynamics over alpha.
type AlphaMaxLayer struct {
	leabra.Layer

	// cycle upon which to start updating AlphaMax value
	AlphaMaxCyc int

	// per-neuron maximum activation value during alpha cycle
	AlphaMaxs []float32
}

var KiT_AlphaMaxLayer = kit.Types.AddType(&AlphaMaxLayer{}, leabra.LayerProps)

func (ly *AlphaMaxLayer) Defaults() {
	ly.Layer.Defaults()
	ly.AlphaMaxCyc = 30
}

func (ly *AlphaMaxLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	nn := len(ly.Neurons)
	ly.AlphaMaxs = make([]float32, nn)
	return nil
}

// InitAlphaMax initializes the AlphaMax to 0
func (ly *AlphaMaxLayer) InitAlphaMax() {
	for pi := range ly.AlphaMaxs {
		ly.AlphaMaxs[pi] = 0
	}
}

func (ly *AlphaMaxLayer) InitActs() {
	ly.Layer.InitActs()
	ly.InitAlphaMax()
}

func (ly *AlphaMaxLayer) AlphaCycInit(updtActAvg bool) {
	ly.Layer.AlphaCycInit(updtActAvg)
	ly.InitAlphaMax()
}

func (ly *AlphaMaxLayer) ActFmG(ltime *leabra.Time) {
	ly.Layer.ActFmG(ltime)
	if ltime.Cycle >= ly.AlphaMaxCyc {
		ly.AlphaMaxFmAct(ltime)
	}
}

// AlphaMaxFmAct computes AlphaMax from Activation
func (ly *AlphaMaxLayer) AlphaMaxFmAct(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		max := &ly.AlphaMaxs[ni]
		*max = mat32.Max(*max, nrn.Act)
	}
}

// ActLrnFmAlphaMax sets ActLrn to AlphaMax
func (ly *AlphaMaxLayer) ActLrnFmAlphaMax() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActLrn = ly.AlphaMaxs[ni]
	}
}

// MaxAlphaMax returns the maximum AlphaMax across the layer
func (ly *AlphaMaxLayer) MaxAlphaMax() float32 {
	mx := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		amx := ly.AlphaMaxs[ni]
		mx = mat32.Max(amx, mx)
	}
	return mx
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *AlphaMaxLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	if varNm != "AlphaMax" {
		return -1, fmt.Errorf("glong.AlphaMaxLayers: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *AlphaMaxLayer) UnitVal1D(varIdx int, idx int) float32 {
	nn := ly.Layer.UnitVarNum()
	if varIdx < 0 || varIdx > nn { // nn = AlphaMax
		return mat32.NaN()
	}
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	return ly.AlphaMaxs[idx]
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *AlphaMaxLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 1
}
