// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"strconv"

	"github.com/emer/leabra/v2/interinhib"
	"github.com/emer/leabra/v2/leabra"
)

// IBlAmygLayer has one method, AsBlAmygLayer, that returns a pointer to the layer specifically as a BLA layer.
type IBlAmygLayer interface {
	AsBlAmygLayer() *BlAmygLayer
}

// AsBlAmygLayer returns a pointer to the layer specifically as a BLA layer.
func (ly *BlAmygLayer) AsBlAmygLayer() *BlAmygLayer {
	return ly
}

// BlAmygLayer contains values specific to BLA layers, including Interlayer Inhibition (ILI)
type BlAmygLayer struct {

	// modulation state
	ModLayer

	// positive or negative valence
	Valence Valence

	// inter-layer inhibition parameters and state
	ILI interinhib.InterInhib
}

func (ly *BlAmygLayer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.Neurons = make([]leabra.Neuron, nu)
	err := ly.AsMod().Build()
	if err != nil {
		return err
	}
	err = ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	if err != nil {
		return err
	}
	return err
}

func (ly *BlAmygLayer) Defaults() {
	ly.ModLayer.Defaults()
	ly.ActModZero = true
	ly.DaMod.BurstGain = 0.04
	ly.DaMod.DipGain = 0.1
	ly.DaMod.On = true
	ly.Minus = 1.0
	ly.Plus = 1.0
	ly.NegGain = 0.1
	ly.PosGain = 0.1

	ly.Act.Init.Vm = 0.55
}

// GetMonitorVal retrieves a value for a trace of some quantity, possibly more than just a variable
func (ly *BlAmygLayer) GetMonitorVal(data []string) float64 {
	var val float32
	var err error
	valType := data[0]
	unitIdx, _ := strconv.Atoi(data[1])
	switch valType {
	case "TotalAct":
		val = TotalAct(ly)
	case "PoolActAvg":
		val = ly.Pools[unitIdx].Inhib.Act.Avg
	case "PoolActMax":
		val = ly.Pools[unitIdx].Inhib.Act.Max
	case "Act":
		val = ly.Neurons[unitIdx].Act
	case "ActDiff":
		val = ly.Neurons[unitIdx].Act - ly.ModNeurs[unitIdx].ModAct
	default:
		mnr := &ly.ModNeurs[unitIdx]
		val, err = mnr.VarByName(valType)
		if err != nil {
			nrn := &ly.Neurons[unitIdx]
			val, err = nrn.VarByName(valType)
			if err != nil {
				fmt.Printf("VarByName error: %v\n", err)
			}
		}
	}
	return float64(val)
}

// InhibiFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *BlAmygLayer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.ILI.Inhib(&ly.Layer) // does inter-layer inhibition
	ly.PoolInhibFmGeAct(ltime)
	ly.InhibFmPool(ltime)
}
