// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"github.com/emer/leabra/interinhib"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"strconv"
)

type IBlAmygLayer interface {
	AsBlAmygLayer() *BlAmygLayer
}

func (ly *BlAmygLayer) AsBlAmygLayer() *BlAmygLayer {
	return ly
}

type BlAmygLayer struct {
	ModLayer `desc:"modulation state"`
	Valence  Valence               `desc:"positive or negative valence"`
	ILI      interinhib.InterInhib `desc:"inter-layer inhibition parameters and state"`
}

var KiT_BlAmygLayer = kit.Types.AddType(&BlAmygLayer{}, nil)

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
