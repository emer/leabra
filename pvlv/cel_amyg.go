// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"

	_ "github.com/emer/etable/etensor"
	"github.com/emer/leabra/v2/leabra"
	"github.com/goki/ki/kit"
)

type ICElAmygLayer interface {
	AsCElAmygLayer() *CElAmygLayer
}

func (ly *CElAmygLayer) AsCElAmygLayer() *CElAmygLayer {
	return ly
}

type AcqExt int

const (
	Acq AcqExt = iota
	Ext
	NAcqExt
)

var KiT_AcqExt = kit.Enums.AddEnum(NAcqExt, kit.NotBitFlag, nil)

type CElAmygLayer struct {
	ModLayer

	// basic parameters determining what type CEl layer this is
	CElTyp CElAmygLayerType `desc:"basic parameters determining what type CEl layer this is"`

	// use deep_mod_net for value from acquisition / go units, instead of inhibition current (otherwise use gi_syn) -- allows simpler parameter setting without titrating inhibition and this learning modulation signal
	AcqDeepMod bool `desc:"use deep_mod_net for value from acquisition / go units, instead of inhibition current (otherwise use gi_syn) -- allows simpler parameter setting without titrating inhibition and this learning modulation signal"`
}

var KiT_CElAmygLayer = kit.Types.AddType(&CElAmygLayer{}, nil)

type CElAmygLayerType struct {

	// acquisition or extinction
	AcqExt AcqExt `desc:"acquisition or extinction"`

	// positive or negative DA valence
	Valence Valence `desc:"positive or negative DA valence"`
}

func (ly *CElAmygLayer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.Neurons = make([]leabra.Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	if err != nil {
		return err
	}
	err = ly.AsMod().Build()
	if err != nil {
		return err
	}
	return err
}

func (ly *CElAmygLayer) Defaults() {
	ly.ModLayer.Defaults()
	ly.Act.Init.Vm = 0.55
	ly.ActModZero = false
	ly.AcqDeepMod = true
	ly.DaMod.BurstGain = 0.04 // differs between CEl layers
	ly.DaMod.DipGain = 0.1    // differs between CEl layers
}
