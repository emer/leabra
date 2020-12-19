// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"strings"
)

func TotalAct(ly emer.Layer) float32 {
	lly := ly.(leabra.LeabraLayer).AsLeabra()
	pl := lly.Pools[0].Inhib.Act
	res := pl.Avg * float32(pl.N)
	if math32.IsNaN(res) {
		fmt.Println("NaN in TotalAct")
	}
	return res
}

type Network struct {
	leabra.Network
}

var NetworkProps = leabra.NetworkProps
var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

type INetwork interface {
	AsLeabra() *leabra.Network
}

func (nt *Network) AsLeabra() *leabra.Network {
	return &nt.Network
}

func (nt *Network) InitActs() {
	for li := range nt.Layers {
		ly := nt.Layers[li]
		ly.(leabra.LeabraLayer).InitActs()
	}
}

func (nt *Network) Cycle(ltime *leabra.Time) {
	nt.CycleImpl(ltime)
	nt.EmerNet.(leabra.LeabraNetwork).CyclePostImpl(ltime) // always call this after std cycle..
}

//
func (nt *Network) CycleImpl(ltime *leabra.Time) {
	nt.QuarterInitPrvs(ltime)
	nt.SendGDelta(ltime) // also does integ
	nt.SendMods(ltime)
	nt.RecvModInc(ltime)
	nt.AvgMaxGe(ltime)
	nt.AvgMaxMod(ltime)
	nt.InhibFmGeAct(ltime)
	nt.ActFmG(ltime)
	nt.AvgMaxAct(ltime)
}

func (nt *Network) QuarterInitPrvs(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(*MSNLayer); ok {
			pl.QuarterInitPrvs(ltime)
		}
	}, "QuarterInitPrvs")
}

func (nt *Network) SendMods(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(ModSender); ok {
			pl.SendMods(ltime)
		}
	}, "SendMods")
}

func (nt *Network) RecvModInc(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(ModReceiver); ok {
			pl.ModsFmInc(ltime)
		}
	}, "RecvModInc")
}

func (nt *Network) ClearModActs(_ *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if ml, ok := ly.(IModLayer); ok {
			ml.AsMod().ClearModActs()
		}
	}, "ClearModActs")
}

func (nt *Network) ClearMSNTraces(_ *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if msnly, ok := ly.(IMSNLayer); ok {
			msnly.AsMSNLayer().ClearMSNTrace()
		}
	}, "ClearMSNTraces")
}

func (nt *Network) AvgMaxMod(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(AvgMaxModLayer); ok {
			pl.AvgMaxMod(ltime)
		}
	}, "AvgMaxMod")
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return ModNeuronVarsAll
}

func (nt *Network) SynVarNames() []string {
	return SynapseVarsAll
}

// SynVarProps returns properties for variables
func (nt *Network) SynVarProps() map[string]string {
	return SynapseVarProps
}

// For special layer types

// AddVTALayer adds a positive or negative Valence VTA layer
func (nt *Network) AddVTALayer(name string, val Valence) *VTALayer {
	return AddVTALayer(nt, name, val)
}

// AddMatrixLayer adds a MSNLayer of given size, with given name.
//
// Geometry is 4D.
//
// nY = number of pools in Y dimension, nX is pools in X dimension,
// and each pool has nNeurY * nNeurX neurons.
//
// cpmt specifies patch or matrix StriatalCompartment
//
//da parameter gives the DaReceptor type (DaRType) (D1R = Go, D2R = NoGo)
func (nt *Network) AddMSNLayer(name string, nY, nX, nNeurY, nNeurX int, cpmt StriatalCompartment, da DaRType) *MSNLayer {
	return AddMSNLayer(nt, name, nY, nX, nNeurY, nNeurX, cpmt, da)
}

// AddCElAmygLayer adds a CentroLateral Amygdala layer with specified 4D geometry, acquisition/extinction, valence, and DA receptor type
//
// Geometry is 4D.
//
// nY = number of pools in Y dimension, nX is pools in X dimension,
// and each pool has nNeurY * nNeurX neurons.  da parameter gives the DaReceptor type (D1R = Go, D2R = NoGo).
// acqExt (AcqExt) specifies whether this layer is involved with acquisition or extinction.
// val is positive (appetitive) or negative (aversive) Valence.
func (nt *Network) AddCElAmygLayer(name string, nY, nX, nNeurY, nNeurX int,
	acqExt AcqExt, val Valence, dar DaRType) *CElAmygLayer {
	ly := CElAmygLayer{CElTyp: CElAmygLayerType{AcqExt: acqExt, Valence: val}}
	ly.DaMod.RecepType = dar
	nt.AddLayerInit(&ly, name, []int{nY, nX, nNeurY, nNeurX}, emer.Hidden)
	ly.ModLayer.Init()
	class := "CEl" + acqExt.String() + strings.Title(strings.ToLower(val.String())) + dar.String()[0:2] + " CElAmyg"
	ly.SetClass(class)
	return &ly
}

// AddBlAmygLayer adds a Basolateral Amygdala layer with specified 4D geometry, acquisition/extinction, valence, and DA receptor type
func (nt *Network) AddBlAmygLayer(name string, nY, nX, nNeurY, nNeurX int, val Valence, dar DaRType, lTyp emer.LayerType) *BlAmygLayer {
	ly := BlAmygLayer{Valence: val}
	ly.DaMod.RecepType = dar
	nt.AddLayerInit(&ly, name, []int{nY, nX, nNeurY, nNeurX}, lTyp)
	ly.ModLayer.Init()
	class := "BlAmyg" + strings.Title(strings.ToLower(val.String())) + dar.String()[0:2] + " BlAmyg"
	ly.SetClass(class)
	return &ly
}

func (nt *Network) ConnectLayersActMod(sender ModSender, rcvr ModReceiver, scale float32) {
	sender.(IModLayer).AsMod().AddModReceiver(rcvr, scale)
}
