// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

///////////////////////////////////////////////////////////////////////////
// Layer

// Layer has GABA-B and NMDA channels, with longer time-constants,
// to supports bistable activation dynamics including active maintenance
// in frontal cortex.  NMDA requires NMDAPrjn on relevant projections.
// It also records AlphaMax = maximum activation within an AlphaCycle,
// which is important given the transient dynamics.
type Layer struct {
	leabra.Layer
	NMDA    NMDAParams  `view:"inline" desc:"NMDA channel parameters plus more general params"`
	GABAB   GABABParams `view:"inline" desc:"GABA-B / GIRK channel parameters"`
	GlNeurs []Neuron    `desc:"slice of extra glong.Neuron state for this layer -- flat list of len = Shape.Len(). You must iterate over index and use pointer to modify values."`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, leabra.LayerProps)

func (ly *Layer) Defaults() {
	ly.Layer.Defaults()
	ly.NMDA.Defaults()
	ly.GABAB.Defaults()
	ly.Act.Init.Decay = 0
	ly.Act.Erev.K = .1 // -90mv
}

func (ly *Layer) InitGlong() {
	for ni := range ly.GlNeurs {
		nrn := &ly.GlNeurs[ni]
		nrn.AlphaMax = 0
		nrn.VmEff = 0
		nrn.GnmdaP = 0
		nrn.GnmdaPInc = 0
		nrn.Gnmda = 0
		nrn.GgabaB = 0
	}
}

// InitAlphaMax initializes the AlphaMax to 0
func (ly *Layer) InitAlphaMax() {
	for ni := range ly.GlNeurs {
		nrn := &ly.GlNeurs[ni]
		nrn.AlphaMax = 0
	}
}

func (ly *Layer) InitGInc() {
	ly.Layer.InitGInc()
	for ni := range ly.GlNeurs {
		nrn := &ly.GlNeurs[ni]
		nrn.GnmdaP = 0
	}
}

func (ly *Layer) InitActs() {
	ly.Layer.InitActs()
	ly.InitGlong()
}

func (ly *Layer) DecayState(decay float32) {
	ly.Layer.DecayState(decay)
	for ni := range ly.GlNeurs {
		gnr := &ly.GlNeurs[ni]
		gnr.VmEff -= decay * gnr.VmEff
		gnr.GnmdaP -= decay * gnr.GnmdaP
		gnr.GnmdaPInc -= decay * gnr.GnmdaPInc
		gnr.Gnmda -= decay * gnr.Gnmda
		gnr.GgabaB -= decay * gnr.GgabaB
	}
}

func (ly *Layer) AlphaCycInit() {
	ly.Layer.AlphaCycInit()
	ly.InitAlphaMax()
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *Layer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	ly.RecvGnmdaPInc(ltime)
	ly.GFmIncNeur(ltime)
}

// RecvGInc calls RecvGInc on receiving projections to collect Neuron-level G*Inc values.
// This is called by GFmInc overall method, but separated out for cases that need to
// do something different.
func (ly *Layer) RecvGInc(ltime *leabra.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		if p.Type() == NMDA { // skip NMDA
			continue
		}
		p.(leabra.LeabraPrjn).RecvGInc()
	}
}

// RecvGnmdaPInc increments the recurrent-specific GeInc
func (ly *Layer) RecvGnmdaPInc(ltime *leabra.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		if p.Type() != NMDA { // skip non-NMDA
			continue
		}
		pj := p.(leabra.LeabraPrjn).AsLeabra()
		for ri := range ly.GlNeurs {
			rn := &ly.GlNeurs[ri]
			rn.GnmdaPInc += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	}
}

// GFmIncNeur is the neuron-level code for GFmInc that integrates G*Inc into G*Raw
// and finally overall Ge, Gi values
func (ly *Layer) GFmIncNeur(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		// note: each step broken out here so other variants can add extra terms to Raw
		ly.Act.GRawFmInc(nrn)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)

		gnr := &ly.GlNeurs[ni]
		gnr.VmEff = ly.NMDA.VmEff(nrn.Vm, nrn.Act)

		gnr.GnmdaP += gnr.GnmdaPInc
		gnr.GnmdaPInc = 0

		gnr.Gnmda = ly.NMDA.Gnmda(gnr.GnmdaP, gnr.Gnmda, gnr.VmEff)

		ly.Act.GeFmRaw(nrn, nrn.GeRaw+gnr.Gnmda)
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime)
}

// PoolInhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) PoolInhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	np := len(ly.Pools)
	if np > 1 {
		for pi := 1; pi < np; pi++ {
			pl := &ly.Pools[pi]
			ly.Inhib.Pool.Inhib(&pl.Inhib)
			pl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi)
			if !ly.Inhib.Layer.On { // keep layer level updated for inter-layer inhib
				lpl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi)
			}
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
				nrn.Gi = pl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
				// above is standard, below is GabaB
				gnr := &ly.GlNeurs[ni]
				gnr.GgabaB, gnr.GgabaBD = ly.GABAB.GgabaB(gnr.GgabaB, gnr.GgabaBD, nrn.Gi, gnr.VmEff)
				nrn.Gk = gnr.GgabaB + ly.GABAB.Gbar*ly.GABAB.Gbase
			}
		}
	} else {
		for ni := lpl.StIdx; ni < lpl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
			nrn.Gi = lpl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
			// above is standard, below is GabaB
			gnr := &ly.GlNeurs[ni]
			gnr.GgabaB, gnr.GgabaBD = ly.GABAB.GgabaB(gnr.GgabaB, gnr.GgabaBD, nrn.Gi, gnr.VmEff)
			nrn.Gk = gnr.GgabaB + ly.GABAB.Gbar*ly.GABAB.Gbase
		}
	}
}

func (ly *Layer) ActFmG(ltime *leabra.Time) {
	ly.Layer.ActFmG(ltime)
	if ltime.Cycle >= ly.NMDA.AlphaMaxCyc {
		ly.AlphaMaxFmAct(ltime)
	}
}

// AlphaMaxFmAct computes AlphaMax from Activation
func (ly *Layer) AlphaMaxFmAct(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		gnr := &ly.GlNeurs[ni]
		gnr.AlphaMax = math32.Max(gnr.AlphaMax, nrn.Act)
	}
}

// ActLrnFmAlphaMax sets ActLrn to AlphaMax
func (ly *Layer) ActLrnFmAlphaMax() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		gnr := &ly.GlNeurs[ni]
		nrn.ActLrn = gnr.AlphaMax
	}
}

// MaxAlphaMax returns the maximum AlphaMax across the layer
func (ly *Layer) MaxAlphaMax() float32 {
	mx := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		gnr := &ly.GlNeurs[ni]
		mx = math32.Max(gnr.AlphaMax, mx)
	}
	return mx
}

///////////////////////////////////////////////////////////////////////////
// Neurons

// Build constructs the layer state, including calling Build on the projections.
func (ly *Layer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.GlNeurs = make([]Neuron, len(ly.Neurons))
	return nil
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = NeuronVarByName(varNm)
	if err != nil {
		return -1, err
	}
	nn := ly.Layer.UnitVarNum()
	return nn + vidx, err
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
	if varIdx < 0 {
		return math32.NaN()
	}
	nn := ly.Layer.UnitVarNum()
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	varIdx -= nn
	if varIdx > len(NeuronVars) {
		return math32.NaN()
	}
	gnr := &ly.GlNeurs[idx]
	return gnr.VarByIndex(varIdx)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + len(NeuronVars)
}
