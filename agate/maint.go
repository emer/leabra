// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// NMDAParams control the NMDA dynamics in PFC Maint neurons, based on Brunel & Wang (2001)
// parameters.  We have to do some things to make it work for rate code neurons..
type NMDAParams struct {
	Tau   float32 `def:"100" desc:"decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential"`
	Gbar  float32 `def:"1.7" desc:"strength of NMDA current -- 1.7 is just over level sufficient to maintain in face of completely blank input"`
	ActVm float32 `def:"0.4" desc:"extra contribution to Vm associated with action potentials, on average -- produces key nonlinearity associated with spiking, from backpropagating action potentials.  0.4 seems good.."`
}

func (np *NMDAParams) Defaults() {
	np.Tau = 100
	np.Gbar = 1.7
	np.ActVm = 0.4
}

// GFmV returns the NMDA conductance as a function of normalized membrane potential
func (np *NMDAParams) GFmV(v float32) float32 {
	vbio := v*100 - 100
	return 1 / (1 + 0.28*math32.Exp(-0.062*vbio))
}

///////////////////////////////////////////////////////////////////////////
// MaintLayer

// MaintLayer is a layer with NMDA channels that supports active maintenance
// in frontal cortex, via layer 3 recurrent circuits.
type MaintLayer struct {
	leabra.Layer
	NMDA       NMDAParams    `view:"inline" desc:"parameters for NMDA channel conductances that sustain active maintenance"`
	MaintNeurs []MaintNeuron `desc:"slice of extra MaintNeuron state for this layer -- flat list of len = Shape.Len(). You must iterate over index and use pointer to modify values."`
}

var KiT_MaintLayer = kit.Types.AddType(&MaintLayer{}, leabra.LayerProps)

func (ly *MaintLayer) Defaults() {
	ly.Layer.Defaults()
	ly.NMDA.Defaults()
	ly.Act.Init.Decay = 0
}

func (ly *MaintLayer) InitNMDA() {
	for ni := range ly.MaintNeurs {
		nrn := &ly.MaintNeurs[ni]
		nrn.Grec = 0
		nrn.GrecInc = 0
		nrn.Gnmda = 0
		nrn.VmEff = 0
	}
}

func (ly *MaintLayer) InitGInc() {
	ly.Layer.InitGInc()
	for ni := range ly.MaintNeurs {
		nrn := &ly.MaintNeurs[ni]
		nrn.Grec = 0
	}
}

func (ly *MaintLayer) InitActs() {
	ly.Layer.InitActs()
	ly.InitNMDA()
}

func (ly *MaintLayer) DecayState(decay float32) {
	ly.Layer.DecayState(decay)
	for ni := range ly.MaintNeurs {
		mnr := &ly.MaintNeurs[ni]
		mnr.Grec -= decay * mnr.Grec
		mnr.GrecInc -= decay * mnr.GrecInc
		mnr.Gnmda -= decay * mnr.Gnmda
		mnr.VmEff -= decay * mnr.VmEff
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *MaintLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	ly.RecvGrecInc(ltime)
	ly.GFmIncNeur(ltime)
}

// RecvGInc calls RecvGInc on receiving projections to collect Neuron-level G*Inc values.
// This is called by GFmInc overall method, but separated out for cases that need to
// do something different.
func (ly *MaintLayer) RecvGInc(ltime *leabra.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		if p.Type() == emer.Lateral { // skip recurrent
			continue
		}
		p.(leabra.LeabraPrjn).RecvGInc()
	}
}

// RecvGrecInc increments the recurrent-specific GeInc
func (ly *MaintLayer) RecvGrecInc(ltime *leabra.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		if p.Type() != emer.Lateral { // skip recurrent
			continue
		}
		pj := p.(leabra.LeabraPrjn).AsLeabra()
		for ri := range ly.MaintNeurs {
			rn := &ly.MaintNeurs[ri]
			rn.GrecInc += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	}
}

// GFmIncNeur is the neuron-level code for GFmInc that integrates G*Inc into G*Raw
// and finally overall Ge, Gi values
func (ly *MaintLayer) GFmIncNeur(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		// note: each step broken out here so other variants can add extra terms to Raw
		ly.Act.GRawFmInc(nrn)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)

		mnr := &ly.MaintNeurs[ni]
		mnr.Grec += mnr.GrecInc
		mnr.GrecInc = 0
		mnr.VmEff = nrn.Vm + ly.NMDA.ActVm*nrn.Act

		mnr.Gnmda = ly.NMDA.Gbar*mnr.Grec*ly.NMDA.GFmV(mnr.VmEff) - (mnr.Gnmda / ly.NMDA.Tau)

		ly.Act.GeFmRaw(nrn, nrn.GeRaw+mnr.Gnmda)
	}
}

///////////////////////////////////////////////////////////////////////////
// Neurons

// Build constructs the layer state, including calling Build on the projections.
func (ly *MaintLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.MaintNeurs = make([]MaintNeuron, len(ly.Neurons))
	return nil
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *MaintLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = MaintNeuronVarByName(varNm)
	if err != nil {
		return -1, err
	}
	nn := len(leabra.NeuronVars)
	return nn + vidx, err
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *MaintLayer) UnitVal1D(varIdx int, idx int) float32 {
	if varIdx < 0 {
		return math32.NaN()
	}
	nn := len(leabra.NeuronVars)
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	varIdx -= nn
	if varIdx >= len(MaintNeuronVars) {
		return math32.NaN()
	}
	mnr := &ly.MaintNeurs[idx]
	return mnr.VarByIndex(varIdx)
}