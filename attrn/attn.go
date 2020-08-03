// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attrn

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// AttnParams map Attn values onto their respective effects on Ge, Act, ActLrn
type AttnParams struct {
	GeMin  float32 `desc:"minimum value (for Attn = 0) for Attn multiplicative effect on Ge -- set to 1 for no modulation"`
	ActMin float32 `desc:"minimum value (for Attn = 0) for Attn multiplicative effect on Act -- set to 1 for no modulation"`
	LrnMin float32 `desc:"minimum value (for Attn = 0) for Attn multiplicative effect on ActLrn -- set to 1 for no modulation"`

	GeRng  float32 `view:"-" desc:"range = 1-min"`
	ActRng float32 `view:"-" desc:"range = 1-min"`
	LrnRng float32 `view:"-" desc:"range = 1-min"`
}

func (ap *AttnParams) Defaults() {
	ap.GeMin = 0.5
	ap.ActMin = 1
	ap.LrnMin = 0
}

func (ap *AttnParams) Update() {
	ap.GeRng = 1.0 - ap.GeMin
	ap.ActRng = 1.0 - ap.ActMin
	ap.LrnRng = 1.0 - ap.LrnMin
}

// GeMod returns attention-modulated value of Ge
func (ap *AttnParams) GeMod(ge, attn float32) float32 {
	return ge * (ap.GeMin + ap.GeRng*attn)
}

// ActMod returns attention-modulated value of Act
func (ap *AttnParams) ActMod(act, attn float32) float32 {
	return act * (ap.ActMin + ap.ActRng*attn)
}

// LrnMod returns attention-modulated value of Lrn
func (ap *AttnParams) LrnMod(lrn, attn float32) float32 {
	return lrn * (ap.LrnMin + ap.LrnRng*attn)
}

///////////////////////////////////////////////////////////////////////////
// AttnLayer

// AttnSetLayer is an interface for setting attention values on a layer
type AttnSetLayer interface {
	leabra.LeabraLayer

	// SetAttn sets the attention value for given pool index
	SetAttn(pidx int, attn float32)
}

// AttnLayer multiplies Ge, Act, and/or ActLrn based on a Pool-level
// normalized Attn factor, which is set externally by e.g., TRCAttnLayer.
type AttnLayer struct {
	leabra.Layer
	Attn  AttnParams `desc:"how Attention modulates Ge, Act and Lrn"`
	Attns []float32  `desc:"per-pool attention modulation factors (normalized 0..1 range"`
}

var KiT_AttnLayer = kit.Types.AddType(&AttnLayer{}, leabra.LayerProps)

func (ly *AttnLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Attn.Defaults()
}

// InitAttn sets all the Attn values to 1
func (ly *AttnLayer) InitAttn() {
	for ni := range ly.Attns {
		ly.Attns[ni] = 1
	}
}

func (ly *AttnLayer) InitActs() {
	ly.Layer.InitActs()
	ly.InitAttn()
}

// SetAttn sets the attention value for given pool index
func (ly *AttnLayer) SetAttn(pidx int, attn float32) {
	ly.Attns[pidx] = attn
}

// RecvGIncPrjn increments the receiver's GeInc or GiInc from that of all the projections.
// Includes Ge Attentional modulation.
func (ly *AttnLayer) RecvGIncPrjn(pj *leabra.Prjn, ltime *leabra.Time) {
	if ly.Attn.GeMin < 1 && pj.Typ == emer.Forward {
		for ri := range ly.Neurons {
			rn := &ly.Neurons[ri]
			ge := ly.Attn.GeMod(pj.GInc[ri], ly.Attns[rn.SubPool])
			rn.GeInc += ge
			pj.GInc[ri] = 0
		}
	} else {
		pj.LeabraPrj.RecvGInc()
	}
}

func (ly *AttnLayer) RecvGInc(ltime *leabra.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		ly.RecvGIncPrjn(p.(leabra.LeabraPrjn).AsLeabra(), ltime)
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *AttnLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	ly.GFmIncNeur(ltime)
}

// AttnActMod modulates the Act and ActLrn by Attn factor
func (ly *AttnLayer) AttnActMod() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		attn := ly.Attns[nrn.SubPool]
		nrn.Act = ly.Attn.ActMod(nrn.Act, attn)
		nrn.ActLrn = ly.Attn.ActMod(nrn.ActLrn, attn)
	}
}

func (ly *AttnLayer) ActFmG(ltime *leabra.Time) {
	ly.Layer.ActFmG(ltime)
	ly.AttnActMod()
}

///////////////////////////////////////////////////////////////////////////
// Neurons

func (ly *AttnLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.Attns = make([]float32, len(ly.Pools))
	return nil
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *AttnLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	if varNm != "Attn" {
		return -1, fmt.Errorf("attrn.AttnLayer: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *AttnLayer) UnitVal1D(varIdx int, idx int) float32 {
	nn := ly.Layer.UnitVarNum()
	if varIdx < 0 || varIdx > nn { // nn = AlphaMax
		return math32.NaN()
	}
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	nrn := &ly.Neurons[idx]
	return ly.Attns[nrn.SubPool]
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *AttnLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 1
}
