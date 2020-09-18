// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"log"

	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// BurstParams determine how the 5IB Burst activation is computed from
// standard Act activation values in SuperLayer -- thresholded.
type BurstParams struct {
	BurstQtr leabra.Quarters `desc:"Quarter(s) when bursting occurs -- typically Q4 but can also be Q2 and Q4 for beta-frequency updating.  Note: this is a bitflag and must be accessed using its Set / Has etc routines, 32 bit versions."`
	ThrRel   float32         `max:"1" def:"0.1,0.2,0.5" desc:"Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds."`
	ThrAbs   float32         `min:"0" max:"1" def:"0.1,0.2,0.5" desc:"Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  Overall effective threshold is MAX of relative and absolute thresholds."`
}

func (db *BurstParams) Defaults() {
	db.BurstQtr.Set(int(leabra.Q4))
	db.ThrRel = 0.1
	db.ThrAbs = 0.1
}

// TRCAttnParams determine how the TRCLayer activation modulates SuperLayer activations
type TRCAttnParams struct {
	On     bool    `desc:"is attentional modulation active?"`
	Min    float32 `desc:"minimum act multiplier if attention is 0"`
	TRCLay string  `desc:"name of TRC layer -- defaults to layer name + P"`
}

func (at *TRCAttnParams) Defaults() {
	at.Min = 0.8
}

// ModVal returns the attn-modulated value
func (at *TRCAttnParams) ModVal(val float32, attn float32) float32 {
	return val * (at.Min + (1-at.Min)*attn)
}

// SuperLayer is the DeepLeabra superficial layer, based on basic rate-coded leabra.Layer.
// Computes the Burst activation from regular activations.
type SuperLayer struct {
	TopoInhibLayer               // access as .TopoInhibLayer
	Burst          BurstParams   `view:"inline" desc:"parameters for computing Burst from act, in Superficial layers (but also needed in Deep layers for deep self connections)"`
	Attn           TRCAttnParams `view:"inline" desc:"determine how the TRCLayer activation modulates SuperLayer feedforward excitatory conductances, representing TRC effects on layer V4 inputs (not separately simulated) -- must have a valid layer."`
	SuperNeurs     []SuperNeuron `desc:"slice of super neuron values -- same size as Neurons"`
}

var KiT_SuperLayer = kit.Types.AddType(&SuperLayer{}, LayerProps)

func (ly *SuperLayer) Defaults() {
	ly.TopoInhibLayer.Defaults()
	ly.Act.Init.Decay = 0 // deep doesn't decay!
	ly.Burst.Defaults()
	ly.Attn.Defaults()
	if ly.Attn.TRCLay == "" {
		ly.Attn.TRCLay = ly.Nm + "P"
	}
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *SuperLayer) UpdateParams() {
	ly.TopoInhibLayer.UpdateParams()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *SuperLayer) InitActs() {
	ly.TopoInhibLayer.InitActs()
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.Burst = 0
		snr.BurstPrv = 0
	}
}

func (ly *SuperLayer) DecayState(decay float32) {
	ly.TopoInhibLayer.DecayState(decay)
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.Burst -= decay * (snr.Burst - ly.Act.Init.Act)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  TRC-Based Attention

// TRCLayer returns the TRC layer for attentional modulation
func (ly *SuperLayer) TRCLayer() (*leabra.Layer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.Attn.TRCLay)
	if err != nil {
		err = fmt.Errorf("SuperLayer %s: TRC Layer: %v", ly.Name(), err)
		log.Println(err)
		return nil, err
	}
	return tly.(leabra.LeabraLayer).AsLeabra(), nil
}

// MaxPoolActAvg returns the max Inhib.Act.Avg value across pools
func MaxPoolActAvg(ly *leabra.Layer) float32 {
	laymax := float32(0)
	np := len(ly.Pools)
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		laymax = math32.Max(laymax, pl.Inhib.Act.Avg)
	}
	return laymax
}

func (ly *SuperLayer) ActFmG(ltime *leabra.Time) {
	ly.TopoInhibLayer.ActFmG(ltime)
	if !ly.Attn.On {
		return
	}
	trc, err := ly.TRCLayer()
	if err != nil { // shouldn't happen
		return
	}
	laymax := MaxPoolActAvg(trc)
	thresh := ly.Inhib.ActAvg.Init * .1 // don't apply attn when activation very weak
	if laymax <= thresh {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.SuperNeurs[ni]
		gpavg := trc.Pools[nrn.SubPool].Inhib.Act.Avg // note: requires same shape, validated
		snr.Attn = gpavg / laymax
		nrn.Act = ly.Attn.ModVal(nrn.Act, snr.Attn)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Burst -- computed in CyclePost

// QuarterFinal does updating after end of a quarter
func (ly *SuperLayer) QuarterFinal(ltime *leabra.Time) {
	ly.TopoInhibLayer.QuarterFinal(ltime)
	if ly.Burst.BurstQtr.HasNext(ltime.Quarter) {
		// if will be updating next quarter, save just prior
		// this logic works for all cases, but e.g., BurstPrv doesn't update
		// until end of minus phase for Q4 BurstQtr
		ly.BurstPrv()
	}
}

// BurstPrv saves Burst as BurstPrv
func (ly *SuperLayer) BurstPrv() {
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.BurstPrv = snr.Burst
	}
}

// CyclePost calls BurstFmAct
func (ly *SuperLayer) CyclePost(ltime *leabra.Time) {
	ly.TopoInhibLayer.CyclePost(ltime)
	ly.BurstFmAct(ltime)
}

// BurstFmAct updates Burst layer 5IB bursting value from current Act
// (superficial activation), subject to thresholding.
func (ly *SuperLayer) BurstFmAct(ltime *leabra.Time) {
	if !ly.Burst.BurstQtr.Has(ltime.Quarter) {
		return
	}
	lpl := &ly.Pools[0]
	actMax := lpl.Inhib.Act.Max
	actAvg := lpl.Inhib.Act.Avg
	thr := actAvg + ly.Burst.ThrRel*(actMax-actAvg)
	thr = math32.Max(thr, ly.Burst.ThrAbs)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.SuperNeurs[ni]
		burst := float32(0)
		if nrn.Act > thr {
			burst = nrn.Act
		}
		snr.Burst = burst
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  DeepCtxt -- once after Burst quarter

// SendCtxtGe sends Burst activation over CTCtxtPrjn projections to integrate
// CtxtGe excitatory conductance on CT layers.
// This must be called at the end of the Burst quarter for this layer.
// Satisfies the CtxtSender interface.
func (ly *SuperLayer) SendCtxtGe(ltime *leabra.Time) {
	if !ly.Burst.BurstQtr.Has(ltime.Quarter) {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		snr := &ly.SuperNeurs[ni]
		if snr.Burst > ly.Act.OptThresh.Send {
			for _, sp := range ly.SndPrjns {
				if sp.IsOff() {
					continue
				}
				ptyp := sp.Type()
				if ptyp != CTCtxt {
					continue
				}
				pj, ok := sp.(*CTCtxtPrjn)
				if !ok {
					continue
				}
				pj.SendCtxtGe(ni, snr.Burst)
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Unit Vars

func (ly *SuperLayer) ValidateTRCLayer() error {
	trc, err := ly.TRCLayer()
	if err != nil {
		ly.Attn.On = false
		return err
	}
	if !(trc.Shp.Dim(0) == ly.Shp.Dim(0) && trc.Shp.Dim(1) == ly.Shp.Dim(1)) {
		ly.Attn.On = false
		err = fmt.Errorf("TRC Layer must have the same group-level shape as this layer")
		log.Println(err)
		return err
	}
	return nil
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *SuperLayer) Build() error {
	err := ly.TopoInhibLayer.Build()
	if err != nil {
		return err
	}
	ly.SuperNeurs = make([]SuperNeuron, len(ly.Neurons))
	if ly.Attn.On {
		err = ly.ValidateTRCLayer()
	}
	return err
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *SuperLayer) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *SuperLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.TopoInhibLayer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = SuperNeuronVarIdxByName(varNm)
	if err != nil {
		return vidx, err
	}
	vidx += ly.TopoInhibLayer.UnitVarNum()
	return vidx, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *SuperLayer) UnitVal1D(varIdx int, idx int) float32 {
	if varIdx < 0 {
		return math32.NaN()
	}
	nn := ly.TopoInhibLayer.UnitVarNum()
	if varIdx < nn {
		return ly.TopoInhibLayer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	varIdx -= nn
	if varIdx >= len(SuperNeuronVars) {
		return math32.NaN()
	}
	snr := &ly.SuperNeurs[idx]
	return snr.VarByIdx(varIdx)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *SuperLayer) UnitVarNum() int {
	return ly.TopoInhibLayer.UnitVarNum() + len(SuperNeuronVars)
}
