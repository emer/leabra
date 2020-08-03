// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/chewxy/math32"
	"github.com/emer/leabra/attrn"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// BurstParams are parameters determining how the DeepBurst activation is computed
// from regular activation values.
type BurstParams struct {
	BurstQtr leabra.Quarters `desc:"Quarter(s) when bursting occurs -- typically Q4 but can also be Q2 and Q4 for beta-frequency updating.  Note: this is a bitflag and must be accessed using its Set / Has etc routines, 32 bit versions."`
	ThrRel   float32         `max:"1" def:"0.1,0.2,0.5" desc:"Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds."`
	ThrAbs   float32         `min:"0" max:"1" def:"0.1,0.2,0.5" desc:"Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  Overall effective threshold is MAX of relative and absolute thresholds."`
}

func (db *BurstParams) Update() {
}

func (db *BurstParams) Defaults() {
	db.BurstQtr.Set(int(leabra.Q4))
	db.ThrRel = 0.1
	db.ThrAbs = 0.1
}

// SuperNeuron has the neuron values for SuperLayer
type SuperNeuron struct {
	Burst    float32 `desc:"5IB bursting activation value, computed by thresholding regular activation"`
	BurstPrv float32 `desc:"previous bursting activation -- used for context-based learning"`
}

func (sn *SuperNeuron) VarByIdx(idx int) float32 {
	switch NeurVars(idx) {
	case BurstVar:
		return sn.Burst
	case BurstPrvVar:
		return sn.BurstPrv
	}
	return math32.NaN()
}

var (
	SuperNeuronVars = []string{"Burst", "BurstPrv"}
)

// SuperLayer is the DeepLeabra superficial layer, based on basic rate-coded leabra.Layer.
// Computes the Burst activation from regular activations.
type SuperLayer struct {
	attrn.AttnLayer               // access as .AttnLayer
	Burst           BurstParams   `view:"inline" desc:"parameters for computing Burst from act, in Superficial layers (but also needed in Deep layers for deep self connections)"`
	SuperNeurs      []SuperNeuron `desc:"slice of super neuron values -- same size as Neurons"`
}

var KiT_SuperLayer = kit.Types.AddType(&SuperLayer{}, LayerProps)

func (ly *SuperLayer) Defaults() {
	ly.AttnLayer.Defaults()
	ly.Act.Init.Decay = 0 // deep doesn't decay!
	ly.Burst.Defaults()
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *SuperLayer) UpdateParams() {
	ly.AttnLayer.UpdateParams()
	ly.Burst.Update()
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *SuperLayer) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *SuperLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.AttnLayer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = NeuronVarByName(varNm)
	if err != nil {
		return vidx, err
	}
	vidx += ly.AttnLayer.UnitVarNum()
	return vidx, err
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *SuperLayer) UnitVal1D(varIdx int, idx int) float32 {
	if varIdx < 0 {
		return math32.NaN()
	}
	nn := ly.AttnLayer.UnitVarNum()
	if varIdx < nn {
		return ly.AttnLayer.UnitVal1D(varIdx, idx)
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
	return ly.AttnLayer.UnitVarNum() + len(SuperNeuronVars)
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *SuperLayer) Build() error {
	err := ly.AttnLayer.Build()
	if err != nil {
		return err
	}
	ly.SuperNeurs = make([]SuperNeuron, len(ly.Neurons))
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *SuperLayer) InitActs() {
	ly.AttnLayer.InitActs()
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.Burst = 0
		snr.BurstPrv = 0
	}
}

func (ly *SuperLayer) DecayState(decay float32) {
	ly.AttnLayer.DecayState(decay)
	for ni := range ly.SuperNeurs {
		snr := &ly.SuperNeurs[ni]
		snr.Burst -= decay * (snr.Burst - ly.Act.Init.Act)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Burst -- computed in CyclePost

// QuarterFinal does updating after end of a quarter
func (ly *SuperLayer) QuarterFinal(ltime *leabra.Time) {
	ly.AttnLayer.QuarterFinal(ltime)
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
	ly.AttnLayer.CyclePost(ltime)
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
