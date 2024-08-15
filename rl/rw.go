// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"log"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"github.com/emer/leabra/v2/leabra"
)

// RWPredLayer computes reward prediction for a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
// Activity is computed as linear function of excitatory conductance
// (which can be negative -- there are no constraints).
// Use with RWPath which does simple delta-rule learning on minus-plus.
type RWPredLayer struct {
	leabra.Layer

	// default 0.1..0.99 range of predictions that can be represented -- having a truncated range preserves some sensitivity in dopamine at the extremes of good or poor performance
	PredRange minmax.F32

	// dopamine value for this layer
	DA float32 `edit:"-"`
}

func (ly *RWPredLayer) Defaults() {
	ly.Layer.Defaults()
	ly.PredRange.Set(0.01, 0.99)
}

// DALayer interface:

func (ly *RWPredLayer) GetDA() float32   { return ly.DA }
func (ly *RWPredLayer) SetDA(da float32) { ly.DA = da }

// ActFromG computes linear activation for RWPred
func (ly *RWPredLayer) ActFromG(ctx *leabra.Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = ly.PredRange.ClipValue(nrn.Ge) // clipped linear
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  RWDaLayer

// RWDaLayer computes a dopamine (DA) signal based on a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
// It computes difference between r(t) and RWPred values.
// r(t) is accessed directly from a Rew layer -- if no external input then no
// DA is computed -- critical for effective use of RW only for PV cases.
// RWPred prediction is also accessed directly from Rew layer to avoid any issues.
type RWDaLayer struct {
	leabra.Layer

	// list of layers to send dopamine to
	SendDA SendDA

	// name of Reward-representing layer from which this computes DA -- if nothing clamped, no dopamine computed
	RewLay string

	// name of RWPredLayer layer that is subtracted from the reward value
	RWPredLay string

	// dopamine value for this layer
	DA float32 `edit:"-"`
}

func (ly *RWDaLayer) Defaults() {
	ly.Layer.Defaults()
	if ly.RewLay == "" {
		ly.RewLay = "Rew"
	}
	if ly.RWPredLay == "" {
		ly.RWPredLay = "RWPred"
	}
}

// DALayer interface:

func (ly *RWDaLayer) GetDA() float32   { return ly.DA }
func (ly *RWDaLayer) SetDA(da float32) { ly.DA = da }

// RWLayers returns the reward and RWPred layers based on names
func (ly *RWDaLayer) RWLayers() (*leabra.Layer, *RWPredLayer, error) {
	tly, err := ly.Network.LayerByName(ly.RewLay)
	if err != nil {
		log.Printf("RWDaLayer %s, RewLay: %v\n", ly.Name(), err)
		return nil, nil, err
	}
	ply, err := ly.Network.LayerByName(ly.RWPredLay)
	if err != nil {
		log.Printf("RWDaLayer %s, RWPredLay: %v\n", ly.Name(), err)
		return nil, nil, err
	}
	return tly.(leabra.LeabraLayer).AsLeabra(), ply.(*RWPredLayer), nil
}

// Build constructs the layer state, including calling Build on the pathways.
func (ly *RWDaLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendDA.Validate(ly.Network, ly.Name()+" SendTo list")
	if err != nil {
		return err
	}
	_, _, err = ly.RWLayers()
	return err
}

func (ly *RWDaLayer) ActFromG(ctx *leabra.Context) {
	rly, ply, _ := ly.RWLayers()
	if rly == nil || ply == nil {
		return
	}
	rnrn := &(rly.Neurons[0])
	hasRew := false
	if rnrn.HasFlag(leabra.NeurHasExt) {
		hasRew = true
	}
	ract := rnrn.Act
	pnrn := &(ply.Neurons[0])
	pact := pnrn.Act
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if hasRew {
			nrn.Act = ract - pact
		} else {
			nrn.Act = 0 // nothing
		}
	}
}

// CyclePost is called at end of Cycle
// We use it to send DA, which will then be active for the next cycle of processing.
func (ly *RWDaLayer) CyclePost(ctx *leabra.Context) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA.SendDA(ly.Network, act)
}

//////////////////////////////////////////////////////////////////////////////////////
//  RWPath

// RWPath does dopamine-modulated learning for reward prediction: Da * Send.Act
// Use in RWPredLayer typically to generate reward predictions.
// Has no weight bounds or limits on sign etc.
type RWPath struct {
	leabra.Path

	// tolerance on DA -- if below this abs value, then DA goes to zero and there is no learning -- prevents prediction from exactly learning to cancel out reward value, retaining a residual valence of signal
	DaTol float32
}

func (pj *RWPath) Defaults() {
	pj.Path.Defaults()
	// no additional factors
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
}

// DWt computes the weight change (learning) -- on sending pathways.
func (pj *RWPath) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(leabra.LeabraLayer).AsLeabra()
	lda := pj.Recv.(DALayer).GetDA()
	if pj.DaTol > 0 {
		if math32.Abs(lda) <= pj.DaTol {
			return // lda = 0 -- no learning
		}
	}
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIndexSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIndex[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]

			da := lda
			if rn.Ge > rn.Act && da > 0 { // clipped at top, saturate up
				da = 0
			}
			if rn.Ge < rn.Act && da < 0 { // clipped at bottom, saturate down
				da = 0
			}

			dwt := da * sn.Act // no recv unit activation

			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFromAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pj.Learn.Momentum.On {
				dwt = norm * pj.Learn.Momentum.MomentFromDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pj.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pj.Learn.Norm.On {
			maxNorm := float32(0)
			for ci := range syns {
				sy := &syns[ci]
				if sy.Norm > maxNorm {
					maxNorm = sy.Norm
				}
			}
			for ci := range syns {
				sy := &syns[ci]
				sy.Norm = maxNorm
			}
		}
	}
}

// WtFromDWt updates the synaptic weight values from delta-weight changes -- on sending pathways
func (pj *RWPath) WtFromDWt() {
	if !pj.Learn.Learn {
		return
	}
	for si := range pj.Syns {
		sy := &pj.Syns[si]
		if sy.DWt != 0 {
			sy.Wt += sy.DWt // straight update, no limits or anything
			sy.LWt = sy.Wt
			sy.DWt = 0
		}
	}
}
