// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"log"

	"github.com/chewxy/math32"
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// RWPredLayer computes reward prediction for a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
// Activity is computed as linear function of excitatory conductance
// (which can be negative -- there are no constraints).
// Use with RWPrjn which does simple delta-rule learning on minus-plus.
type RWPredLayer struct {
	ModLayer
	PredRange minmax.F32 `desc:"default 0.1..0.99 range of predictions that can be represented -- having a truncated range preserves some sensitivity in dopamine at the extremes of good or poor performance"`
}

var KiT_RWPredLayer = kit.Types.AddType(&RWPredLayer{}, deep.LayerProps)

func (ly *RWPredLayer) Defaults() {
	ly.ModLayer.Defaults()
	ly.PredRange.Set(0.01, 0.99)
}

// ActFmG computes linear activation for RWPred
func (ly *RWPredLayer) ActFmG(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = ly.PredRange.ClipVal(nrn.Ge) // clipped linear
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  RWDaLayer

// RWDaLayer computes a dopamine (Da) signal based on a simple Rescorla-Wagner
// learning dynamic (i.e., PV learning in the PVLV framework).
// It computes difference between r(t) and RWPred inputs.
// r(t) is accessed directly from a Rew layer -- if no external input then no
// DA is computed -- critical for effective use of RW only for PV cases.
// Receives RWPred prediction from direct (fixed) weights.
type RWDaLayer struct {
	DaSrcLayer
	RewLay string `desc:"name of Reward-representing layer from which this computes DA -- if nothing clamped, no dopamine computed"`
}

var KiT_RWDaLayer = kit.Types.AddType(&RWDaLayer{}, deep.LayerProps)

func (ly *RWDaLayer) Defaults() {
	ly.DaSrcLayer.Defaults()
	ly.RewLay = "Rew"
}

func (ly *RWDaLayer) RewLayer() (*ModLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewLay)
	if err != nil {
		log.Printf("RWDaLayer %s, RewLay: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(PBWMLayer).AsMod(), nil
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *RWDaLayer) Build() error {
	err := ly.ModLayer.Build()
	if err != nil {
		return err
	}
	_, err = ly.RewLayer()
	return err
}

func (ly *RWDaLayer) ActFmG(ltime *leabra.Time) {
	rly, _ := ly.RewLayer()
	if rly == nil {
		return
	}
	rnrn := &(rly.Neurons[0])
	hasRew := false
	if rnrn.HasFlag(leabra.NeurHasExt) {
		hasRew = true
	}
	ract := rnrn.Act
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if hasRew {
			nrn.Act = ract - nrn.Ge
		} else {
			nrn.Act = 0 // nothing
		}
	}
}

// SendMods is called at end of Cycle to send modulator signals (DA, etc)
// which will then be active for the next cycle of processing
func (ly *RWDaLayer) SendMods(ltime *leabra.Time) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA(act)
}

//////////////////////////////////////////////////////////////////////////////////////
//  RWPrjn

// RWPrjn does dopamine-modulated learning for reward prediction: Da * Send.Act
// Use in RWPredLayer typically to generate reward predictions.
// Has no weight bounds or limits on sign etc.
type RWPrjn struct {
	deep.Prjn
	DaTol float32 `desc:"tolerance on DA -- if below this abs value, then DA goes to zero and there is no learning -- prevents prediction from exactly learning to cancel out reward value, retaining a residual valence of signal"`
}

var KiT_RWPrjn = kit.Types.AddType(&RWPrjn{}, deep.PrjnProps)

func (pj *RWPrjn) Defaults() {
	pj.Prjn.Defaults()
	// no additional factors
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *RWPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlayi := pj.Recv.(PBWMLayer)
	rlay := rlayi.AsLeabra()
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]

			da := rlayi.UnitValByIdx(DALrn, int(ri))
			if pj.DaTol > 0 {
				if math32.Abs(da) <= pj.DaTol {
					da = 0
				}
			}
			if rn.Ge > rn.Act && da > 0 { // clipped at top, saturate up
				da = 0
			}
			if rn.Ge < rn.Act && da < 0 { // clipped at bottom, saturate down
				da = 0
			}

			dwt := da * sn.Act // no recv unit activation

			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFmAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pj.Learn.Momentum.On {
				dwt = norm * pj.Learn.Momentum.MomentFmDWt(&sy.Moment, dwt)
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

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *RWPrjn) WtFmDWt() {
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
