// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"log"

	"cogentcore.org/core/math32"
	"github.com/emer/leabra/v2/leabra"
)

// TDRewPredLayer is the temporal differences reward prediction layer.
// It represents estimated value V(t) in the minus phase, and computes
// estimated V(t+1) based on its learned weights in plus phase.
// Use TDRewPredPrjn for DA modulated learning.
type TDRewPredLayer struct {
	leabra.Layer

	// dopamine value for this layer
	DA float32 `inactive:"+"`
}

// DALayer interface:

func (ly *TDRewPredLayer) GetDA() float32   { return ly.DA }
func (ly *TDRewPredLayer) SetDA(da float32) { ly.DA = da }

// ActFmG computes linear activation for TDRewPred
func (ly *TDRewPredLayer) ActFmG(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ltime.Quarter == 3 { // plus phase
			nrn.Act = nrn.Ge // linear
		} else {
			nrn.Act = nrn.ActP // previous actP
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  TDRewIntegLayer

// TDRewIntegParams are params for reward integrator layer
type TDRewIntegParams struct {

	// discount factor -- how much to discount the future prediction from RewPred
	Discount float32

	// name of TDRewPredLayer to get reward prediction from
	RewPred string
}

func (tp *TDRewIntegParams) Defaults() {
	tp.Discount = 0.9
	if tp.RewPred == "" {
		tp.RewPred = "RewPred"
	}
}

// TDRewIntegLayer is the temporal differences reward integration layer.
// It represents estimated value V(t) in the minus phase, and
// estimated V(t+1) + r(t) in the plus phase.
// It computes r(t) from (typically fixed) weights from a reward layer,
// and directly accesses values from RewPred layer.
type TDRewIntegLayer struct {
	leabra.Layer

	// parameters for reward integration
	RewInteg TDRewIntegParams

	// dopamine value for this layer
	DA float32
}

func (ly *TDRewIntegLayer) Defaults() {
	ly.Layer.Defaults()
	ly.RewInteg.Defaults()
}

// DALayer interface:

func (ly *TDRewIntegLayer) GetDA() float32   { return ly.DA }
func (ly *TDRewIntegLayer) SetDA(da float32) { ly.DA = da }

func (ly *TDRewIntegLayer) RewPredLayer() (*TDRewPredLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewInteg.RewPred)
	if err != nil {
		log.Printf("TDRewIntegLayer %s RewPredLayer: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(*TDRewPredLayer), nil
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *TDRewIntegLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	_, err = ly.RewPredLayer()
	return err
}

func (ly *TDRewIntegLayer) ActFmG(ltime *leabra.Time) {
	rply, _ := ly.RewPredLayer()
	if rply == nil {
		return
	}
	rpActP := rply.Neurons[0].ActP
	rpAct := rply.Neurons[0].Act
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ltime.Quarter == 3 { // plus phase
			nrn.Act = nrn.Ge + ly.RewInteg.Discount*rpAct
		} else {
			nrn.Act = rpActP // previous actP
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  TDDaLayer

// TDDaLayer computes a dopamine (DA) signal as the temporal difference (TD)
// between the TDRewIntegLayer activations in the minus and plus phase.
type TDDaLayer struct {
	leabra.Layer

	// list of layers to send dopamine to
	SendDA SendDA

	// name of TDRewIntegLayer from which this computes the temporal derivative
	RewInteg string

	// dopamine value for this layer
	DA float32
}

func (ly *TDDaLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Clamp.Range.Set(-100, 100)
	if ly.RewInteg == "" {
		ly.RewInteg = "RewInteg"
	}
}

// DALayer interface:

func (ly *TDDaLayer) GetDA() float32   { return ly.DA }
func (ly *TDDaLayer) SetDA(da float32) { ly.DA = da }

func (ly *TDDaLayer) RewIntegLayer() (*TDRewIntegLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.RewInteg)
	if err != nil {
		log.Printf("TDDaLayer %s RewIntegLayer: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(*TDRewIntegLayer), nil
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *TDDaLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendDA.Validate(ly.Network, ly.Name()+" SendTo list")
	if err != nil {
		return err
	}
	_, err = ly.RewIntegLayer()
	return err
}

func (ly *TDDaLayer) ActFmG(ltime *leabra.Time) {
	rily, _ := ly.RewIntegLayer()
	if rily == nil {
		return
	}
	rpActP := rily.Neurons[0].Act
	rpActM := rily.Neurons[0].ActM
	da := rpActP - rpActM
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ltime.Quarter == 3 { // plus phase
			nrn.Act = da
		} else {
			nrn.Act = 0
		}
	}
}

// CyclePost is called at end of Cycle
// We use it to send DA, which will then be active for the next cycle of processing.
func (ly *TDDaLayer) CyclePost(ltime *leabra.Time) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA.SendDA(ly.Network, act)
}

//////////////////////////////////////////////////////////////////////////////////////
//  TDRewPredPrjn

// TDRewPredPrjn does dopamine-modulated learning for reward prediction:
// DWt = Da * Send.ActQ0 (activity on *previous* timestep)
// Use in TDRewPredLayer typically to generate reward predictions.
// Has no weight bounds or limits on sign etc.
type TDRewPredPrjn struct {
	leabra.Prjn
}

func (pj *TDRewPredPrjn) Defaults() {
	pj.Prjn.Defaults()
	// no additional factors
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *TDRewPredPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	// rlay := pj.Recv.(leabra.LeabraLayer).AsLeabra()
	da := pj.Recv.(DALayer).GetDA()
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIndexSt[si])
		syns := pj.Syns[st : st+nc]
		// scons := pj.SConIndex[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			// ri := scons[ci]

			dwt := da * sn.ActQ0 // no recv unit activation, prior trial act

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
func (pj *TDRewPredPrjn) WtFmDWt() {
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
