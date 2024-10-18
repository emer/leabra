// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32/minmax"
	"github.com/emer/emergent/v2/paths"
)

////////  RW

type RWParams struct {
	// PredRange is the range of predictions that can be represented by the [RWRewPredLayer].
	// Having a truncated range preserves some sensitivity in dopamine at the extremes
	// of good or poor performance.
	PredRange minmax.F32

	// RewLay is the reward layer name, for [RWDaLayer], from which DA is obtained.
	// If nothing clamped, no dopamine computed.
	RewLay string

	// PredLay is the name of [RWPredLayer] layer, for [RWDaLayer], that is used for
	// subtracting prediction from the reward value.
	PredLay string
}

func (rp *RWParams) Defaults() {
	rp.PredRange.Set(0.01, 0.99)
	rp.RewLay = "Rew"
	rp.PredLay = "RWPred"
}

func (rp *RWParams) Update() {
}

// ActFromGRWPred computes linear activation for [RWPredLayer].
func (ly *Layer) ActFromGRWPred(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Act = ly.RW.PredRange.ClipValue(nrn.Ge) // clipped linear
		ly.Learn.AvgsFromAct(nrn)
	}
}

// RWLayers returns the reward and RWPredLayer layers based on names.
func (ly *Layer) RWLayers() (*Layer, *Layer, error) {
	tly := ly.Network.LayerByName(ly.RW.RewLay)
	if tly == nil {
		err := fmt.Errorf("RWDaLayer %s, RewLay: %q not found", ly.Name, ly.RW.RewLay)
		return nil, nil, errors.Log(err)
	}
	ply := ly.Network.LayerByName(ly.RW.PredLay)
	if ply == nil {
		err := fmt.Errorf("RWDaLayer %s, RWPredLay: %q not found", ly.Name, ly.RW.PredLay)
		return nil, nil, errors.Log(err)
	}
	return tly, ply, nil
}

func (ly *Layer) ActFromGRWDa(ctx *Context) {
	rly, ply, _ := ly.RWLayers()
	if rly == nil || ply == nil {
		return
	}
	rnrn := &(rly.Neurons[0])
	hasRew := false
	if rnrn.HasFlag(NeurHasExt) {
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
		ly.Learn.AvgsFromAct(nrn)
	}
}

// AddRWLayers adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
func (nt *Network) AddRWLayers(prefix string, space float32) (rew, rp, da *Layer) {
	rew = nt.AddLayer2D(prefix+"Rew", 1, 1, InputLayer)
	rp = nt.AddLayer2D(prefix+"RWPred", 1, 1, RWPredLayer)
	da = nt.AddLayer2D(prefix+"DA", 1, 1, RWDaLayer)
	da.RW.RewLay = rew.Name
	rp.PlaceBehind(rew, space)
	da.PlaceBehind(rp, space)
	return
}

func (pt *Path) RWDefaults() {
	pt.Learn.WtSig.Gain = 1
	pt.Learn.Norm.On = false
	pt.Learn.Momentum.On = false
	pt.Learn.WtBal.On = false
}

// DWtRW computes the weight change (learning) for [RWPath].
func (pt *Path) DWtRW() {
	slay := pt.Send
	rlay := pt.Recv
	lda := pt.Recv.DA
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pt.SConN[si])
		st := int(pt.SConIndexSt[si])
		syns := pt.Syns[st : st+nc]
		scons := pt.SConIndex[st : st+nc]

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
			sy.DWt += pt.Learn.Lrate * dwt
		}
	}
}

////////  TD

// TDParams are params for TD temporal differences computation.
type TDParams struct {

	// discount factor -- how much to discount the future prediction from RewPred.
	Discount float32

	// name of [TDPredLayer] to get reward prediction from.
	PredLay string

	// name of [TDIntegLayer] from which this computes the temporal derivative.
	IntegLay string
}

func (tp *TDParams) Defaults() {
	tp.Discount = 0.9
	tp.PredLay = "Pred"
	tp.IntegLay = "Integ"
}

func (tp *TDParams) Update() {
}

// ActFromGTDPred computes linear activation for [TDPredLayer].
func (ly *Layer) ActFromGTDPred(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ctx.Quarter == 3 { // plus phase
			nrn.Act = nrn.Ge // linear
		} else {
			nrn.Act = nrn.ActP // previous actP
		}
		ly.Learn.AvgsFromAct(nrn)
	}
}

func (ly *Layer) TDPredLayer() (*Layer, error) {
	tly := ly.Network.LayerByName(ly.TD.PredLay)
	if tly == nil {
		err := fmt.Errorf("TDIntegLayer %s RewPredLayer: %q not found", ly.Name, ly.TD.PredLay)
		return nil, errors.Log(err)
	}
	return tly, nil
}

func (ly *Layer) ActFromGTDInteg(ctx *Context) {
	rply, _ := ly.TDPredLayer()
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
		if ctx.Quarter == 3 { // plus phase
			nrn.Act = nrn.Ge + ly.TD.Discount*rpAct
		} else {
			nrn.Act = rpActP // previous actP
		}
		ly.Learn.AvgsFromAct(nrn)
	}
}

func (ly *Layer) TDIntegLayer() (*Layer, error) {
	tly := ly.Network.LayerByName(ly.TD.IntegLay)
	if tly == nil {
		err := fmt.Errorf("TDIntegLayer %s RewIntegLayer: %q not found", ly.Name, ly.TD.IntegLay)
		return nil, errors.Log(err)
	}
	return tly, nil
}

func (ly *Layer) TDDaDefaults() {
	ly.Act.Clamp.Range.Set(-100, 100)
}

func (ly *Layer) ActFromGTDDa(ctx *Context) {
	rily, _ := ly.TDIntegLayer()
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
		if ctx.Quarter == 3 { // plus phase
			nrn.Act = da
		} else {
			nrn.Act = 0
		}
	}
}

func (pt *Path) TDPredDefaults() {
	pt.Learn.WtSig.Gain = 1
	pt.Learn.Norm.On = false
	pt.Learn.Momentum.On = false
	pt.Learn.WtBal.On = false
}

// DWtTDPred computes the weight change (learning) for [TDPredPath].
func (pt *Path) DWtTDPred() {
	slay := pt.Send
	rlay := pt.Recv
	da := rlay.DA
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pt.SConN[si])
		st := int(pt.SConIndexSt[si])
		syns := pt.Syns[st : st+nc]
		// scons := pj.SConIndex[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			// ri := scons[ci]
			dwt := da * sn.ActQ0 // no recv unit activation, prior trial act
			sy.DWt += pt.Learn.Lrate * dwt
		}
	}
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Pathway from Rew to RewInteg is given class TDToInteg -- should
// have no learning and 1 weight.
func (nt *Network) AddTDLayers(prefix string, space float32) (rew, rp, ri, td *Layer) {
	rew = nt.AddLayer2D(prefix+"Rew", 1, 1, InputLayer)
	rp = nt.AddLayer2D(prefix+"Pred", 1, 1, TDPredLayer)
	ri = nt.AddLayer2D(prefix+"Integ", 1, 1, TDIntegLayer)
	td = nt.AddLayer2D(prefix+"TD", 1, 1, TDDaLayer)
	ri.TD.PredLay = rp.Name
	td.TD.IntegLay = ri.Name
	rp.PlaceBehind(rew, space)
	ri.PlaceBehind(rp, space)
	td.PlaceBehind(ri, space)

	pt := nt.ConnectLayers(rew, ri, paths.NewFull(), ForwardPath)
	pt.AddClass("TDToInteg")
	pt.Learn.Learn = false
	pt.WtInit.Mean = 1
	pt.WtInit.Var = 0
	pt.WtInit.Sym = false
	// {Sel: ".TDToInteg", Desc: "rew to integ",
	// 	Params: params.Params{
	// 		"Path.Learn.Learn": "false",
	// 		"Path.WtInit.Mean": "1",
	// 		"Path.WtInit.Var":  "0",
	// 		"Path.WtInit.Sym":  "false",
	// 	}},
	return
}
