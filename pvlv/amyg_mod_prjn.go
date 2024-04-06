// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"cogentcore.org/core/mat32"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/leabra/v2/leabra"
)

// IAmygPrjn has one method, AsAmygModPrjn, which recasts the projection as a moddulatory projection
type IAmygPrjn interface {
	AsAmygModPrjn() *AmygModPrjn // recast the projection as a moddulatory projection
}

// AsAmygModPrjn returns a pointer to the modulatory variables for an amygdala projection
func (pj *AmygModPrjn) AsAmygModPrjn() *AmygModPrjn {
	return pj
}

// ISetScalePrjn initializes weights, including special scale calculations
type ISetScalePrjn interface {
	InitWts()
}

// AmygModPrjn holds parameters and state variables for modulatory projections to amygdala layers
type AmygModPrjn struct {
	leabra.Prjn

	// only for Leabra algorithm: if initializing the weights, set the connection scaling parameter in addition to intializing the weights -- for specifically-supported specs, this will for example set a gaussian scaling parameter on top of random initial weights, instead of just setting the initial weights to a gaussian weighted value -- for other specs that do not support a custom init_wts function, this will set the scale values to what the random weights would otherwise be set to, and set the initial weight value to a constant (init_wt_val)
	SetScale bool

	// minimum scale value for SetScale projections
	SetScaleMin float32

	// maximum scale value for SetScale projections
	SetScaleMax float32

	// constant initial weight value for specs that do not support a custom init_wts function and have set_scale set: the scale values are set to what the random weights would otherwise be set to, and the initial weight value is set to this constant: the net actual weight value is scale * init_wt_val..
	InitWtVal float32

	// gain multiplier on abs(DA) learning rate multiplier
	DALRGain float32

	// constant baseline amount of learning prior to abs(DA) factor -- should be near zero otherwise offsets in activation will drive learning in the absence of DA significance
	DALRBase float32

	// minimum threshold for phasic abs(da) signals to count as non-zero;  useful to screen out spurious da signals due to tiny VSPatch-to-LHb signals on t2 & t4 timesteps that can accumulate over many trials - 0.02 seems to work okay
	DALrnThr float32

	// minimum threshold for delta activation to count as non-zero;  useful to screen out spurious learning due to unintended delta activity - 0.02 seems to work okay for both acquisition and extinction guys
	ActDeltaThr float32

	// if true, recv unit deep_lrn value modulates learning
	ActLrnMod bool

	// only ru->deep_lrn values > this get to learn - 0.05f seems to work okay
	ActLrnThr float32

	// parameters for dopaminergic modulation
	DaMod DaModParams
}

// These null variables serve as a check that AmygModPrjn actually implements the IAmygPrjn and ISetScalePrjn interfaces
// If any methods are not implemented, these statements will not compile
var _ IAmygPrjn = (*AmygModPrjn)(nil)
var _ ISetScalePrjn = (*AmygModPrjn)(nil)

// InitWts sets initial weights, possibly including SetScale calculations
func (pj *AmygModPrjn) InitWts() {
	if pj.SetScale {
		pj.SetScalesFunc(pj.GaussScale)
		pj.SetWtsFunc(func(_, _ int, _, _ *etensor.Shape) float32 {
			return pj.InitWtVal
		})
		for si := range pj.Syns {
			sy := &pj.Syns[si]
			sy.DWt = 0
			sy.Norm = 0
			sy.Moment = 0
		}
	} else {
		pj.Prjn.InitWts()
	}
}

// GaussScale returns gaussian weight value for given unit indexes in
// given send and recv layers according to Gaussian Sigma and MaxWt.
func (pj *AmygModPrjn) GaussScale(_, _ int, _, _ *etensor.Shape) float32 {
	scale := float32(pj.WtInit.Gen(-1))
	scale = mat32.Max(pj.SetScaleMin, scale)
	scale = mat32.Min(pj.SetScaleMax, scale)
	return scale
}

func (pj *AmygModPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.SetScale = false
	pj.InitWtVal = 0.1
	pj.DALRGain = 1.0
	pj.DALRBase = 1.0
	pj.DALrnThr = 0.0
	pj.ActDeltaThr = 0.05
	pj.ActLrnThr = 0.05
	pj.ActLrnMod = true
}

// DWt computes DA-modulated weight changes for amygdala layers
func (pj *AmygModPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlayi := pj.Recv.(IModLayer)
	rlay := rlayi.AsMod()
	clRate := pj.Learn.Lrate // * rlay.CosDiff.ModAvgLLrn
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		snAct := sn.ActQ0
		nc := int(pj.SConN[si])
		st := int(pj.SConIndexSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIndex[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			mn := &rlay.ModNeurs[ri]

			if rn.IsOff() {
				continue
			}
			// filter any tiny spurious da signals on t2 & t4 trials - best for ext guys since
			// they have zero dalr_base value
			if mat32.Abs(mn.DA) < pj.DALrnThr {
				mn.DA = 0
			}

			lRateEff := clRate
			// learning dependent on non-zero deep_lrn
			if pj.ActLrnMod {
				var effActLrn float32
				if mn.ModLrn > pj.ActLrnThr {
					effActLrn = 1
				} else {
					effActLrn = 0 // kills all learning
				}
				lRateEff *= effActLrn
			}

			rnActDelta := mn.ModAct - rn.ActQ0
			if mat32.Abs(rnActDelta) < pj.ActDeltaThr {
				rnActDelta = 0
			}
			delta := lRateEff * snAct * rnActDelta
			// dopamine signal further modulates learning
			daLRate := pj.DALRBase + pj.DALRGain*mat32.Abs(mn.DA)
			sy.DWt += daLRate * delta
		}
	}
}
