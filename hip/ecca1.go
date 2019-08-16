// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hip

import (
	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
)

// hip.EcCa1Prjn is for EC <-> CA1 projections, to perform error-driven
// learning of this encoder pathway according to the ThetaPhase algorithm
// uses Contrastive Hebbian Learning (CHL) on ActP - ActQ1
// Q1: ECin -> CA1 -> ECout       : ActQ1 = minus phase for auto-encoder
// Q2, 3: CA3 -> CA1 -> ECout     : ActM = minus phase for recall
// Q4: ECin -> CA1, ECin -> ECout : ActP = plus phase for everything
type EcCa1Prjn struct {
	leabra.Prjn // access as .Prjn
}

// AsLeabra returns this prjn as a leabra.Prjn -- all derived prjns must redefine
// this to return the base Prjn type, so that the LeabraPrjn interface does not
// need to include accessors to all the basic stuff.
func (pj *EcCa1Prjn) AsLeabra() *leabra.Prjn {
	return &pj.Prjn
}

func (pj *EcCa1Prjn) Defaults() {
	pj.Prjn.Defaults()
	pj.Prjn.Learn.Norm.On = false     // off by default
	pj.Prjn.Learn.Momentum.On = false // off by default
	pj.Prjn.Learn.WtBal.On = false    // todo: experiment
}

func (pj *EcCa1Prjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) -- on sending projections
// CHL version supported if On
func (pj *EcCa1Prjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(leabra.LeabraLayer).AsLeabra()
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

			dwt := (sn.ActP * rn.ActP) - (sn.ActQ1 * rn.ActQ1)
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
