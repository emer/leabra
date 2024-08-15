// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hip

import (
	"cogentcore.org/core/math32"
	"github.com/emer/leabra/v2/leabra"
)

// hip.EcCa1Path is for EC <-> CA1 pathways, to perform error-driven
// learning of this encoder pathway according to the ThetaPhase algorithm
// uses Contrastive Hebbian Learning (CHL) on ActP - ActQ1
// Q1: ECin -> CA1 -> ECout       : ActQ1 = minus phase for auto-encoder
// Q2, 3: CA3 -> CA1 -> ECout     : ActM = minus phase for recall
// Q4: ECin -> CA1, ECin -> ECout : ActP = plus phase for everything
type EcCa1Path struct {
	leabra.Path // access as .Path
}

func (pj *EcCa1Path) Defaults() {
	pj.Path.Defaults()
	pj.Path.Learn.Norm.On = false     // off by default
	pj.Path.Learn.Momentum.On = false // off by default
	pj.Path.Learn.WtBal.On = false    // todo: experiment
}

func (pj *EcCa1Path) UpdateParams() {
	pj.Path.UpdateParams()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) -- on sending pathways
// Delta version
func (pj *EcCa1Path) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(leabra.LeabraLayer).AsLeabra()
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

			err := (sn.ActP * rn.ActP) - (sn.ActQ1 * rn.ActQ1)
			bcm := pj.Learn.BCMdWt(sn.AvgSLrn, rn.AvgSLrn, rn.AvgL)
			bcm *= pj.Learn.XCal.LongLrate(rn.AvgLLrn)
			err *= pj.Learn.XCal.MLrn
			dwt := bcm + err

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
