// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/math32"
)

func (pt *Path) CTCtxtDefaults() {
	pp := &pt.Params
	if pt.FromSuper {
		pp.Learn.Learn = false
		pp.WtInit.Mean = 0.5 // .5 better than .8 in several cases..
		pp.WtInit.Var = 0
	}
}

// SendCtxtGe sends the full Burst activation from sending neuron index si,
// to integrate CtxtGe excitatory conductance on receivers
func (pt *Path) SendCtxtGe(si int, dburst float32) {
	scdb := dburst * pt.GScale
	nc := pt.SConN[si]
	st := pt.SConIndexSt[si]
	syns := pt.Syns[st : st+nc]
	scons := pt.SConIndex[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pt.CtxtGeInc[ri] += scdb * syns[ci].Wt
	}
}

// RecvCtxtGeInc increments the receiver's CtxtGe from that of all the pathways
func (pt *Path) RecvCtxtGeInc() {
	rlay := pt.Recv
	for ri := range rlay.Neurons {
		rlay.Neurons[ri].CtxtGe += pt.CtxtGeInc[ri]
		pt.CtxtGeInc[ri] = 0
	}
}

// DWt computes the weight change (learning) for CTCtxt pathways.
func (pt *Path) DWtCTCtxt() {
	pp := &pt.Params
	slay := pt.Send
	issuper := pt.Send.Params.Type == SuperLayer
	rlay := pt.Recv
	for si := range slay.Neurons {
		sact := float32(0)
		if issuper {
			sact = slay.Neurons[si].BurstPrv
		} else {
			sact = slay.Neurons[si].ActQ0
		}
		nc := int(pt.SConN[si])
		st := int(pt.SConIndexSt[si])
		syns := pt.Syns[st : st+nc]
		scons := pt.SConIndex[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			// following line should be ONLY diff: sact for *both* short and medium *sender*
			// activations, which are first two args:
			err, bcm := pp.Learn.CHLdWt(sact, sact, rn.AvgSLrn, rn.AvgM, rn.AvgL)

			bcm *= pp.Learn.XCal.LongLrate(rn.AvgLLrn)
			err *= pp.Learn.XCal.MLrn
			dwt := bcm + err
			norm := float32(1)
			if pp.Learn.Norm.On {
				norm = pp.Learn.Norm.NormFromAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pp.Learn.Momentum.On {
				dwt = norm * pp.Learn.Momentum.MomentFromDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pp.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pp.Learn.Norm.On {
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
