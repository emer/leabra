// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/math32"
)

// Params for for trace-based learning in the MatrixTracePath
type TraceParams struct {

	// learning rate for all not-gated stripes, which learn in the opposite direction to the gated stripes, and typically with a slightly lower learning rate -- although there are different learning logics associated with each of these different not-gated cases, in practice the same learning rate for all works best, and is simplest
	NotGatedLR float32 `default:"0.7" min:"0"`

	// learning rate for gated, NoGo (D2), positive dopamine (weights decrease) -- this is the single most important learning parameter here -- by making this relatively small (but non-zero), an asymmetry in the role of Go vs. NoGo is established, whereby the NoGo pathway focuses largely on punishing and preventing actions associated with negative outcomes, while those assoicated with positive outcomes only very slowly get relief from this NoGo pressure -- this is critical for causing the model to explore other possible actions even when a given action SOMETIMES produces good results -- NoGo demands a very high, consistent level of good outcomes in order to have a net decrease in these avoidance weights.  Note that the gating signal applies to both Go and NoGo MSN's for gated stripes, ensuring learning is about the action that was actually selected (see not_ cases for logic for actions that were close but not taken)
	GateNoGoPosLR float32 `default:"0.1" min:"0"`

	// decay driven by receiving unit ACh value, sent by CIN units, for reseting the trace
	AChDecay float32 `min:"0" default:"0"`

	// multiplier on trace activation for decaying prior traces -- new trace magnitude drives decay of prior trace -- if gating activation is low, then new trace can be low and decay is slow, so increasing this factor causes learning to be more targeted on recent gating changes
	Decay float32 `min:"0" default:"1"`

	// use the sigmoid derivative factor 2 * act * (1-act) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)
	Deriv bool `default:"true"`
}

func (tp *TraceParams) Defaults() {
	tp.NotGatedLR = 0.7
	tp.GateNoGoPosLR = 0.1
	tp.AChDecay = 0 // not useful at all, surprisingly.
	tp.Decay = 1
	tp.Deriv = true
}

func (tp *TraceParams) Update() {
}

// LrnFactor resturns multiplicative factor for level of msn activation.  If Deriv
// is 2 * act * (1-act) -- the factor of 2 compensates for otherwise reduction in
// learning from these factors.  Otherwise is just act.
func (tp *TraceParams) LrnFactor(act float32) float32 {
	if !tp.Deriv {
		return act
	}
	return 2 * act * (1 - act)
}

// LrateMod returns the learning rate modulator based on gating, d2r, and posDa factors
func (tp *TraceParams) LrateMod(gated, d2r, posDa bool) float32 {
	if !gated {
		return tp.NotGatedLR
	}
	if d2r && posDa {
		return tp.GateNoGoPosLR
	}
	return 1
}

func (pt *Path) MatrixDefaults() {
	pt.Learn.WtSig.Gain = 1
	pt.Learn.Norm.On = false
	pt.Learn.Momentum.On = false
	pt.Learn.WtBal.On = false
}

func (pt *Path) ClearTrace() {
	for si := range pt.Syns {
		sy := &pt.Syns[si]
		sy.NTr = 0
		sy.Tr = 0
	}
}

// DWtMatrix computes the weight change (learning) for MatrixPath.
func (pt *Path) DWtMatrix() {
	slay := pt.Send
	rlay := pt.Recv
	d2r := (rlay.PBWM.DaR == D2R)
	da := rlay.DA
	ach := rlay.ACh
	gateActIdx, _ := NeuronVarIndexByName("GateAct")
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
			daLrn := rn.DALrn
			// da := rlay.UnitValueByIndex(DA, int(ri)) // note: more efficient to just assume same for all units
			// ach := rlay.UnitValueByIndex(ACh, int(ri))
			gateAct := rlay.UnitValue1D(gateActIdx, int(ri), 0)
			achDk := math32.Min(1, ach*pt.Trace.AChDecay)
			tr := sy.Tr

			dwt := float32(0)
			if da != 0 {
				dwt = daLrn * tr
				if d2r && da > 0 && tr < 0 {
					dwt *= pt.Trace.GateNoGoPosLR
				}
			}

			tr -= achDk * tr

			newNTr := pt.Trace.LrnFactor(rn.Act) * sn.Act
			ntr := float32(0)
			if gateAct > 0 { // gated
				ntr = newNTr
			} else { // not-gated
				ntr = -pt.Trace.NotGatedLR * newNTr // opposite sign for non-gated
			}

			decay := pt.Trace.Decay * math32.Abs(ntr) // decay is function of new trace
			if decay > 1 {
				decay = 1
			}
			tr += ntr - decay*tr
			sy.Tr = tr
			sy.NTr = ntr

			norm := float32(1)
			if pt.Learn.Norm.On {
				norm = pt.Learn.Norm.NormFromAbsDWt(&sy.Norm, math32.Abs(dwt))
			} else {
				sy.Norm = sy.NTr // store in norm, moment!
				sy.Moment = sy.Tr
			}
			if pt.Learn.Momentum.On {
				dwt = norm * pt.Learn.Momentum.MomentFromDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pt.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pt.Learn.Norm.On {
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
