// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/chewxy/math32"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
)

// TraceSyn holds extra synaptic state for trace projections
type TraceSyn struct {
	NTr float32 `desc:"new trace -- drives updates to trace value -- su * (1-ru_msn) for gated, or su * ru_msn for not-gated (or for non-thalamic cases)"`
	Tr  float32 `desc:" current ongoing trace of activations, which drive learning -- adds ntr and clears after learning on current values -- includes both thal gated (+ and other nongated, - inputs)"`
}

var TraceSynVars = []string{"NTr", "Tr"}

// Params for for trace-based learning in the MatrixTracePrjn
type TraceParams struct {
	NotGatedLR    float32 `def:"0.7" min:"0" desc:"learning rate for all not-gated stripes, which learn in the opposite direction to the gated stripes, and typically with a slightly lower learning rate -- although there are different learning logics associated with each of these different not-gated cases, in practice the same learning rate for all works best, and is simplest"`
	GateNoGoPosLR float32 `def:"0.1" min:"0" desc:"learning rate for gated, NoGo (D2), positive dopamine (weights decrease) -- this is the single most important learning parameter here -- by making this relatively small (but non-zero), an asymmetry in the role of Go vs. NoGo is established, whereby the NoGo pathway focuses largely on punishing and preventing actions associated with negative outcomes, while those assoicated with positive outcomes only very slowly get relief from this NoGo pressure -- this is critical for causing the model to explore other possible actions even when a given action SOMETIMES produces good results -- NoGo demands a very high, consistent level of good outcomes in order to have a net decrease in these avoidance weights.  Note that the gating signal applies to both Go and NoGo MSN's for gated stripes, ensuring learning is about the action that was actually selected (see not_ cases for logic for actions that were close but not taken)"`
	AChResetThr   float32 `min:"0" def:"0.5" desc:"threshold on receiving unit ACh value, sent by TAN units, for reseting the trace"`
	Deriv         bool    `def:"true" desc:"use the sigmoid derivative factor 2 * act * (1-act) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)"`
	Decay         float32 `def:"1" min:"0" desc:"multiplier on trace activation for decaying prior traces -- new trace magnitude drives decay of prior trace -- if gating activation is low, then new trace can be low and decay is slow, so increasing this factor causes learning to be more targeted on recent gating changes"`
}

func (tp *TraceParams) Defaults() {
	tp.NotGatedLR = 0.7
	tp.GateNoGoPosLR = 0.1
	tp.AChResetThr = 0.5
	tp.Deriv = true
	tp.Decay = 1
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

//////////////////////////////////////////////////////////////////////////////////////
//  MatrixTracePrjn

// MatrixTracePrjn does dopamine-modulated, gated trace learning, for Matrix learning
// in PBWM context
type MatrixTracePrjn struct {
	deep.Prjn
	Trace  TraceParams `view:"inline" desc:"special parameters for matrix trace learning"`
	TrSyns []TraceSyn  `desc:"trace synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIdx array"`
}

func (pj *MatrixTracePrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.Trace.Defaults()
	// no additional factors
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
}

func (pj *MatrixTracePrjn) Build() error {
	err := pj.Prjn.Build()
	pj.TrSyns = make([]TraceSyn, len(pj.SConIdx))
	return err
}

func (pj *MatrixTracePrjn) ClearTrace() {
	for si := range pj.TrSyns {
		sy := &pj.TrSyns[si]
		sy.NTr = 0
		sy.Tr = 0
	}
}

func (pj *MatrixTracePrjn) InitWts() {
	pj.Prjn.InitWts()
	pj.ClearTrace()
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *MatrixTracePrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlayi := pj.Recv.(PBWMLayer)
	rlay := rlayi.(*MatrixLayer) // note: won't work if derived
	d2r := (rlay.DaR == D2R)
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		trsyns := pj.TrSyns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			trsy := &trsyns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]

			da := rlayi.UnitValByIdx(DA, int(ri))
			daLrn := rlayi.UnitValByIdx(DALrn, int(ri))
			ach := rlayi.UnitValByIdx(ACh, int(ri))
			tr := trsy.Tr
			gateAct := rlayi.UnitValByIdx(GateAct, int(ri))

			dwt := float32(0)
			if da != 0 {
				dwt = daLrn * tr
				if d2r && da > 0 && tr < 0 {
					dwt *= pj.Trace.GateNoGoPosLR
				}
			}

			if ach >= pj.Trace.AChResetThr {
				tr = 0
			}

			newNTr := pj.Trace.LrnFactor(rn.Act) * sn.Act
			ntr := float32(0)
			if gateAct > 0 { // gated
				ntr = newNTr
			} else { // not-gated
				ntr = -pj.Trace.NotGatedLR * newNTr // opposite sign for non-gated
			}

			decay := pj.Trace.Decay * math32.Abs(ntr) // decay is function of new trace
			if decay > 1 {
				decay = 1
			}
			tr += ntr - decay*tr
			trsy.Tr = tr
			trsy.NTr = ntr

			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFmAbsDWt(&sy.Norm, math32.Abs(dwt))
			} else {
				sy.Norm = trsy.NTr // store in norm, moment!
				sy.Moment = trsy.Tr
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
