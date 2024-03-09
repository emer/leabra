// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"fmt"

	"github.com/emer/leabra/v2/leabra"
	"github.com/goki/mat32"
)

// TraceSyn holds extra synaptic state for trace projections
type TraceSyn struct {

	// new trace -- drives updates to trace value -- su * (1-ru_msn) for gated, or su * ru_msn for not-gated (or for non-thalamic cases)
	NTr float32 `desc:"new trace -- drives updates to trace value -- su * (1-ru_msn) for gated, or su * ru_msn for not-gated (or for non-thalamic cases)"`

	//  current ongoing trace of activations, which drive learning -- adds ntr and clears after learning on current values -- includes both thal gated (+ and other nongated, - inputs)
	Tr float32 `desc:" current ongoing trace of activations, which drive learning -- adds ntr and clears after learning on current values -- includes both thal gated (+ and other nongated, - inputs)"`
}

// VarByName returns synapse variable by name
func (sy *TraceSyn) VarByName(varNm string) float32 {
	switch varNm {
	case "NTr":
		return sy.NTr
	case "Tr":
		return sy.Tr
	}
	return mat32.NaN()
}

// VarByIndex returns synapse variable by index
func (sy *TraceSyn) VarByIndex(varIdx int) float32 {
	switch varIdx {
	case 0:
		return sy.NTr
	case 1:
		return sy.Tr
	}
	return mat32.NaN()
}

var TraceSynVars = []string{"NTr", "Tr"}

// Params for for trace-based learning in the MatrixTracePrjn
type TraceParams struct {

	// [def: 0.7] [min: 0] learning rate for all not-gated stripes, which learn in the opposite direction to the gated stripes, and typically with a slightly lower learning rate -- although there are different learning logics associated with each of these different not-gated cases, in practice the same learning rate for all works best, and is simplest
	NotGatedLR float32 `def:"0.7" min:"0" desc:"learning rate for all not-gated stripes, which learn in the opposite direction to the gated stripes, and typically with a slightly lower learning rate -- although there are different learning logics associated with each of these different not-gated cases, in practice the same learning rate for all works best, and is simplest"`

	// [def: 0.1] [min: 0] learning rate for gated, NoGo (D2), positive dopamine (weights decrease) -- this is the single most important learning parameter here -- by making this relatively small (but non-zero), an asymmetry in the role of Go vs. NoGo is established, whereby the NoGo pathway focuses largely on punishing and preventing actions associated with negative outcomes, while those assoicated with positive outcomes only very slowly get relief from this NoGo pressure -- this is critical for causing the model to explore other possible actions even when a given action SOMETIMES produces good results -- NoGo demands a very high, consistent level of good outcomes in order to have a net decrease in these avoidance weights.  Note that the gating signal applies to both Go and NoGo MSN's for gated stripes, ensuring learning is about the action that was actually selected (see not_ cases for logic for actions that were close but not taken)
	GateNoGoPosLR float32 `def:"0.1" min:"0" desc:"learning rate for gated, NoGo (D2), positive dopamine (weights decrease) -- this is the single most important learning parameter here -- by making this relatively small (but non-zero), an asymmetry in the role of Go vs. NoGo is established, whereby the NoGo pathway focuses largely on punishing and preventing actions associated with negative outcomes, while those assoicated with positive outcomes only very slowly get relief from this NoGo pressure -- this is critical for causing the model to explore other possible actions even when a given action SOMETIMES produces good results -- NoGo demands a very high, consistent level of good outcomes in order to have a net decrease in these avoidance weights.  Note that the gating signal applies to both Go and NoGo MSN's for gated stripes, ensuring learning is about the action that was actually selected (see not_ cases for logic for actions that were close but not taken)"`

	// [def: 0] [min: 0] decay driven by receiving unit ACh value, sent by CIN units, for reseting the trace
	AChDecay float32 `min:"0" def:"0" desc:"decay driven by receiving unit ACh value, sent by CIN units, for reseting the trace"`

	// [def: 1] [min: 0] multiplier on trace activation for decaying prior traces -- new trace magnitude drives decay of prior trace -- if gating activation is low, then new trace can be low and decay is slow, so increasing this factor causes learning to be more targeted on recent gating changes
	Decay float32 `min:"0" def:"1" desc:"multiplier on trace activation for decaying prior traces -- new trace magnitude drives decay of prior trace -- if gating activation is low, then new trace can be low and decay is slow, so increasing this factor causes learning to be more targeted on recent gating changes"`

	// [def: true] use the sigmoid derivative factor 2 * act * (1-act) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)
	Deriv bool `def:"true" desc:"use the sigmoid derivative factor 2 * act * (1-act) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)"`
}

func (tp *TraceParams) Defaults() {
	tp.NotGatedLR = 0.7
	tp.GateNoGoPosLR = 0.1
	tp.AChDecay = 0 // not useful at all, surprisingly.
	tp.Decay = 1
	tp.Deriv = true
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
	leabra.Prjn

	// [view: inline] special parameters for matrix trace learning
	Trace TraceParams `view:"inline" desc:"special parameters for matrix trace learning"`

	// trace synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIdx array
	TrSyns []TraceSyn `desc:"trace synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIdx array"`
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

			da := rlayi.UnitValByIdx(DA, int(ri)) // note: more efficient to just assume same for all units
			daLrn := rlayi.UnitValByIdx(DALrn, int(ri))
			ach := rlayi.UnitValByIdx(ACh, int(ri))
			gateAct := rlayi.UnitValByIdx(GateAct, int(ri))
			achDk := mat32.Min(1, ach*pj.Trace.AChDecay)
			tr := trsy.Tr

			dwt := float32(0)
			if da != 0 {
				dwt = daLrn * tr
				if d2r && da > 0 && tr < 0 {
					dwt *= pj.Trace.GateNoGoPosLR
				}
			}

			tr -= achDk * tr

			newNTr := pj.Trace.LrnFactor(rn.Act) * sn.Act
			ntr := float32(0)
			if gateAct > 0 { // gated
				ntr = newNTr
			} else { // not-gated
				ntr = -pj.Trace.NotGatedLR * newNTr // opposite sign for non-gated
			}

			decay := pj.Trace.Decay * mat32.Abs(ntr) // decay is function of new trace
			if decay > 1 {
				decay = 1
			}
			tr += ntr - decay*tr
			trsy.Tr = tr
			trsy.NTr = ntr

			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFmAbsDWt(&sy.Norm, mat32.Abs(dwt))
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

///////////////////////////////////////////////////////////////////////////////
// SynVals

// SynVarIdx returns the index of given variable within the synapse,
// according to *this prjn's* SynVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (pj *MatrixTracePrjn) SynVarIdx(varNm string) (int, error) {
	vidx, err := pj.Prjn.SynVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	nn := len(leabra.SynapseVars)
	switch varNm {
	case "NTr":
		return nn, nil
	case "Tr":
		return nn + 1, nil
	}
	return -1, fmt.Errorf("MatrixTracePrjn SynVarIdx: variable name: %v not valid", varNm)
}

// SynVal1D returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pj *MatrixTracePrjn) SynVal1D(varIdx int, synIdx int) float32 {
	if varIdx < 0 || varIdx >= len(SynVarsAll) {
		return mat32.NaN()
	}
	nn := len(leabra.SynapseVars)
	if varIdx < nn {
		return pj.Prjn.SynVal1D(varIdx, synIdx)
	}
	if synIdx < 0 || synIdx >= len(pj.TrSyns) {
		return mat32.NaN()
	}
	varIdx -= nn
	sy := &pj.TrSyns[synIdx]
	return sy.VarByIndex(varIdx)
}
