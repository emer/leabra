// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/mat32"
)

// TraceSyn holds extra synaptic state for trace projections
type TraceSyn struct {
	NTr float32 `desc:"new trace -- drives updates to trace value -- su * (1-ru_msn) for gated, or su * ru_msn for not-gated (or for non-thalamic cases)"`
	Tr  float32 `desc:" current ongoing trace of activations, which drive learning -- adds ntr and clears after learning on current values -- includes both thal gated (+ and other nongated, - inputs)"`
}

// VarByName returns synapse variable by name
func (sy *TraceSyn) VarByName(varNm string) float32 {
	switch varNm {
	case "NTr":
		return sy.NTr
	case "Tr":
		return sy.Tr
	}
	return math32.NaN()
}

// VarByIndex returns synapse variable by index
func (sy *TraceSyn) VarByIndex(varIdx int) float32 {
	switch varIdx {
	case 0:
		return sy.NTr
	case 1:
		return sy.Tr
	}
	return math32.NaN()
}

var TraceSynVars = []string{"NTr", "Tr"}

// Params for for trace-based learning in the MatrixTracePrjn.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is reset at time of reward based on ACh level from TANs.
type TraceParams struct {
	CurTrlDA bool    `def:"false" desc:"if true, current trial DA dopamine can drive learning (i.e., synaptic co-activity trace is updated prior to DA-driven dWt), otherwise DA is applied to existing trace before trace is updated, meaning that at least one trial must separate gating activity and DA"`
	Deriv    bool    `def:"true" desc:"use the sigmoid derivative factor 2 * act * (1-act) for matrix (recv) activity in modulating learning -- otherwise just multiply by activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)"`
	Decay    float32 `def:"2" min:"0" desc:"multiplier on TAN ACh level for decaying prior traces -- decay never exceeds 1.  larger values drive strong credit assignment for any US outcome."`
}

func (tp *TraceParams) Defaults() {
	tp.CurTrlDA = false
	tp.Deriv = true
	tp.Decay = 2
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

//////////////////////////////////////////////////////////////////////////////////////
//  MatrixTracePrjn

// MatrixTracePrjn does dopamine-modulated, gated trace learning, for Matrix learning
// in PBWM context
type MatrixTracePrjn struct {
	leabra.Prjn
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
	rlay := pj.Recv.(*MatrixLayer)

	da := rlay.DA
	daLrn := rlay.DALrn // includes d2 reversal etc

	ach := rlay.ACh
	dk := mat32.Min(1, ach*pj.Trace.Decay)

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

			tr := trsy.Tr

			ntr := pj.Trace.LrnFactor(rn.Act) * sn.Act
			dwt := float32(0)

			if pj.Trace.CurTrlDA {
				tr += ntr
			}

			if da != 0 {
				dwt = daLrn * tr
			}
			tr -= dk * tr // decay trace that drove dwt

			if !pj.Trace.CurTrlDA {
				tr += ntr
			}
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
		return math32.NaN()
	}
	nn := len(leabra.SynapseVars)
	if varIdx < nn {
		return pj.Prjn.SynVal1D(varIdx, synIdx)
	}
	if synIdx < 0 || synIdx >= len(pj.TrSyns) {
		return math32.NaN()
	}
	varIdx -= nn
	sy := &pj.TrSyns[synIdx]
	return sy.VarByIndex(varIdx)
}
