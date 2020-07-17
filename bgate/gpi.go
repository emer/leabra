// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"fmt"
	"strings"

	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// GPiLayer represents the GPi / SNr output nucleus of the BG.
// It gets inhibited by the MtxGo and GPeIn layers, and its minimum
// activation during this inhibition is recorded in ActLrn, for learning.
// Typically just a single unit per Pool representing a given stripe.
type GPiLayer struct {
	GPLayer
}

var KiT_GPiLayer = kit.Types.AddType(&GPiLayer{}, leabra.LayerProps)

func (ly *GPiLayer) Defaults() {
	ly.GPLayer.Defaults()

	// note: GPLayer took care of STN input prjns

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.5
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		if _, ok := pj.Send.(*MatrixLayer); ok { // MtxGoToGPi
			pj.WtScale.Abs = 0.8 // slightly weaker than GPeIn
		} else if _, ok := pj.Send.(*GPLayer); ok { // GPeInToGPi
			pj.WtScale.Abs = 1 // stronger because integrated signal, also act can be weaker
		} else if strings.HasSuffix(pj.Send.Name(), "STNp") { // STNpToGPi
			pj.WtScale.Abs = 1
		} else if strings.HasSuffix(pj.Send.Name(), "STNs") { // STNsToGPi
			pj.WtScale.Abs = 0.2
		}
	}

	ly.UpdateParams()
}

//////////////////////////////////////////////////////////////////////
// GPiPrjn

// GPiTraceParams for for trace-based learning in the GPiPrjn.
// A trace of synaptic co-activity is formed, and then modulated by dopamine
// whenever it occurs.  This bridges the temporal gap between gating activity
// and subsequent activity, and is based biologically on synaptic tags.
// Trace is reset at time of reward based on ACh level from CINs.
type GPiTraceParams struct {
	CurTrlDA bool    `def:"false" desc:"if true, current trial DA dopamine can drive learning (i.e., synaptic co-activity trace is updated prior to DA-driven dWt), otherwise DA is applied to existing trace before trace is updated, meaning that at least one trial must separate gating activity and DA"`
	Decay    float32 `def:"2" min:"0" desc:"multiplier on CIN ACh level for decaying prior traces -- decay never exceeds 1.  larger values drive strong credit assignment for any US outcome."`
	GateAct  float32 `def:"0.2" desc:"activity level below which the stripe is considered to have gated -- provides a crossover point for gating vs. not, which changes sign of learning dynamics"`
}

func (tp *GPiTraceParams) Defaults() {
	tp.CurTrlDA = false
	tp.Decay = 2
	tp.GateAct = 0.2
}

// LrnFactor returns multiplicative factor for GPi activation, centered on GateAct param.
// If act < GateAct, returns (GateAct - act) / GateAct.
// If act > GateAct, returns (GateAct - act) / (1 - GateAct)
func (tp *GPiTraceParams) LrnFactor(act float32) float32 {
	if act < tp.GateAct {
		return (tp.GateAct - act) / tp.GateAct
	}
	return (tp.GateAct - act) / (1 - tp.GateAct)
}

// GPiPrjn must be used with GPi recv layer, from MtxGo, GPeIn senders.
// Learns from DA and ActLrn on GPi neuron.
type GPiPrjn struct {
	leabra.Prjn
	Trace  GPiTraceParams `view:"inline" desc:"parameters for GPi trace learning"`
	TrSyns []TraceSyn     `desc:"trace synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIdx array"`
}

var KiT_GPiPrjn = kit.Types.AddType(&GPiPrjn{}, leabra.PrjnProps)

func (pj *GPiPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.Trace.Defaults()
	// no additional factors
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
	pj.Learn.Lrate = 0.01
}

func (pj *GPiPrjn) Build() error {
	err := pj.Prjn.Build()
	pj.TrSyns = make([]TraceSyn, len(pj.SConIdx))
	return err
}

func (pj *GPiPrjn) ClearTrace() {
	for si := range pj.TrSyns {
		sy := &pj.TrSyns[si]
		sy.NTr = 0
		sy.Tr = 0
	}
}

func (pj *GPiPrjn) InitWts() {
	pj.Prjn.InitWts()
	pj.ClearTrace()
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *GPiPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(*GPiLayer)

	da := rlay.DA

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

			ntr := pj.Trace.LrnFactor(rn.ActLrn) * sn.ActLrn
			dwt := float32(0)

			if pj.Trace.CurTrlDA {
				tr += ntr
			}

			if da != 0 {
				dwt = da * tr
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
func (pj *GPiPrjn) SynVarIdx(varNm string) (int, error) {
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
	return -1, fmt.Errorf("GPiPrjn SynVarIdx: variable name: %v not valid", varNm)
}

// SynVal1D returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pj *GPiPrjn) SynVal1D(varIdx int, synIdx int) float32 {
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
