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
)

// GPLayer represents the dorsal matrisome MSN's that are the main
// Go / NoGo gating units in BG.  D1R = Go, D2R = NoGo.
// Minimum activation during gating period drives ActLrn value used for learning.
// Typically just a single unit per Pool representing a given stripe.
type GPLayer struct {
	Layer
	MinActCyc   int       `def:"30" desc:"cycle after which the AlphaMinAct starts being updated, which in turn determines the ActLearn learning activation -- default of 30 is after initial oscillations and should capture gating time activations"`
	AlphaMinAct []float32 `desc:"per-neuron minimum activation value during alpha cycle, after MinActCyc"`
	ACh         float32   `inactive:"+" desc:"acetylcholine value from CIN cholinergic interneurons reflecting the absolute value of reward or CS predictions thereof -- used for resetting the trace of matrix learning"`
}

var KiT_GPLayer = kit.Types.AddType(&GPLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "GPLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.9",
// 		"Layer.Act.Init.Act":  "0.5",
// 		"Layer.Act.Erev.L":    "0.8",
// 		"Layer.Act.Gbar.L":    "0.3", // 0.2 orig
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3", // 5 orig
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *GPLayer) Defaults() {
	ly.Layer.Defaults()
	ly.DA = 0
	if ly.MinActCyc == 0 {
		ly.MinActCyc = 30
	}

	// GP is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Vm = 0.57
	ly.Act.Init.Act = 0.65
	ly.Act.Erev.L = 0.8
	ly.Act.Gbar.L = 0.3
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.ActAvg.Init = 0.25
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.Self.Tau = 3.0
	ly.Act.XX1.Gain = 20  // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3 // fastest
	ly.Act.Dt.GTau = 3
	ly.Act.Init.Decay = 0

	switch {
	case strings.HasSuffix(ly.Nm, "GPeIn"):
		ly.Act.Init.Act = 0.77
	case strings.HasSuffix(ly.Nm, "GPeTA"):
		ly.Act.Init.Act = 0.15
		ly.Act.Init.Vm = 0.50
	}

	for _, pjii := range ly.RcvPrjns {
		pji := pjii.(leabra.LeabraPrjn)
		pj := pji.AsLeabra()
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.9
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		if _, ok := pji.(*GPeInPrjn); ok {
			if _, ok := pj.Send.(*MatrixLayer); ok { // MtxNoToGPeIn -- primary NoGo pathway
				pj.WtScale.Abs = 1
			} else if _, ok := pj.Send.(*GPLayer); ok { // GPeOutToGPeIn
				pj.WtScale.Abs = 0.5
			}
			continue
		}
		if _, ok := pji.(*GPiPrjn); ok {
			continue
		}
		pj.Learn.Learn = false

		if _, ok := pj.Send.(*MatrixLayer); ok {
			pj.WtScale.Abs = 0.5
		} else if _, ok := pj.Send.(*STNLayer); ok {
			pj.WtScale.Abs = 0.1 // default level for GPeOut and GPeTA -- weaker to not oppose GPeIn surge
		}

		switch {
		case strings.HasSuffix(ly.Nm, "GPeOut"):
			if _, ok := pj.Send.(*MatrixLayer); ok { // MtxGoToGPeOut
				pj.WtScale.Abs = 0.5 // Go firing threshold
			}
		case strings.HasSuffix(ly.Nm, "GPeIn"):
			if _, ok := pj.Send.(*STNLayer); ok { // STNpToGPeIn -- stronger to drive burst of activity
				pj.WtScale.Abs = 0.5
			}
		case strings.HasSuffix(ly.Nm, "GPeTA"):
			if _, ok := pj.Send.(*GPLayer); ok { // GPeInToGPeTA
				pj.WtScale.Abs = 0.9 // just enough to knock down to near-zero at baseline
			}
		}
	}

	ly.UpdateParams()
}

// AChLayer interface:

func (ly *GPLayer) GetACh() float32    { return ly.ACh }
func (ly *GPLayer) SetACh(ach float32) { ly.ACh = ach }

// Build constructs the layer state, including calling Build on the projections
// you MUST have properly configured the Inhib.Pool.On setting by this point
// to properly allocate Pools for the unit groups if necessary.
func (ly *GPLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	nn := len(ly.Neurons)
	ly.AlphaMinAct = make([]float32, nn)
	return nil
}

// InitMinAct initializes AlphaMinAct to 0
func (ly *GPLayer) InitMinAct() {
	for pi := range ly.AlphaMinAct {
		ly.AlphaMinAct[pi] = ly.Act.Init.Act
	}
}

func (ly *GPLayer) InitActs() {
	ly.Layer.InitActs()
	ly.InitMinAct()
}

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
// should already have presented the external input to the network at this point.
func (ly *GPLayer) AlphaCycInit() {
	ly.Layer.AlphaCycInit()
	ly.InitMinAct()
}

// MinActFmAct computes the AlphaMinAct values from current activations,
// and updates ActLrn
func (ly *GPLayer) MinActFmAct(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		min := &ly.AlphaMinAct[ni]
		*min = math32.Min(*min, nrn.Act)
		nrn.ActLrn = *min
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act.
// GP extends to compute AlphaMinAct
func (ly *GPLayer) ActFmG(ltime *leabra.Time) {
	ly.Layer.ActFmG(ltime)
	if ltime.Cycle >= ly.MinActCyc {
		ly.MinActFmAct(ltime)
	}
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *GPLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	if !(varNm == "ACh") {
		return -1, fmt.Errorf("bgate.NeuronVars: variable named: %s not found", varNm)
	}
	nn := len(leabra.NeuronVars)
	// nn = DA
	return nn + 1, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *GPLayer) UnitVal1D(varIdx int, idx int) float32 {
	nn := len(leabra.NeuronVars)
	if varIdx < 0 || varIdx > nn+1 { // nn = DA, nn+1 = ACh
		return math32.NaN()
	}
	if varIdx <= nn { //
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	if varIdx > nn+1 {
		return math32.NaN()
	}
	return ly.ACh
}

//////////////////////////////////////////////////////////////////////
// GPeInPrjn

// GPeInPrjn must be used with GPLayer.
// Learns from DA and gating status.
type GPeInPrjn struct {
	leabra.Prjn
}

var KiT_GPeInPrjn = kit.Types.AddType(&GPeInPrjn{}, leabra.PrjnProps)

func (pj *GPeInPrjn) Defaults() {
	pj.Prjn.Defaults()
	// no additional factors
	pj.WtInit.Mean = 0.9
	pj.WtInit.Var = 0
	pj.WtInit.Sym = false
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
	pj.Learn.Lrate = 0.01
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *GPeInPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(*GPLayer)

	da := rlay.DA

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

			dwt := da * rn.ActLrn * sn.ActLrn

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

/*

///////////////////////////////////////////////////////////////////////////////
// SynVals

// SynVarIdx returns the index of given variable within the synapse,
// according to *this prjn's* SynVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (pj *GPeInPrjn) SynVarIdx(varNm string) (int, error) {
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
func (pj *GPeInPrjn) SynVal1D(varIdx int, synIdx int) float32 {
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

*/
