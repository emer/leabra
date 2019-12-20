// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// MatrixParams has parameters for Dorsal Striatum Matrix computation
// These are the main Go / NoGo gating units in BG driving updating of PFC WM in PBWM
type MatrixParams struct {
	PatchShunt  float32 `def:"0.2,0.5" min:"0" max:"1" desc:"how much the patch shunt activation multiplies the dopamine values -- 0 = complete shunting, 1 = no shunting -- should be a factor < 1.0"`
	ShuntACh    bool    `def:"true" desc:"also shunt the ACh value driven from TAN units -- this prevents clearing of MSNConSpec traces -- more plausibly the patch units directly interfere with the effects of TAN's rather than through ach, but it is easier to implement with ach shunting here."`
	OutAChInhib float32 `def:"0,0.3" desc:"how much does the LACK of ACh from the TAN units drive extra inhibition to output-gating Matrix units -- gi += out_ach_inhib * (1-ach) -- provides a bias for output gating on reward trials -- do NOT apply to NoGo, only Go -- this is a key param -- between 0.1-0.3 usu good -- see how much output gating happening and change accordingly"`
	BurstGain   float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
	DipGain     float32 `def:"1" desc:"multiplicative gain factor applied to positive (burst) dopamine signals in computing DALrn effect learning dopamine value based on raw DA that we receive (D2R reversal occurs *after* applying Burst based on sign of raw DA)"`
}

func (mp *MatrixParams) Defaults() {
	mp.PatchShunt = 0.2
	mp.ShuntACh = true
	mp.OutAChInhib = 0.3
	mp.BurstGain = 1
	mp.DipGain = 1
}

// MatrixNeuron contains extra variables for MatrixLayer neurons -- stored separately
type MatrixNeuron struct {
	DA    float32 `desc:"per-neuron modulated dopamine level, derived from layer DA and Shunt"`
	DALrn float32 `desc:"per-neuron effective learning dopamine value -- gain modulated and sign reversed for D2R"`
	ACh   float32 `desc:"per-neuron modulated ACh level, derived from layer ACh and Shunt"`
	Shunt float32 `desc:"shunting input received from Patch neurons (in reality flows through SNc DA pathways)"`
	ActG  float32 `desc:"gating activation -- the activity value when gating occurred in this pool"`
}

// MatrixLayer represents the dorsal matrisome MSN's that are the main
// Go / NoGo gating units in BG driving updating of PFC WM in PBWM.
// D1R = Go, D2R = NoGo, and outer 4D Pool X dimension determines GateTypes per MaintN
// (Maint on the left up to MaintN, Out on the right after)
type MatrixLayer struct {
	GateLayer
	MaintN      int            `desc:"number of Maint Pools in X outer dimension of 4D shape -- Out gating after that"`
	DaR         DaReceptors    `desc:"dominant type of dopamine receptor -- D1R for Go pathway, D2R for NoGo"`
	Matrix      MatrixParams   `view:"inline" desc:"matrix parameters"`
	MatrixNeurs []MatrixNeuron `desc:"slice of MatrixNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values."`
}

var KiT_MatrixLayer = kit.Types.AddType(&MatrixLayer{}, deep.LayerProps)

// Defaults in param.Sheet format
// Sel: "MatrixLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.Gi":     "1.9",
// 		"Layer.Inhib.Layer.FB":     "0.5",
// 		"Layer.Inhib.Pool.On":      "true",
// 		"Layer.Inhib.Pool.Gi":      "1.9",
// 		"Layer.Inhib.Pool.FB":      "0",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "1.3",
// 		"Layer.Inhib.ActAvg.Init":  "0.2",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 	}}

func (ly *MatrixLayer) Defaults() {
	ly.GateLayer.Defaults()
	ly.Matrix.Defaults()
	ly.DeepBurst.SetBurstQtr(leabra.Q2) // also
	// special inhib params
	ly.Inhib.Layer.Gi = 1.9
	ly.Inhib.Layer.FB = 0.5
	ly.Inhib.Pool.On = true
	ly.Inhib.Pool.Gi = 1.9
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.3
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 0.2
}

func (ly *MatrixLayer) GateType() GateTypes {
	return MaintOut // always both
}

// DALrnFmDA returns effective learning dopamine value from given raw DA value
// applying Burst and Dip Gain factors, and then reversing sign for D2R.
func (ly *MatrixLayer) DALrnFmDA(da float32) float32 {
	if da > 0 {
		da *= ly.Matrix.BurstGain
	} else {
		da *= ly.Matrix.DipGain
	}
	if ly.DaR == D2R {
		da *= -1
	}
	return da
}

// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *MatrixLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	mnrn := &ly.MatrixNeurs[idx]
	nrn := &ly.Neurons[idx]
	gs := ly.GateState(int(nrn.SubPool) - 1) // 0-based
	switch vidx {
	case DA:
		return mnrn.DA
	case DALrn:
		return mnrn.DALrn
	case ACh:
		return mnrn.ACh
	case SE:
		return ly.SE
	case GateAct:
		return gs.Act
	case GateNow:
		if gs.Now {
			return 1
		}
		return 0
	case GateCnt:
		return float32(gs.Cnt)
	case ActG:
		return mnrn.ActG
	case Cust1:
		return mnrn.Shunt
	}
	return 0
}

// Build constructs the layer state, including calling Build on the projections
// you MUST have properly configured the Inhib.Pool.On setting by this point
// to properly allocate Pools for the unit groups if necessary.
func (ly *MatrixLayer) Build() error {
	err := ly.GateLayer.Build()
	if err != nil {
		return err
	}
	ly.MatrixNeurs = make([]MatrixNeuron, len(ly.Neurons))
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *MatrixLayer) InitActs() {
	ly.GateLayer.InitActs()
	for ni := range ly.MatrixNeurs {
		mnr := &ly.MatrixNeurs[ni]
		mnr.DA = 0
		mnr.DALrn = 0
		mnr.ACh = 0
		mnr.Shunt = 0
		mnr.ActG = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// InhibiFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
// Matrix version applies OutAChInhib to bias output gating on reward trials
func (ly *MatrixLayer) InhibFmGeAct(ltime *leabra.Time) {
	ly.ModLayer.InhibFmGeAct(ltime)

	if ly.Matrix.OutAChInhib == 0 {
		return
	}

	ypN := ly.Shp.Dim(0)
	xpN := ly.Shp.Dim(1)
	ynN := ly.Shp.Dim(2)
	xnN := ly.Shp.Dim(3)
	for yp := 0; yp < ypN; yp++ {
		for xp := ly.MaintN; xp < xpN; xp++ {
			for yn := 0; yn < ynN; yn++ {
				for xn := 0; xn < xnN; xn++ {
					ni := ly.Shp.Offset([]int{yp, xp, yn, xn})
					nrn := &ly.Neurons[ni]
					if nrn.IsOff() {
						continue
					}
					mnr := &ly.MatrixNeurs[ni]
					achI := ly.Matrix.OutAChInhib * (1 - mnr.ACh) // ACh comes from TAN neurons, represents ??
					nrn.Gi += achI
				}
			}
		}
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act.
// Matrix extends to call DaAChFmLay
func (ly *MatrixLayer) ActFmG(ltime *leabra.Time) {
	ly.DaAChFmLay(ltime)
	ly.GateLayer.ActFmG(ltime)
}

// DaAChFmLay computes Da and ACh from layer and Shunt received from PatchLayer units
func (ly *MatrixLayer) DaAChFmLay(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		mnr := &ly.MatrixNeurs[ni]
		mnr.DA = ly.DA
		mnr.ACh = ly.ACh
		if mnr.Shunt > 0 { // note: treating Shunt as binary variable -- could multiply
			mnr.DA *= ly.Matrix.PatchShunt
			if ly.Matrix.ShuntACh {
				mnr.ACh *= ly.Matrix.PatchShunt
			}
		}
		mnr.DALrn = ly.DALrnFmDA(mnr.DA)
	}
}

// RecGateAct records the gating activation from current activation, when gating occcurs
// based on GateState.Now
func (ly *MatrixLayer) RecGateAct(ltime *leabra.Time) {
	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		if !gs.Now { // not gating now
			continue
		}
		pl := &ly.Pools[1+gi]
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			mnr := &ly.MatrixNeurs[ni]
			mnr.ActG = nrn.Act
		}
	}
}

// DoQuarter2DWt indicates whether to do optional Q2 DWt
func (ly *MatrixLayer) DoQuarter2DWt() bool {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(1) {
		return false
	}
	return true
}
