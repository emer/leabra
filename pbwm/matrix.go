// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import "github.com/emer/leabra/leabra"

// MatrixParams has parameters for Dorsal Striatum Matrix computation
// These are the main Go / NoGo gating units in BG driving updating of PFC WM in PBWM
type MatrixParams struct {
	PatchShunt  float32 `def:"0.2,0.5" min:"0" max:"1" desc:"how much the patch shunt activation multiplies the dopamine values -- 0 = complete shunting, 1 = no shunting -- should be a factor < 1.0"`
	ShuntACh    bool    `def:"true" desc:"also shunt the ACh value driven from TAN units -- this prevents clearing of MSNConSpec traces -- more plausibly the patch units directly interfere with the effects of TAN's rather than through ach, but it is easier to implement with ach shunting here."`
	OutAChInhib float32 `def:"0:0.3" desc:"how much does the LACK of ACh from the TAN units drive extra inhibition to output-gating Matrix units -- gi += out_ach_inhib * (1-ach) -- provides a bias for output gating on reward trials -- do NOT apply to NoGo, only Go -- this is a key param -- between 0.1-0.3 usu good -- see how much output gating happening and change accordingly"`
}

func (mp *MatrixParams) Defaults() {
	mp.PatchShunt = 0.2
	mp.ShuntACh = true
	mp.OutAChInhib = 0.3
}

// MatrixNeuron contains extra variables for MatrixLayer neurons -- stored separately
type MatrixNeuron struct {
	DA    float32 `desc:"per-neuron modulated dopamine level, derived from layer DA and Shunt"`
	ACh   float32 `desc:"per-neuron modulated ACh level, derived from layer ACh and Shunt"`
	Shunt float32 `desc:"Shunting input received from Patch neurons (in reality flows through SNc DA pathways)"`
}

// MatrixParams has parameters for Dorsal Striatum Matrix computation
// These are the main Go / NoGo gating units in BG driving updating of PFC WM in PBWM
// D1R = Go, D2R = NoGo, and outer 4D Pool X dimension determines GateTypes per MaintN
// (Maint on the left up to MaintN, Out on the right after)
type MatrixLayer struct {
	ModLayer
	MaintN      int            `desc:"number of Maint Pools in X outer dimension of 4D shape -- Out gating after that"`
	DaR         DaReceptors    `desc:"dominant type of dopamine receptor -- D1R for Go pathway, D2R for NoGo"`
	Matrix      MatrixParams   `desc:"matrix parameters"`
	ThalGate    []ThalGate     `desc:"slice of thalamic gating state values for this layer -- in one-to-one correspondence with Pools (0 = layer, 1 = first Pool, etc)"`
	MatrixNeurs []MatrixNeuron `desc:"slice of MatrixNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values."`
}

// UnitValByIdx returns value of given variable by variable index
// and flat neuron index (from layer or neuron-specific one).
// First indexes are ModNeuronVars
func (ly *MatrixLayer) UnitValByIdx(vidx int, idx int) float32 {
	mnrn := &ly.MatrixNeurs[idx]
	nrn := &ly.Neurons[idx]
	switch vidx {
	case 0:
		return mnrn.DA
	case 1:
		return mnrn.ACh
	case 2:
		return ly.SE
	case 3: // ThalAct
		return ly.ThalGate[nrn.SubPool].Act
	case 4: // ThalGate
		return ly.ThalGate[nrn.SubPool].Gate
	case 5: // ThalCnt
		return float32(ly.ThalGate[nrn.SubPool].Cnt)
	case 6: // shows up as Cust1 in GUI
		return mnrn.Shunt
	}
	return 0
}

// Build constructs the layer state, including calling Build on the projections
// you MUST have properly configured the Inhib.Pool.On setting by this point
// to properly allocate Pools for the unit groups if necessary.
func (ly *MatrixLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.ThalGate = make([]ThalGate, len(ly.Pools))
	ly.MatrixNeurs = make([]MatrixNeuron, len(ly.Neurons))
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *MatrixLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.MatrixNeurs {
		mnr := &ly.MatrixNeurs[ni]
		mnr.DA = 0
		mnr.ACh = 0
		mnr.Shunt = 0
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
	achI := ly.Matrix.OutAChInhib * (1 - ly.ACh) // ACh comes from TAN neurons, represents ??

	ypN := ly.Shp.Dim(0)
	xpN := ly.Shp.Dim(1)
	ynN := ly.Shp.Dim(2)
	xnN := ly.Shp.Dim(3)
	for yp := 0; yp < ypN; yp++ {
		for xp := ly.MaintN; xp < xpN; xp++ {
			for yn := 0; yn < ynN; yn++ {
				for xn := 0; xn < xnN; xn++ {
					i := ly.Shp.Offset([]int{yp, xp, yn, xn})
					nrn := &ly.Neurons[i]
					if nrn.IsOff() {
						continue
					}
					nrn.Gi += achI
				}
			}
		}
	}
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
	}
}

// Todo: Save ThalGate state?  what else?  when are above called?
