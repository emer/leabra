// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"fmt"

	"cogentcore.org/core/kit"
	"cogentcore.org/core/mat32"
	"github.com/emer/leabra/v2/leabra"
)

// pbwm.Layer is the base layer type for PBWM framework -- has variables for the
// layer-level neuromodulatory variables: dopamine, ach, serotonin.
// See ModLayer for a version that includes DA-modulated learning parameters,
type Layer struct {
	leabra.Layer

	// current dopamine level for this layer
	DA float32 `inactive:"+"`

	// current acetylcholine level for this layer
	ACh float32 `inactive:"+"`

	// current serotonin level for this layer
	SE float32 `inactive:"+"`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, leabra.LayerProps)

// DALayer interface:

func (ly *Layer) GetDA() float32   { return ly.DA }
func (ly *Layer) SetDA(da float32) { ly.DA = da }

func (ly *Layer) GetACh() float32    { return ly.ACh }
func (ly *Layer) SetACh(ach float32) { ly.ACh = ach }

func (ly *Layer) GetSE() float32   { return ly.SE }
func (ly *Layer) SetSE(se float32) { ly.SE = se }

// AsPBWM returns this layer as a pbwm.Layer
func (ly *Layer) AsPBWM() *Layer {
	return ly
}

// AsGate returns this layer as a pbwm.GateLayer -- nil for Layer
func (ly *Layer) AsGate() *GateLayer {
	return nil
}

// GateSend updates gating state and sends it along to other layers.
// most layers don't implement -- only gating layers
func (ly *Layer) GateSend(ltime *leabra.Time) {
}

// RecGateAct records the gating activation from current activation, when gating occcurs
// based on GateState.Now -- only for gating layers
func (ly *Layer) RecGateAct(ltime *leabra.Time) {
}

// SendMods is called at end of Cycle to send modulator signals (DA, etc)
// which will then be active for the next cycle of processing
func (ly *Layer) SendMods(ltime *leabra.Time) {
}

func (ly *Layer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Init.Decay = 0
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *Layer) UpdateParams() {
	ly.Layer.UpdateParams()
}

// Note: Special layer types should define this!
// func (ly *Layer) Class() string

// UnitVarNames returns a list of variable names available on the units in this layer.
func (ly *Layer) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
// This must be updated for specialized PBWM layer types to return correct variables!
func (ly *Layer) UnitValByIdx(vidx NeurVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	case DALrn:
		return ly.DA
	case ACh:
		return ly.ACh
	case SE:
		return ly.SE
	}
	return mat32.NaN()
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	var ok bool
	vidx, ok = NeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("pbwm.NeuronVars: variable named: %s not found", varNm)
	}
	vidx += ly.Layer.UnitVarNum()
	return vidx, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int, di int) float32 {
	if varIdx < 0 {
		return mat32.NaN()
	}
	nn := ly.Layer.UnitVarNum()
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx, di)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	varIdx -= nn
	return ly.LeabraLay.(PBWMLayer).UnitValByIdx(NeurVars(varIdx), idx)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + int(NeurVarsN)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *Layer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
	ly.ACh = 0
	ly.SE = 0
}

// DoQuarter2DWt indicates whether to do optional Q2 DWt
func (ly *Layer) DoQuarter2DWt() bool {
	return false
}

// QuarterFinal does updating after end of a quarter
func (ly *Layer) QuarterFinal(ltime *leabra.Time) {
	ly.Layer.QuarterFinal(ltime)
	if ltime.Quarter == 1 {
		ly.LeabraLay.(PBWMLayer).Quarter2DWt()
	}
}

// Quarter2DWt is optional Q2 DWt -- define where relevant
func (ly *Layer) Quarter2DWt() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		if rly, ok := p.RecvLay().(PBWMLayer); ok {
			if rly.DoQuarter2DWt() {
				p.(leabra.LeabraPrjn).DWt()
			}
		}
	}
}
