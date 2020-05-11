// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"fmt"
	"log"

	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// ModLayer is the base layer type for PBWM framework -- has variables for the
// layer-level neuromodulatory variables: dopamine, ach, serotonin.
// The pbwm.Layer is a usable generic version of this base ModLayer,
// and other more specialized types build directly from ModLayer.
type ModLayer struct {
	deep.Layer
	DA  float32 `desc:"current dopamine level for this layer"`
	ACh float32 `desc:"current acetylcholine level for this layer"`
	SE  float32 `desc:"current serotonin level for this layer"`
}

var KiT_ModLayer = kit.Types.AddType(&ModLayer{}, deep.LayerProps)

// AsMod returns this layer as a pbwm.ModLayer
func (ly *ModLayer) AsMod() *ModLayer {
	return ly
}

// AsGate returns this layer as a pbwm.GateLayer -- nil for ModLayer
func (ly *ModLayer) AsGate() *GateLayer {
	return nil
}

// GateSend updates gating state and sends it along to other layers.
// most layers don't implement -- only gating layers
func (ly *ModLayer) GateSend(ltime *leabra.Time) {
}

// RecGateAct records the gating activation from current activation, when gating occcurs
// based on GateState.Now -- only for gating layers
func (ly *ModLayer) RecGateAct(ltime *leabra.Time) {
}

// SendMods is called at end of Cycle to send modulator signals (DA, etc)
// which will then be active for the next cycle of processing
func (ly *ModLayer) SendMods(ltime *leabra.Time) {
}

func (ly *ModLayer) Defaults() {
	ly.Layer.Defaults()
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *ModLayer) UpdateParams() {
	ly.Layer.UpdateParams()
}

// Note: Special layer types should define this!
// func (ly *Layer) Class() string

// UnitVarNames returns a list of variable names available on the units in this layer
// Mod returns *layer level* vars
func (ly *ModLayer) UnitVarNames() []string {
	return ModNeuronVarsAll
}

// NeuronVars are indexes into extra PBWM neuron-level variables
type NeuronVars int

const (
	DA NeuronVars = iota
	DALrn
	ACh
	SE
	GateAct
	GateNow
	GateCnt
	ActG
	Cust1
	NeuronVarsN
)

var (
	// ModNeuronVars are the modulator neurons plus some custom variables that sub-types use for their
	// algo-specific cases -- need a consistent set of overall network-level vars for display / generic
	// interface.
	ModNeuronVars    = []string{"DA", "DALrn", "ACh", "SE", "GateAct", "GateNow", "GateCnt", "ActG", "Cust1"}
	ModNeuronVarsMap map[string]int
	ModNeuronVarsAll []string
)

func init() {
	ModNeuronVarsMap = make(map[string]int, len(ModNeuronVars))
	for i, v := range ModNeuronVars {
		ModNeuronVarsMap[v] = i
	}
	ln := len(deep.NeuronVarsAll)
	ModNeuronVarsAll = make([]string, len(ModNeuronVars)+ln)
	copy(ModNeuronVarsAll, deep.NeuronVarsAll)
	copy(ModNeuronVarsAll[ln:], ModNeuronVars)
}

// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *ModLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
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
	return 0
}

// UnitValTry returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *ModLayer) UnitValTry(varNm string, idx []int) (float32, error) {
	vidx, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return ly.Layer.UnitValTry(varNm, idx)
	}
	fidx := ly.Shp.Offset(idx)
	return ly.LeabraLay.(PBWMLayer).UnitValByIdx(NeuronVars(vidx), fidx), nil
}

// UnitVal1DTry returns value of given variable name on given unit,
// using 1-dimensional index.
//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *ModLayer) UnitVal1DTry(varNm string, idx int) (float32, error) {
	vidx, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return ly.Layer.UnitVal1DTry(varNm, idx)
	}
	return ly.LeabraLay.(PBWMLayer).UnitValByIdx(NeuronVars(vidx), idx), nil
}

// UnitVals fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *ModLayer) UnitVals(vals *[]float32, varNm string) error {
	vidx, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return ly.Layer.UnitVals(vals, varNm)
	}
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	for i := 0; i < nn; i++ {
		(*vals)[i] = ly.LeabraLay.(PBWMLayer).UnitValByIdx(NeuronVars(vidx), i)
	}
	return nil
}

// UnitValsTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *ModLayer) UnitValsTensor(tsr etensor.Tensor, varNm string) error {
	if tsr == nil {
		err := fmt.Errorf("leabra.UnitValsTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	vidx, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return ly.Layer.UnitValsTensor(tsr, varNm)
	}
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	for i := range ly.DeepNeurs {
		vl := ly.LeabraLay.(PBWMLayer).UnitValByIdx(NeuronVars(vidx), i)
		tsr.SetFloat1D(i, float64(vl))
	}
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *ModLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
	ly.ACh = 0
	ly.SE = 0
}

// DoQuarter2DWt indicates whether to do optional Q2 DWt
func (ly *ModLayer) DoQuarter2DWt() bool {
	return false
}

// QuarterFinal does updating after end of a quarter
func (ly *ModLayer) QuarterFinal(ltime *leabra.Time) {
	ly.Layer.QuarterFinal(ltime)
	if ltime.Quarter == 1 {
		ly.LeabraLay.(PBWMLayer).Quarter2DWt()
	}
}

// Quarter2DWt is optional Q2 DWt -- define where relevant
func (ly *ModLayer) Quarter2DWt() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		rly := p.RecvLay().(PBWMLayer)
		if rly.DoQuarter2DWt() {
			p.(leabra.LeabraPrjn).DWt()
		}
	}
}
