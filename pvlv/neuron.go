// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"log"
	"unsafe"

	"cogentcore.org/core/mat32"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/leabra/v2/leabra"
)

// UnitVarNames returns a list of variable names available on the units in this layer
// Mod returns *layer level* vars
func (ly *ModLayer) UnitVarNames() []string {
	return ModNeuronVarsAll
}

// NeuronVars are indexes into extra neuron-level variables
type ModNeuronVar int //enums:enum

const (
	DA ModNeuronVar = iota
	ACh
	SE
	ModAct
	ModLevel
	ModNet
	ModLrn
	PVAct
	Cust1
)

var (
	// ModNeuronVars are the modulator neurons plus some custom variables that sub-types use for their
	// algo-specific cases -- need a consistent set of overall network-level vars for display / generic
	// interface.
	ModNeuronVars = []string{
		DA.String(), ACh.String(), SE.String(),
		ModAct.String(), ModLevel.String(), ModNet.String(), ModLrn.String(),
		PVAct.String(),
	}
	ModNeuronVarsMap map[string]int
	ModNeuronVarsAll []string
)

func init() {
	ModNeuronVarsMap = make(map[string]int, len(ModNeuronVars))
	for i, v := range ModNeuronVars {
		ModNeuronVarsMap[v] = i
	}
	ln := len(leabra.NeuronVars)
	ModNeuronVarsAll = make([]string, len(ModNeuronVars)+ln)
	copy(ModNeuronVarsAll, leabra.NeuronVars)
	copy(ModNeuronVarsAll[ln:], ModNeuronVars)
}

func (mnr *ModNeuron) VarNames() []string {
	return ModNeuronVars
}

// NeuronVarIndexByName returns the index of the variable in the Neuron, or error
func NeuronVarIndexByName(varNm string) (int, error) {
	i, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Neuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in NeuronVars list)
func (mnr *ModNeuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(mnr)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (mnr *ModNeuron) VarByName(varNm string) (float32, error) {
	var i int
	var err error
	i, err = NeuronVarIndexByName(varNm)
	if err != nil {
		return mat32.NaN(), err
	}
	return mnr.VarByIndex(i), nil
}

// // UnitValues fills in values of given variable name on unit,
// // for each unit in the layer, into given float32 slice (only resized if not big enough).
// // Returns error on invalid var name.
func (ly *ModLayer) UnitValues(vals *[]float32, varNm string, di int) error {
	vidx, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return ly.Layer.UnitValues(vals, "Act", di)
	}
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	for i := 0; i < nn; i++ {
		(*vals)[i] = ly.LeabraLay.(IModLayer).AsMod().UnitValueByIndex(ModNeuronVar(vidx), i)
	}
	return nil
}

// // UnitValuesTensor returns values of given variable name on unit
// // for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *ModLayer) UnitValuesTensor(tsr etensor.Tensor, varNm string, di int) error {
	if tsr == nil {
		err := fmt.Errorf("leabra.UnitValuesTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	vidx, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return ly.Layer.UnitValuesTensor(tsr, varNm, di)
	}
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	for i := range ly.ModNeurs {
		vl := ly.LeabraLay.(*ModLayer).UnitValueByIndex(ModNeuronVar(vidx), i)
		tsr.SetFloat1D(i, float64(vl))
	}
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *ModLayer) ModUnitValues(vals *[]float32, varNm string, di int) error {

	vidx, err := NeuronVarIndexByName(varNm)
	if err != nil {
		return ly.ModUnitValues(vals, varNm, di)
	}
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	for i := range ly.Neurons {
		(*vals)[i] = ly.LeabraLay.UnitVal1D(vidx, i, di)
	}
	return nil
}
