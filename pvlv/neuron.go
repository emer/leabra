// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"log"
	"unsafe"

	"github.com/emer/etable/v2/etensor"
	"github.com/emer/leabra/v2/leabra"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// UnitVarNames returns a list of variable names available on the units in this layer
// Mod returns *layer level* vars
func (ly *ModLayer) UnitVarNames() []string {
	return ModNeuronVarsAll
}

// NeuronVars are indexes into extra neuron-level variables
type ModNeuronVar int

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
	ModNeuronVarsN
)

var KiT_ModNeuronVar = kit.Enums.AddEnum(ModNeuronVarsN, kit.NotBitFlag, nil)

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

// NeuronVarIdxByName returns the index of the variable in the Neuron, or error
func NeuronVarIdxByName(varNm string) (int, error) {
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
	i, err = NeuronVarIdxByName(varNm)
	if err != nil {
		return mat32.NaN(), err
	}
	return mnr.VarByIndex(i), nil
}

// // UnitVals fills in values of given variable name on unit,
// // for each unit in the layer, into given float32 slice (only resized if not big enough).
// // Returns error on invalid var name.
func (ly *ModLayer) UnitVals(vals *[]float32, varNm string) error {
	vidx, ok := ModNeuronVarsMap[varNm]
	if !ok {
		return ly.Layer.UnitVals(vals, "Act")
	}
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	for i := 0; i < nn; i++ {
		(*vals)[i] = ly.LeabraLay.(IModLayer).AsMod().UnitValByIdx(ModNeuronVar(vidx), i)
	}
	return nil
}

// // UnitValsTensor returns values of given variable name on unit
// // for each unit in the layer, as a float32 tensor in same shape as layer units.
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
	for i := range ly.ModNeurs {
		vl := ly.LeabraLay.(*ModLayer).UnitValByIdx(ModNeuronVar(vidx), i)
		tsr.SetFloat1D(i, float64(vl))
	}
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *ModLayer) ModUnitVals(vals *[]float32, varNm string) error {

	vidx, err := NeuronVarIdxByName(varNm)
	if err != nil {
		return ly.ModUnitVals(vals, varNm)
	}
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	for i := range ly.Neurons {
		(*vals)[i] = ly.LeabraLay.UnitVal1D(vidx, i)
	}
	return nil
}
