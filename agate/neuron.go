// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"fmt"
	"unsafe"

	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/pcore"
)

var (
	// NeuronVars are extra neuron variables for agate, on top of pcore
	NeuronVars = []string{"Grec", "GrecInc", "Gnmda", "VmEff"}

	// NeuronVarsAll is the agate collection of all neuron-level vars
	NeuronVarsAll []string

	// SynVarsAll is the agate collection of all synapse-level vars (includes TraceSynVars)
	SynVarsAll []string
)

func init() {
	dln := len(deep.NeuronVarsAll)
	pln := len(pcore.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+dln+pln)
	copy(NeuronVarsAll, deep.NeuronVarsAll)
	copy(NeuronVarsAll[dln:], pcore.NeuronVars)
	copy(NeuronVarsAll[dln+pln:], NeuronVars)

	MaintNeuronVarsMap = make(map[string]int, len(MaintNeuronVars))
	for i, v := range MaintNeuronVars {
		MaintNeuronVarsMap[v] = i
	}
}

//////////////////////////////////////////////////////////////////////
// Maint neurons

// MaintNeuron holds the extra neuron (unit) level variables for STN computation.
type MaintNeuron struct {
	Grec     float32 `desc:"recurrent-only (self) conductance"`
	GrecInc  float32 `desc:"increment for recurrent-only (self) conductance"`
	Gnmda    float32 `desc:"NMDA conductance, total -- added directly to Ge as it has the same reversal potential."`
	VmEff    float32 `desc:"Effective membrane potential, including simulated backpropagating action potential contribution from activity level."`
	AlphaMax float32 `desc:"Maximum activation over Alpha cycle period"`
}

var (
	MaintNeuronVars    = []string{"Grec", "GrecInc", "Gnmda", "VmEff", "AlphaMax"}
	MaintNeuronVarsMap map[string]int
)

func (nrn *MaintNeuron) VarNames() []string {
	return MaintNeuronVars
}

// MaintNeuronVarByName returns the index of the variable in the MaintNeuron, or error
func MaintNeuronVarByName(varNm string) (int, error) {
	i, ok := MaintNeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("MaintNeuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in MaintNeuronVars list)
func (nrn *MaintNeuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *MaintNeuron) VarByName(varNm string) (float32, error) {
	i, err := MaintNeuronVarByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}
