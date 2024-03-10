// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"fmt"
	"unsafe"

	"github.com/emer/leabra/v2/leabra"
)

var (
	// NeuronVars are extra neuron variables for pcore
	NeuronVars = []string{"DA", "DALrn", "ACh", "Ca", "KCa"}

	// NeuronVarsAll is the pcore collection of all neuron-level vars
	NeuronVarsAll []string

	// SynVarsAll is the pcore collection of all synapse-level vars (includes TraceSynVars)
	SynVarsAll []string
)

func init() {
	ln := len(leabra.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, leabra.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)

	ln = len(leabra.SynapseVars)
	SynVarsAll = make([]string, len(TraceSynVars)+ln)
	copy(SynVarsAll, leabra.SynapseVars)
	copy(SynVarsAll[ln:], TraceSynVars)
}

//////////////////////////////////////////////////////////////////////
// STN neurons

// STNNeuron holds the extra neuron (unit) level variables for STN computation.
type STNNeuron struct {

	// intracellular Calcium concentration -- increased by bursting and elevated levels of activation, drives KCa currents that result in hyperpolarization / inhibition.
	Ca float32

	// Calcium-gated potassium channel conductance level, computed using function from gillies & Willshaw 2006 as function of Ca.
	KCa float32
}

var (
	STNNeuronVars    = []string{"Ca", "KCa"}
	STNNeuronVarsMap map[string]int
)

func init() {
	STNNeuronVarsMap = make(map[string]int, len(STNNeuronVars))
	for i, v := range STNNeuronVars {
		STNNeuronVarsMap[v] = i
	}
}

func (nrn *STNNeuron) VarNames() []string {
	return STNNeuronVars
}

// STNNeuronVarIdxByName returns the index of the variable in the STNNeuron, or error
func STNNeuronVarIdxByName(varNm string) (int, error) {
	i, ok := STNNeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("STNNeuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in STNNeuronVars list)
func (nrn *STNNeuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *STNNeuron) VarByName(varNm string) (float32, error) {
	i, err := STNNeuronVarIdxByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}
