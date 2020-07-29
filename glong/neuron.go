// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"fmt"
	"unsafe"

	"github.com/emer/leabra/leabra"
)

var (
	// NeuronVars are extra neuron variables for glong
	NeuronVars = []string{"AlphaMax", "VmEff", "GnmdaP", "GnmdaPInc", "Gnmda", "Ggabab"}

	// NeuronVarsAll is the agate collection of all neuron-level vars
	NeuronVarsAll []string

	NeuronVarsMap map[string]int
)

func init() {
	ln := len(leabra.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, leabra.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)

	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	for i, v := range NeuronVars {
		NeuronVarsMap[v] = i
	}
}

//////////////////////////////////////////////////////////////////////
// Maint neurons

// Neuron holds the extra neuron (unit) level variables for STN computation.
type Neuron struct {
	AlphaMax  float32 `desc:"Maximum activation over Alpha cycle period"`
	VmEff     float32 `desc:"Effective membrane potential, including simulated backpropagating action potential contribution from activity level."`
	Gnmda     float32 `desc:"NMDA conductance, total -- added directly to Ge as it has the same reversal potential."`
	GnmdaP    float32 `desc:"raw NMDA conductance from projection(s)"`
	GnmdaPInc float32 `desc:"increment for prjn NMDA conductance"`
	Ggabab    float32 `desc:"GABA-B conductance, total -- added to Gk for GIRK, with .1 reversal potential."`
}

func (nrn *Neuron) VarNames() []string {
	return NeuronVars
}

// NeuronVarByName returns the index of the variable in the Neuron, or error
func NeuronVarByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("Neuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in NeuronVars list)
func (nrn *Neuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *Neuron) VarByName(varNm string) (float32, error) {
	i, err := NeuronVarByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}
