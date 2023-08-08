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
	NeuronVars = []string{"AlphaMax", "VmEff", "Gnmda", "NMDA", "NMDASyn", "GgabaB", "GABAB", "GABABx"}

	// NeuronVarsAll is the glong collection of all neuron-level vars
	NeuronVarsAll []string

	NeuronVarsMap map[string]int

	// NeuronVarProps are integrated neuron var props including leabra
	NeuronVarProps = map[string]string{
		"NMDA":   `auto-scale:"+"`,
		"GABAB":  `auto-scale:"+"`,
		"GABABx": `auto-scale:"+"`,
	}
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
	for v, p := range leabra.NeuronVarProps {
		NeuronVarProps[v] = p
	}
}

// Neuron holds the extra neuron (unit) level variables for glong computation.
type Neuron struct {

	// Maximum activation over Alpha cycle period
	AlphaMax float32 `desc:"Maximum activation over Alpha cycle period"`

	// Effective membrane potential, including simulated backpropagating action potential contribution from activity level.
	VmEff float32 `desc:"Effective membrane potential, including simulated backpropagating action potential contribution from activity level."`

	// net NMDA conductance, after Vm gating and Gbar -- added directly to Ge as it has the same reversal potential.
	Gnmda float32 `desc:"net NMDA conductance, after Vm gating and Gbar -- added directly to Ge as it has the same reversal potential."`

	// NMDA channel activation -- underlying time-integrated value with decay
	NMDA float32 `desc:"NMDA channel activation -- underlying time-integrated value with decay"`

	// synaptic NMDA activation directly from projection(s)
	NMDASyn float32 `desc:"synaptic NMDA activation directly from projection(s)"`

	// net GABA-B conductance, after Vm gating and Gbar + Gbase -- set to Gk for GIRK, with .1 reversal potential.
	GgabaB float32 `desc:"net GABA-B conductance, after Vm gating and Gbar + Gbase -- set to Gk for GIRK, with .1 reversal potential."`

	// GABA-B / GIRK activation -- time-integrated value with rise and decay time constants
	GABAB float32 `desc:"GABA-B / GIRK activation -- time-integrated value with rise and decay time constants"`

	// GABA-B / GIRK internal drive variable -- gets the raw activation and decays
	GABABx float32 `desc:"GABA-B / GIRK internal drive variable -- gets the raw activation and decays"`
}

func (nrn *Neuron) VarNames() []string {
	return NeuronVars
}

// NeuronVarIdxByName returns the index of the variable in the Neuron, or error
func NeuronVarIdxByName(varNm string) (int, error) {
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
	i, err := NeuronVarIdxByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}
