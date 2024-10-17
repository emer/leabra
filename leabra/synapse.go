// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"strings"
	"unsafe"

	"cogentcore.org/core/types"
)

// leabra.Synapse holds state for the synaptic connection between neurons
type Synapse struct {

	// synaptic weight value, sigmoid contrast-enhanced version
	// of the linear weight LWt.
	Wt float32

	// linear (underlying) weight value, which learns according
	// to the lrate specified in the connection spec.
	// This is converted into the effective weight value, Wt,
	// via sigmoidal contrast enhancement (see WtSigParams).
	LWt float32

	// change in synaptic weight, driven by learning algorithm.
	DWt float32

	// DWt normalization factor, reset to max of abs value of DWt,
	// decays slowly down over time. Serves as an estimate of variance
	// in weight changes over time.
	Norm float32

	// momentum, as time-integrated DWt changes, to accumulate a
	// consistent direction of weight change and cancel out
	// dithering contradictory changes.
	Moment float32

	// scaling parameter for this connection: effective weight value
	// is scaled by this factor in computing G conductance.
	// This is useful for topographic connectivity patterns e.g.,
	// to enforce more distant connections to always be lower in magnitude
	// than closer connections.  Value defaults to 1 (cannot be exactly 0,
	// otherwise is automatically reset to 1; use a very small number to
	// approximate 0). Typically set by using the paths.Pattern Weights()
	// values where appropriate.
	Scale float32

	// NTr is the new trace, which drives updates to trace value.
	// su * (1-ru_msn) for gated, or su * ru_msn for not-gated (or for non-thalamic cases).
	NTr float32

	// Tr is the current ongoing trace of activations, which drive learning.
	// Adds NTr and clears after learning on current values, and includes both
	// thal gated (+ and other nongated, - inputs).
	Tr float32
}

func (sy *Synapse) VarNames() []string {
	return SynapseVars
}

var SynapseVars = []string{"Wt", "LWt", "DWt", "Norm", "Moment", "Scale", "NTr", "Tr"}

var SynapseVarProps = map[string]string{
	"Wt":     `cat:"Wts"`,
	"LWt":    `cat:"Wts"`,
	"DWt":    `cat:"Wts" auto-scale:"+"`,
	"Norm":   `cat:"Wts"`,
	"Moment": `cat:"Wts" auto-scale:"+"`,
	"Scale":  `cat:"Wts"`,
	"NTr":    `cat:"Wts"`,
	"Tr":     `cat:"Wts"`,
}

var SynapseVarsMap map[string]int

func init() {
	SynapseVarsMap = make(map[string]int, len(SynapseVars))
	for i, v := range SynapseVars {
		SynapseVarsMap[v] = i
	}
	styp := types.For[Synapse]()
	for _, fld := range styp.Fields {
		tag := SynapseVarProps[fld.Name]
		SynapseVarProps[fld.Name] = tag + ` doc:"` + strings.ReplaceAll(fld.Doc, "\n", " ") + `"`
	}
}

// SynapseVarByName returns the index of the variable in the Synapse, or error
func SynapseVarByName(varNm string) (int, error) {
	i, ok := SynapseVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Synapse VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in SynapseVars list)
func (sy *Synapse) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(sy)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (sy *Synapse) VarByName(varNm string) (float32, error) {
	i, err := SynapseVarByName(varNm)
	if err != nil {
		return 0, err
	}
	return sy.VarByIndex(i), nil
}

func (sy *Synapse) SetVarByIndex(idx int, val float32) {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(sy)) + uintptr(4*idx)))
	*fv = val
}

// SetVarByName sets synapse variable to given value
func (sy *Synapse) SetVarByName(varNm string, val float32) error {
	i, err := SynapseVarByName(varNm)
	if err != nil {
		return err
	}
	sy.SetVarByIndex(i, val)
	return nil
}
