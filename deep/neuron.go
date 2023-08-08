// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"unsafe"

	"github.com/emer/leabra/leabra"
)

var (
	// NeuronVars are for full list across all deep Layer types
	NeuronVars = []string{"Burst", "BurstPrv", "Attn", "CtxtGe"}

	// SuperNeuronVars are for SuperLayer directly
	SuperNeuronVars = []string{"Burst", "BurstPrv", "Attn"}

	SuperNeuronVarsMap map[string]int

	// NeuronVarsAll is full integrated list across inherited layers and NeuronVars
	NeuronVarsAll []string
)

func init() {
	SuperNeuronVarsMap = make(map[string]int, len(SuperNeuronVars))
	for i, v := range SuperNeuronVars {
		SuperNeuronVarsMap[v] = i
	}
	ln := len(leabra.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, leabra.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)
}

// SuperNeuron has the neuron values for SuperLayer
type SuperNeuron struct {

	// 5IB bursting activation value, computed by thresholding regular activation
	Burst float32 `desc:"5IB bursting activation value, computed by thresholding regular activation"`

	// previous bursting activation -- used for context-based learning
	BurstPrv float32 `desc:"previous bursting activation -- used for context-based learning"`

	// attentional signal from TRC layer
	Attn float32 `desc:"attentional signal from TRC layer"`
}

// SuperNeuronVarIdxByName returns the index of the variable in the SuperNeuron, or error
func SuperNeuronVarIdxByName(varNm string) (int, error) {
	i, ok := SuperNeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("SuperNeuron VarIdxByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

func (sn *SuperNeuron) VarByIdx(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(sn)) + uintptr(4*idx)))
	return *fv
}
