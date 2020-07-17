// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"github.com/chewxy/math32"
)

// TraceSyn holds extra synaptic state for trace projections
type TraceSyn struct {
	NTr float32 `desc:"new trace = send * recv -- drives updates to trace value: sn.ActLrn * rn.ActLrn (subject to derivative too)"`
	Tr  float32 `desc:" current ongoing trace of activations, which drive learning -- adds ntr and clears after ACh-modulated learning on current values"`
}

// VarByName returns synapse variable by name
func (sy *TraceSyn) VarByName(varNm string) float32 {
	switch varNm {
	case "NTr":
		return sy.NTr
	case "Tr":
		return sy.Tr
	}
	return math32.NaN()
}

// VarByIndex returns synapse variable by index
func (sy *TraceSyn) VarByIndex(varIdx int) float32 {
	switch varIdx {
	case 0:
		return sy.NTr
	case 1:
		return sy.Tr
	}
	return math32.NaN()
}

var TraceSynVars = []string{"NTr", "Tr"}
