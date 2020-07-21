// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"

	"github.com/emer/leabra/leabra"
)

type NeurVars int32

const (
	BurstVar NeurVars = iota

	BurstPrvVar

	CtxtGeVar
)

var (
	NeuronVars    = []string{"Burst", "BurstPrv", "CtxtGe"}
	NeuronVarsMap map[string]int
	NeuronVarsAll []string
)

func init() {
	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	for i, v := range NeuronVars {
		NeuronVarsMap[v] = i
	}
	ln := len(leabra.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, leabra.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)
}

// NeuronVarByName returns the index of the variable in the Neuron, or error
func NeuronVarByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Neuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}
