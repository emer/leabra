// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import "github.com/emer/leabra/v2/leabra"

// NeurVars are indexes into extra PBWM neuron-level variables
type NeurVars int

const (
	DA NeurVars = iota
	DALrn
	ACh
	SE
	GateAct
	GateNow
	GateCnt
	ActG
	NrnMaint
	MaintGe
	NeurVarsN
)

var (
	// NeuronVars are the pbwm neurons plus some custom variables that sub-types use for their
	// algo-specific cases -- need a consistent set of overall network-level vars for display / generic
	// interface.
	NeuronVars    = []string{"DA", "DALrn", "ACh", "SE", "GateAct", "GateNow", "GateCnt", "ActG", "Maint", "MaintGe"}
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
