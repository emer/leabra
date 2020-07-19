// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/pcore"
)

var (
	// NeuronVars are extra neuron variables for agate and pcore
	NeuronVars = []string{"DA", "DALrn", "ACh", "Ca", "KCa"}

	// NeuronVarsAll is the agate collection of all neuron-level vars
	NeuronVarsAll []string

	// SynVarsAll is the agate collection of all synapse-level vars (includes TraceSynVars)
	SynVarsAll []string
)

func init() {
	ln := len(deep.NeuronVarsAll)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, deep.NeuronVarsAll)
	copy(NeuronVarsAll[ln:], NeuronVars)

	ln = len(leabra.SynapseVars)
	SynVarsAll = make([]string, len(pcore.TraceSynVars)+ln)
	copy(SynVarsAll, leabra.SynapseVars)
	copy(SynVarsAll[ln:], pcore.TraceSynVars)
}
