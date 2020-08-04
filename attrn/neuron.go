// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attrn

import (
	"github.com/emer/leabra/leabra"
)

var (
	// NeuronVars are extra neuron variables for attrn -- GeFwd is per-Neuron, Attn per Pool
	NeuronVars = []string{"GeFwd", "Attn"}

	// NeuronVarsAll is the attrn collection of all neuron-level vars
	NeuronVarsAll []string
)

func init() {
	ln := len(leabra.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, leabra.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)
}
