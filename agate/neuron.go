// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/leabra/v2/deep"
	"github.com/emer/leabra/v2/glong"
	"github.com/emer/leabra/v2/pcore"
)

var (
	// NeuronVarsAll is the agate collection of all neuron-level vars (deep, glong, pcore)
	NeuronVarsAll []string
)

func init() {
	dln := len(deep.NeuronVarsAll)
	gln := len(glong.NeuronVars)
	pln := len(pcore.NeuronVars)
	NeuronVarsAll = make([]string, dln+gln+pln)
	copy(NeuronVarsAll, deep.NeuronVarsAll)
	copy(NeuronVarsAll[dln:], glong.NeuronVars)
	copy(NeuronVarsAll[dln+gln:], pcore.NeuronVars)
}
