// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/emer/etable/v2/minmax"
	"github.com/emer/leabra/v2/fffb"
)

// Pool contains computed values for FFFB inhibition, and various other state values for layers
// and pools (unit groups) that can be subject to inhibition, including:
// * average / max stats on Ge and Act that drive inhibition
// * average activity overall that is used for normalizing netin (at layer level)
type Pool struct {

	// starting and ending (exlusive) indexes for the list of neurons in this pool
	StIdx, EdIdx int

	// FFFB inhibition computed values, including Ge and Act AvgMax which drive inhibition
	Inhib fffb.Inhib

	// minus phase average and max Act activation values, for ActAvg updt
	ActM minmax.AvgMax32

	// plus phase average and max Act activation values, for ActAvg updt
	ActP minmax.AvgMax32

	// running-average activation levels used for netinput scaling and adaptive inhibition
	ActAvg ActAvg
}

func (pl *Pool) Init() {
	pl.Inhib.Init()
}

// ActAvg are running-average activation levels used for netinput scaling and adaptive inhibition
type ActAvg struct {

	// running-average minus-phase activity -- used for adapting inhibition -- see ActAvgParams.Tau for time constant etc
	ActMAvg float32

	// running-average plus-phase activity -- used for synaptic input scaling -- see ActAvgParams.Tau for time constant etc
	ActPAvg float32

	// ActPAvg * ActAvgParams.Adjust -- adjusted effective layer activity directly used in synaptic input scaling
	ActPAvgEff float32
}
