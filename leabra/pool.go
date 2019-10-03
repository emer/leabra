// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/fffb"
)

// Pool contains computed values for FFFB inhibition, and various other state values for layers
// and pools (unit groups) that can be subject to inhibition, including:
// * average / max stats on Ge and Act that drive inhibition
// * average activity overall that is used for normalizing netin (at layer level)
type Pool struct {
	StIdx, EdIdx int             `desc:"starting and ending (exlusive) indexes for the list of neurons in this pool"`
	Inhib        fffb.Inhib      `desc:"FFFB inhibition computed values, including Ge and Act AvgMax which drive inhibition"`
	ActM         minmax.AvgMax32 `desc:"minus phase average and max Act activation values, for ActAvg updt"`
	ActP         minmax.AvgMax32 `desc:"plus phase average and max Act activation values, for ActAvg updt"`
	ActAvg       ActAvg          `desc:"running-average activation levels used for netinput scaling and adaptive inhibition"`
}

func (pl *Pool) Init() {
	pl.Inhib.Init()
}

// ActAvg are running-average activation levels used for netinput scaling and adaptive inhibition
type ActAvg struct {
	ActMAvg    float32 `desc:"running-average minus-phase activity -- used for adapting inhibition -- see ActAvgParams.Tau for time constant etc"`
	ActPAvg    float32 `desc:"running-average plus-phase activity -- used for synaptic input scaling -- see ActAvgParams.Tau for time constant etc"`
	ActPAvgEff float32 `desc:"ActPAvg * ActAvgParams.Adjust -- adjusted effective layer activity directly used in synaptic input scaling"`
}
