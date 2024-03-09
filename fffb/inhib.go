// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fffb

import "github.com/emer/etable/v2/minmax"

// Inhib contains state values for computed FFFB inhibition
type Inhib struct {

	// computed feedforward inhibition
	FFi float32

	// computed feedback inhibition (total)
	FBi float32

	// overall value of the inhibition -- this is what is added into the unit Gi inhibition level (along with any synaptic unit-driven inhibition)
	Gi float32

	// original value of the inhibition (before pool or other effects)
	GiOrig float32

	// for pools, this is the layer-level inhibition that is MAX'd with the pool-level inhibition to produce the net inhibition
	LayGi float32

	// average and max Ge excitatory conductance values, which drive FF inhibition
	Ge minmax.AvgMax32

	// average and max Act activation values, which drive FB inhibition
	Act minmax.AvgMax32
}

func (fi *Inhib) Init() {
	fi.Zero()
	fi.Ge.Init()
	fi.Act.Init()
}

// Zero clears inhibition but does not affect Ge, Act averages
func (fi *Inhib) Zero() {
	fi.FFi = 0
	fi.FBi = 0
	fi.Gi = 0
	fi.GiOrig = 0
	fi.LayGi = 0
}

// Decay reduces inhibition values by given decay proportion
func (fi *Inhib) Decay(decay float32) {
	fi.Ge.Max -= decay * fi.Ge.Max
	fi.Ge.Avg -= decay * fi.Ge.Avg
	fi.Act.Max -= decay * fi.Act.Max
	fi.Act.Avg -= decay * fi.Act.Avg
	fi.FFi -= decay * fi.FFi
	fi.FBi -= decay * fi.FBi
	fi.Gi -= decay * fi.Gi
}

// Inhibs is a slice of Inhib records
type Inhibs []Inhib
