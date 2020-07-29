// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"github.com/chewxy/math32"
)

// GABABParams control the GABAB dynamics in PFC Maint neurons, based on Brunel & Wang (2001)
// parameters.  We have to do some things to make it work for rate code neurons..
type GABABParams struct {
	RiseTau  float32 `def:"45" desc:"rise time for bi-exponential time dynamics of GABA-B"`
	DecayTau float32 `def:"50" desc:"decay time for bi-exponential time dynamics of GABA-B"`
	Gbar     float32 `def:"0.2" desc:"overall strength multiplier of GABA-B current"`
	Gbase    float32 `def:"0.2" desc:"baseline level of GABA-B channels open independent of inhibitory input (is added to spiking-produced conductance)"`
	Smult    float32 `def:"10" desc:"multiplier for converting Gi from FFFB to GABA spikes"`
}

func (np *GABABParams) Defaults() {
	np.RiseTau = 45
	np.DecayTau = 50
	np.Gbar = 0.2
	np.Gbase = 0.2
	np.Smult = 10
}

// GFmV returns the GABAB conductance as a function of normalized membrane potential
func (np *GABABParams) GFmV(v float32) float32 {
	vbio := v*100 - 100
	return 1 / (1 + math32.Exp(0.1*((vbio+90)+10)))
}

// GFmS returns the GABAB conductance as a function of GABA spiking rate,
// based on normalized spiking factor (i.e., Gi from FFFB etc)
func (np *GABABParams) GFmS(s float32) float32 {
	ss := s * np.Smult // convert to spikes
	return 1 / (1 + math32.Exp(-(ss-7.1)/1.4))
}
