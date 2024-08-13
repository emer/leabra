// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"cogentcore.org/core/math32"
)

// GABABParams control the GABAB dynamics in PFC Maint neurons, based on Brunel & Wang (2001)
// parameters.  We have to do some things to make it work for rate code neurons..
type GABABParams struct {

	// rise time for bi-exponential time dynamics of GABA-B
	RiseTau float32 `def:"45"`

	// decay time for bi-exponential time dynamics of GABA-B
	DecayTau float32 `def:"50"`

	// overall strength multiplier of GABA-B current
	Gbar float32 `def:"0.2"`

	// baseline level of GABA-B channels open independent of inhibitory input (is added to spiking-produced conductance)
	Gbase float32 `def:"0.2"`

	// multiplier for converting Gi from FFFB to GABA spikes
	Smult float32 `def:"15"`

	// time offset when peak conductance occurs, in msec, computed from RiseTau and DecayTau
	MaxTime float32 `edit:"-"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float32 `display:"-"`
}

func (gp *GABABParams) Defaults() {
	gp.RiseTau = 45
	gp.DecayTau = 50
	gp.Gbar = 0.2
	gp.Gbase = 0.2
	gp.Smult = 15
	gp.Update()
}

func (gp *GABABParams) Update() {
	gp.TauFact = math32.Pow(gp.DecayTau/gp.RiseTau, gp.RiseTau/(gp.DecayTau-gp.RiseTau))
	gp.MaxTime = ((gp.RiseTau * gp.DecayTau) / (gp.DecayTau - gp.RiseTau)) * math32.Log(gp.DecayTau/gp.RiseTau)
}

// GFmV returns the GABA-B conductance as a function of normalized membrane potential
func (gp *GABABParams) GFmV(v float32) float32 {
	vbio := math32.Max(v*100-100, -90) // critical to not go past -90
	return 1 / (1 + math32.FastExp(0.1*((vbio+90)+10)))
}

// GFmS returns the GABA-B conductance as a function of GABA spiking rate,
// based on normalized spiking factor (i.e., Gi from FFFB etc)
func (gp *GABABParams) GFmS(s float32) float32 {
	ss := s * gp.Smult // convert to spikes
	return 1 / (1 + math32.FastExp(-(ss-7.1)/1.4))
}

// BiExp computes bi-exponential update, returns dG and dX deltas to add to g and x
func (gp *GABABParams) BiExp(g, x float32) (dG, dX float32) {
	dG = (gp.TauFact*x - g) / gp.RiseTau
	dX = -x / gp.DecayTau
	return
}

// GABAB returns the updated GABA-B / GIRK activation and underlying x value
// based on current values and gi inhibitory conductance (proxy for GABA spikes)
func (gp *GABABParams) GABAB(gabaB, gabaBx, gi float32) (g, x float32) {
	dG, dX := gp.BiExp(gabaB, gabaBx)
	x = gabaBx + gp.GFmS(gi) + dX // gets new input
	g = gabaB + dG
	return
}

// GgabaB returns the overall net GABAB / GIRK conductance including
// Gbar, Gbase, and voltage-gating
func (gp *GABABParams) GgabaB(gabaB, vm float32) float32 {
	return gp.Gbar * gp.GFmV(vm) * (gabaB + gp.Gbase)
}
