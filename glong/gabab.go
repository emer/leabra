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
	MaxTime  float32 `inactive:"+" desc:"time offset when peak conductance occurs, in msec, computed from RiseTau and DecayTau"`
	TauFact  float32 `view:"-" desc:"time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))"`
}

func (gp *GABABParams) Defaults() {
	gp.RiseTau = 45
	gp.DecayTau = 50
	gp.Gbar = 0.2
	gp.Gbase = 0.2
	gp.Smult = 10
	gp.Update()
}

func (gp *GABABParams) Update() {
	gp.TauFact = math32.Pow(gp.DecayTau/gp.RiseTau, gp.RiseTau/(gp.DecayTau-gp.RiseTau))
	gp.MaxTime = ((gp.RiseTau * gp.DecayTau) / (gp.DecayTau - gp.RiseTau)) * math32.Log(gp.DecayTau/gp.RiseTau)
}

// GFmV returns the GABA-B conductance as a function of normalized membrane potential
func (gp *GABABParams) GFmV(v float32) float32 {
	vbio := v*100 - 100
	return 1 / (1 + math32.Exp(0.1*((vbio+90)+10)))
}

// GFmS returns the GABA-B conductance as a function of GABA spiking rate,
// based on normalized spiking factor (i.e., Gi from FFFB etc)
func (gp *GABABParams) GFmS(s float32) float32 {
	ss := s * gp.Smult // convert to spikes
	return 1 / (1 + math32.Exp(-(ss-7.1)/1.4))
}

// BiExp computes bi-exponential update, returns dG and dD deltas to add to g and gD
func (gp *GABABParams) BiExp(g, gD float32) (dG, dD float32) {
	dG = (gp.TauFact*gD - g) / gp.RiseTau
	dD = -gD / gp.DecayTau
	return
}

// GgabaB returns the updated GABA-B conductance g and decay of g (d)
// based on Vm (VmEff), Gi (GABA spiking) and current GgabaB, GgabaBD.
func (gp *GABABParams) GgabaB(gGabaB, gGabaBD, gi, vm float32) (g, d float32) {
	ng := gp.Gbar * gp.GFmS(gi) * gp.GFmV(vm)
	dG, dD := gp.BiExp(gGabaB, gGabaBD)
	d += dD
	g = ng + dG
	return
}
