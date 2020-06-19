// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

// PFC dynamic behavior element -- defines the dynamic behavior of deep layer PFC units
type PFCDyn struct {
	Init     float32 `desc:"initial value at point when gating starts -- MUST be > 0 when used."`
	RiseTau  float32 `desc:"time constant for linear rise in maintenance activation (per quarter when deep is updated) -- use integers -- if both rise and decay then rise comes first"`
	DecayTau float32 `desc:"time constant for linear decay in maintenance activation (per quarter when deep is updated) -- use integers -- if both rise and decay then rise comes first"`
	Desc     string  `desc:"description of this factor"`
}

func (pd *PFCDyn) Defaults() {
	pd.Init = 1
}

func (pd *PFCDyn) Set(init, rise, decay float32, desc string) {
	pd.Init = init
	pd.RiseTau = rise
	pd.DecayTau = decay
	pd.Desc = desc
}

// Value returns dynamic value at given time point
func (pd *PFCDyn) Value(time float32) float32 {
	val := pd.Init
	if time <= 0 {
		return val
	}
	if pd.RiseTau > 0 && pd.DecayTau > 0 {
		if time >= pd.RiseTau {
			val = 1 - ((time - pd.RiseTau) / pd.DecayTau)
		} else {
			val = pd.Init + (1-pd.Init)*(time/pd.RiseTau)
		}
	} else if pd.RiseTau > 0 {
		val = pd.Init + (1-pd.Init)*(time/pd.RiseTau)
	} else if pd.DecayTau > 0 {
		val = pd.Init - pd.Init*(time/pd.DecayTau)
	}
	if val > 1 {
		val = 1
	}
	if val < 0.001 {
		val = 0.001
	}
	return val
}

//////////////////////////////////////////////////////////////////////////////
//  PFCDyns

// PFCDyns is a slice of dyns. Provides deterministic control over PFC
// maintenance dynamics -- the rows of PFC units (along Y axis) behave
// according to corresponding index of Dyns.
// ensure layer Y dim has even multiple of len(Dyns).
type PFCDyns []*PFCDyn

// SetDyn sets given dynamic maint element to given parameters (must be allocated in list first)
func (pd *PFCDyns) SetDyn(dyn int, init, rise, decay float32, desc string) *PFCDyn {
	dy := &PFCDyn{}
	dy.Set(init, rise, decay, desc)
	(*pd)[dyn] = dy
	return dy
}

// MaintOnly creates basic default maintenance dynamic configuration -- every
// unit just maintains over time.
// This should be used for Output gating layer.
func (pd *PFCDyns) MaintOnly() {
	*pd = make([]*PFCDyn, 1)
	pd.SetDyn(0, 1, 0, 0, "maintain stable act")
}

// FullDyn creates full dynamic Dyn configuration, with 5 different
// dynamic profiles: stable maint, phasic, rising maint, decaying maint,
// and up / down maint.  tau is the rise / decay base time constant.
func (pd *PFCDyns) FullDyn(tau float32) {
	ndyn := 5
	*pd = make([]*PFCDyn, ndyn)

	pd.SetDyn(0, 1, 0, 0, "maintain stable act")
	pd.SetDyn(1, 1, 0, 1, "immediate phasic response")
	pd.SetDyn(2, .1, tau, 0, "maintained, rising value over time")
	pd.SetDyn(3, 1, 0, tau, "maintained, decaying value over time")
	pd.SetDyn(4, .1, .5*tau, tau, "maintained, rising then falling over time")
}

// Value returns value for given dyn item at given time step
func (pd *PFCDyns) Value(dyn int, time float32) float32 {
	sz := len(*pd)
	if sz == 0 {
		return 1
	}
	dy := (*pd)[dyn%sz]
	return dy.Value(time)
}
