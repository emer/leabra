// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package fffb provides feedforward (FF) and feedback (FB) inhibition (FFFB)
based on average (or maximum) excitatory netinput (FF) and activation (FB).

This produces a robust, graded k-Winners-Take-All dynamic of sparse
distributed representations having approximately k out of N neurons
active at any time, where k is typically 10-20 percent of N.
*/
package fffb

// Params parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on average (or maximum) netinput (FF) and activation (FB)
type Params struct {

	// enable this level of inhibition
	On bool

	// overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly
	Gi float32 `viewif:"On" min:"0" def:"1.8"`

	// overall inhibitory contribution from feedforward inhibition -- multiplies average netinput (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value
	FF float32 `viewif:"On" min:"0" def:"1"`

	// overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)
	FB float32 `viewif:"On" min:"0" def:"1"`

	// time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing
	FBTau float32 `viewif:"On" min:"0" def:"1.4,3,5"`

	// what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0
	MaxVsAvg float32 `viewif:"On" def:"0,0.5,1"`

	// feedforward zero point for average netinput -- below this level, no FF inhibition is computed based on avg netinput, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average netinputs are lower than typical, you may need to lower it
	FF0 float32 `viewif:"On" def:"0.1"`

	// rate = 1 / tau
	FBDt float32 `inactive:"+" view:"-" json:"-" xml:"-"`
}

func (fb *Params) Update() {
	fb.FBDt = 1 / fb.FBTau
}

func (fb *Params) Defaults() {
	fb.Gi = 1.8
	fb.FF = 1
	fb.FB = 1
	fb.FBTau = 1.4
	fb.MaxVsAvg = 0
	fb.FF0 = 0.1
	fb.Update()
}

// FFInhib returns the feedforward inhibition value based on average and max excitatory conductance within
// relevant scope
func (fb *Params) FFInhib(avgGe, maxGe float32) float32 {
	ffNetin := avgGe + fb.MaxVsAvg*(maxGe-avgGe)
	var ffi float32
	if ffNetin > fb.FF0 {
		ffi = fb.FF * (ffNetin - fb.FF0)
	}
	return ffi
}

// FBInhib computes feedback inhibition value as function of average activation
func (fb *Params) FBInhib(avgAct float32) float32 {
	fbi := fb.FB * avgAct
	return fbi
}

// FBUpdate updates feedback inhibition using time-integration rate constant
func (fb *Params) FBUpdate(fbi *float32, newFbi float32) {
	*fbi += fb.FBDt * (newFbi - *fbi)
}

// Inhib is full inhibition computation for given inhib state, which must have
// the Ge and Act values updated to reflect the current Avg and Max of those
// values in relevant inhibitory pool.
func (fb *Params) Inhib(inh *Inhib) {
	if !fb.On {
		inh.Zero()
		return
	}

	ffi := fb.FFInhib(inh.Ge.Avg, inh.Ge.Max)
	fbi := fb.FBInhib(inh.Act.Avg)

	inh.FFi = ffi
	fb.FBUpdate(&inh.FBi, fbi)

	inh.Gi = fb.Gi * (ffi + inh.FBi)
	inh.GiOrig = inh.Gi
}
