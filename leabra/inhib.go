// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import "github.com/emer/leabra/v2/fffb"

// leabra.InhibParams contains all the inhibition computation params and functions for basic Leabra
// This is included in leabra.Layer to support computation.
// This also includes other misc layer-level params such as running-average activation in the layer
// which is used for netinput rescaling and potentially for adapting inhibition over time
type InhibParams struct {

	// inhibition across the entire layer
	Layer fffb.Params `display:"inline"`

	// inhibition across sub-pools of units, for layers with 4D shape
	Pool fffb.Params `display:"inline"`

	// neuron self-inhibition parameters -- can be beneficial for producing more graded, linear response -- not typically used in cortical networks
	Self SelfInhibParams `display:"inline"`

	// running-average activation computation values -- for overall estimates of layer activation levels, used in netinput scaling
	ActAvg ActAvgParams `display:"inline"`
}

func (ip *InhibParams) Update() {
	ip.Layer.Update()
	ip.Pool.Update()
	ip.Self.Update()
	ip.ActAvg.Update()
}

func (ip *InhibParams) Defaults() {
	ip.Layer.Defaults()
	ip.Pool.Defaults()
	ip.Self.Defaults()
	ip.ActAvg.Defaults()
}

///////////////////////////////////////////////////////////////////////
//  SelfInhibParams

// SelfInhibParams defines parameters for Neuron self-inhibition -- activation of the neuron directly feeds back
// to produce a proportional additional contribution to Gi
type SelfInhibParams struct {

	// enable neuron self-inhibition
	On bool

	// strength of individual neuron self feedback inhibition -- can produce proportional activation behavior in individual units for specialized cases (e.g., scalar val or BG units), but not so good for typical hidden layers
	Gi float32 `viewif:"On" def:"0.4"`

	// time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating unit self feedback inhibitory values -- prevents oscillations that otherwise occur -- relatively rapid 1.4 typically works, but may need to go longer if oscillations are a problem
	Tau float32 `viewif:"On" def:"1.4"`

	// rate = 1 / tau
	Dt float32 `edit:"-" display:"-" json:"-" xml:"-"`
}

func (si *SelfInhibParams) Update() {
	si.Dt = 1 / si.Tau
}

func (si *SelfInhibParams) Defaults() {
	si.On = false
	si.Gi = 0.4
	si.Tau = 1.4
	si.Update()
}

// Inhib updates the self inhibition value based on current unit activation
func (si *SelfInhibParams) Inhib(self *float32, act float32) {
	if si.On {
		*self += si.Dt * (si.Gi*act - *self)
	} else {
		*self = 0
	}
}

///////////////////////////////////////////////////////////////////////
//  ActAvgParams

// ActAvgParams represents expected average activity levels in the layer.
// Used for computing running-average computation that is then used for netinput scaling.
// Also specifies time constant for updating average
// and for the target value for adapting inhibition in inhib_adapt.
type ActAvgParams struct {

	// initial estimated average activity level in the layer (see also UseFirst option -- if that is off then it is used as a starting point for running average actual activity level, ActMAvg and ActPAvg) -- ActPAvg is used primarily for automatic netinput scaling, to balance out layers that have different activity levels -- thus it is important that init be relatively accurate -- good idea to update from recorded ActPAvg levels
	Init float32 `min:"0"`

	// if true, then the Init value is used as a constant for ActPAvgEff (the effective value used for netinput rescaling), instead of using the actual running average activation
	Fixed bool `def:"false"`

	// if true, then use the activation level computed from the external inputs to this layer (avg of targ or ext unit vars) -- this will only be applied to layers with Input or Target / Compare layer types, and falls back on the targ_init value if external inputs are not available or have a zero average -- implies fixed behavior
	UseExtAct bool `def:"false"`

	// use the first actual average value to override targ_init value -- actual value is likely to be a better estimate than our guess
	UseFirst bool `viewif:"Fixed=false" def:"true"`

	// time constant in trials for integrating time-average values at the layer level -- used for computing Pool.ActAvg.ActsMAvg, ActsPAvg
	Tau float32 `viewif:"Fixed=false" def:"100" min:"1"`

	// adjustment multiplier on the computed ActPAvg value that is used to compute ActPAvgEff, which is actually used for netinput rescaling -- if based on connectivity patterns or other factors the actual running-average value is resulting in netinputs that are too high or low, then this can be used to adjust the effective average activity value -- reducing the average activity with a factor < 1 will increase netinput scaling (stronger net inputs from layers that receive from this layer), and vice-versa for increasing (decreases net inputs)
	Adjust float32 `viewif:"Fixed=false" def:"1"`

	// rate = 1 / tau
	Dt float32 `edit:"-" display:"-" json:"-" xml:"-"`
}

func (aa *ActAvgParams) Update() {
	aa.Dt = 1 / aa.Tau
}

func (aa *ActAvgParams) Defaults() {
	aa.Init = 0.15
	aa.Fixed = false
	aa.UseExtAct = false
	aa.UseFirst = true
	aa.Tau = 100
	aa.Adjust = 1
	aa.Update()
}

// EffInit returns the initial value applied during InitWeights for the AvgPAvgEff effective layer activity
func (aa *ActAvgParams) EffInit() float32 {
	if aa.Fixed {
		return aa.Init
	}
	return aa.Adjust * aa.Init
}

// AvgFromAct updates the running-average activation given average activity level in layer
func (aa *ActAvgParams) AvgFromAct(avg *float32, act float32) {
	if act < 0.0001 {
		return
	}
	if aa.UseFirst && *avg == aa.Init {
		*avg += 0.5 * (act - *avg)
	} else {
		*avg += aa.Dt * (act - *avg)
	}
}

// EffFromAvg updates the effective value from the running-average value
func (aa *ActAvgParams) EffFromAvg(eff *float32, avg float32) {
	if aa.Fixed {
		*eff = aa.Init
	} else {
		*eff = aa.Adjust * avg
	}
}
