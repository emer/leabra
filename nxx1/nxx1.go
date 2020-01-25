// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package nxx1 provides the Noisy-X-over-X-plus-1 activation function that well-characterizes
the neural response function empirically, as a saturating sigmoid-like nonlinear response
with an initial largely-linear regime.

The basic x/(x+1) sigmoid function is convolved with a gaussian noise kernel to produce
a better approximation of the effects of noise on neural firing -- the main effect is
to create a continuous graded early level of firing even slightly below threshold, softening
the otherwise hard transition to firing at threshold.

A hand-optimized piece-wise function approximation is used to generate the NXX1 function
instead of requiring a lookup table of the gaussian convolution.  This is much easier
to use across a range of computational platforms including GPU's, and produces very similar
overall values.
*/
package nxx1

import "github.com/chewxy/math32"

// Params are the Noisy X/(X+1) rate-coded activation function parameters.
// This function well-characterizes the neural response function empirically,
// as a saturating sigmoid-like nonlinear response with an initial largely-linear regime.
// The basic x/(x+1) sigmoid function is convolved with a gaussian noise kernel to produce
// a better approximation of the effects of noise on neural firing -- the main effect is
// to create a continuous graded early level of firing even slightly below threshold, softening
// the otherwise hard transition to firing at threshold.
// A hand-optimized piece-wise function approximation is used to generate the NXX1 function
// instead of requiring a lookup table of the gaussian convolution.  This is much easier
// to use across a range of computational platforms including GPU's, and produces very similar
// overall values.  abc.
type Params struct {
	Thr          float32 `def:"0.5" desc:"threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization"`
	Gain         float32 `def:"80,100,40,20" min:"0" desc:"gain (gamma) of the rate-coded activation functions -- 100 is default, 80 works better for larger models, and 20 is closer to the actual spiking behavior of the AdEx model -- use lower values for more graded signals, generally in lower input/sensory layers of the network"`
	NVar         float32 `def:"0.005,0.01" min:"0" desc:"variance of the Gaussian noise kernel for convolving with XX1 in NOISY_XX1 and NOISY_LINEAR -- determines the level of curvature of the activation function near the threshold -- increase for more graded responding there -- note that this is not actual stochastic noise, just constant convolved gaussian smoothness to the activation function"`
	VmActThr     float32 `def:"0.01" desc:"threshold on activation below which the direct vm - act.thr is used -- this should be low -- once it gets active should use net - g_e_thr ge-linear dynamics (gelin)"`
	SigMult      float32 `def:"0.33" view:"-" json:"-" xml:"-" desc:"multiplier on sigmoid used for computing values for net < thr"`
	SigMultPow   float32 `def:"0.8" view:"-" json:"-" xml:"-" desc:"power for computing sig_mult_eff as function of gain * nvar"`
	SigGain      float32 `def:"3" view:"-" json:"-" xml:"-" desc:"gain multipler on (net - thr) for sigmoid used for computing values for net < thr"`
	InterpRange  float32 `def:"0.01" view:"-" json:"-" xml:"-" desc:"interpolation range above zero to use interpolation"`
	GainCorRange float32 `def:"10" view:"-" json:"-" xml:"-" desc:"range in units of nvar over which to apply gain correction to compensate for convolution"`
	GainCor      float32 `def:"0.1" view:"-" json:"-" xml:"-" desc:"gain correction multiplier -- how much to correct gains"`

	SigGainNVar float32 `view:"-" json:"-" xml:"-" desc:"sig_gain / nvar"`
	SigMultEff  float32 `view:"-" json:"-" xml:"-" desc:"overall multiplier on sigmoidal component for values below threshold = sig_mult * pow(gain * nvar, sig_mult_pow)"`
	SigValAt0   float32 `view:"-" json:"-" xml:"-" desc:"0.5 * sig_mult_eff -- used for interpolation portion"`
	InterpVal   float32 `view:"-" json:"-" xml:"-" desc:"function value at interp_range - sig_val_at_0 -- for interpolation"`
}

func (xp *Params) Update() {
	xp.SigGainNVar = xp.SigGain / xp.NVar
	xp.SigMultEff = xp.SigMult * math32.Pow(xp.Gain*xp.NVar, xp.SigMultPow)
	xp.SigValAt0 = 0.5 * xp.SigMultEff
	xp.InterpVal = xp.XX1GainCor(xp.InterpRange) - xp.SigValAt0
}

func (xp *Params) Defaults() {
	xp.Thr = 0.5
	xp.Gain = 100
	xp.NVar = 0.005
	xp.VmActThr = 0.01
	xp.SigMult = 0.33
	xp.SigMultPow = 0.8
	xp.SigGain = 3.0
	xp.InterpRange = 0.01
	xp.GainCorRange = 10.0
	xp.GainCor = 0.1
	xp.Update()
}

// XX1 computes the basic x/(x+1) function
func (xp *Params) XX1(x float32) float32 { return x / (x + 1) }

// XX1GainCor computes x/(x+1) with gain correction within GainCorRange
// to compensate for convolution effects
func (xp *Params) XX1GainCor(x float32) float32 {
	gainCorFact := (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
	if gainCorFact < 0 {
		return xp.XX1(xp.Gain * x)
	}
	newGain := xp.Gain * (1 - xp.GainCor*gainCorFact)
	return xp.XX1(newGain * x)
}

// NoisyXX1 computes the Noisy x/(x+1) function -- directly computes close approximation
// to x/(x+1) convolved with a gaussian noise function with variance nvar.
// No need for a lookup table -- very reasonable approximation for standard range of parameters
// (nvar = .01 or less -- higher values of nvar are less accurate with large gains,
// but ok for lower gains)
func (xp *Params) NoisyXX1(x float32) float32 {
	if x < 0 { // sigmoidal for < 0
		return xp.SigMultEff / (1 + math32.Exp(-(x * xp.SigGainNVar)))
	} else if x < xp.InterpRange {
		interp := 1 - ((xp.InterpRange - x) / xp.InterpRange)
		return xp.SigValAt0 + interp*xp.InterpVal
	} else {
		return xp.XX1GainCor(x)
	}
}

// X11GainCorGain computes x/(x+1) with gain correction within GainCorRange
// to compensate for convolution effects -- using external gain factor
func (xp *Params) XX1GainCorGain(x, gain float32) float32 {
	gainCorFact := (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
	if gainCorFact < 0 {
		return xp.XX1(gain * x)
	}
	newGain := gain * (1 - xp.GainCor*gainCorFact)
	return xp.XX1(newGain * x)
}

// NoisyXX1Gain computes the noisy x/(x+1) function -- directly computes close approximation
// to x/(x+1) convolved with a gaussian noise function with variance nvar.
// No need for a lookup table -- very reasonable approximation for standard range of parameters
// (nvar = .01 or less -- higher values of nvar are less accurate with large gains,
// but ok for lower gains).  Using external gain factor.
func (xp *Params) NoisyXX1Gain(x, gain float32) float32 {
	if x < xp.InterpRange {
		sigMultEffArg := xp.SigMult * math32.Pow(gain*xp.NVar, xp.SigMultPow)
		sigValAt0Arg := 0.5 * sigMultEffArg

		if x < 0 { // sigmoidal for < 0
			return sigMultEffArg / (1 + math32.Exp(-(x * xp.SigGainNVar)))
		} else { // else x < interp_range
			interp := 1 - ((xp.InterpRange - x) / xp.InterpRange)
			return sigValAt0Arg + interp*xp.InterpVal
		}
	} else {
		return xp.XX1GainCorGain(x, gain)
	}
}
