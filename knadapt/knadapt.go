// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package knadapt provides code for sodium (Na) gated potassium (K) currents that drive
adaptation (accommodation) in neural firing.  As neurons spike, driving an influx of Na,
this activates the K channels, which, like leak channels, pull the membrane potential
back down toward rest (or even below).  Multiple different time constants have been
identified and this implementation supports 3:
M-type (fast), Slick (medium), and Slack (slow)

Here's a good reference:

Kaczmarek, L. K. (2013). Slack, Slick, and Sodium-Activated Potassium Channels.
ISRN Neuroscience, 2013. https://doi.org/10.1155/2013/354262

This package supports both spiking and rate-coded activations.
*/
package knadapt

//go:generate core generate -add-types

// Chan describes one channel type of sodium-gated adaptation, with a specific
// set of rate constants.
type Chan struct {

	// if On, use this component of K-Na adaptation
	On bool

	// Rise rate of fast time-scale adaptation as function of Na concentration -- directly multiplies -- 1/rise = tau for rise rate
	Rise float32

	// Maximum potential conductance of fast K channels -- divide nA biological value by 10 for the normalized units here
	Max float32

	// time constant in cycles for decay of adaptation, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life)
	Tau float32

	// 1/Tau rate constant
	Dt float32 `display:"-"`
}

func (ka *Chan) Defaults() {
	ka.On = true
	ka.Rise = 0.01
	ka.Max = 0.1
	ka.Tau = 100
	ka.Update()
}

func (ka *Chan) Update() {
	ka.Dt = 1 / ka.Tau
}

func (ka *Chan) ShouldDisplay(field string) bool {
	switch field {
	case "Rise", "Max", "Tau":
		return ka.On
	default:
		return true
	}
}

// GcFromSpike updates the KNa conductance based on spike or not
func (ka *Chan) GcFromSpike(gKNa *float32, spike bool) {
	if ka.On {
		if spike {
			*gKNa += ka.Rise * (ka.Max - *gKNa)
		} else {
			*gKNa -= ka.Dt * *gKNa
		}
	} else {
		*gKNa = 0
	}
}

// GcFromRate updates the KNa conductance based on rate-coded activation.
// act should already have the compensatory rate multiplier prior to calling.
func (ka *Chan) GcFromRate(gKNa *float32, act float32) {
	if ka.On {
		*gKNa += act*ka.Rise*(ka.Max-*gKNa) - (ka.Dt * *gKNa)
	} else {
		*gKNa = 0
	}
}

// Params describes sodium-gated potassium channel adaptation mechanism.
// Evidence supports at least 3 different time constants:
// M-type (fast), Slick (medium), and Slack (slow)
type Params struct {

	// if On, apply K-Na adaptation
	On bool

	// extra multiplier for rate-coded activations on rise factors -- adjust to match discrete spiking
	Rate float32 `default:"0.8"`

	// fast time-scale adaptation
	Fast Chan `display:"inline"`

	// medium time-scale adaptation
	Med Chan `display:"inline"`

	// slow time-scale adaptation
	Slow Chan `display:"inline"`
}

func (ka *Params) Defaults() {
	ka.Rate = 0.8
	ka.Fast.Defaults()
	ka.Med.Defaults()
	ka.Slow.Defaults()
	ka.Fast.Tau = 50
	ka.Fast.Rise = 0.05
	ka.Fast.Max = 0.1
	ka.Med.Tau = 200
	ka.Med.Rise = 0.02
	ka.Med.Max = 0.1
	ka.Slow.Tau = 1000
	ka.Slow.Rise = 0.001
	ka.Slow.Max = 1
	ka.Update()
}

func (ka *Params) Update() {
	ka.Fast.Update()
	ka.Med.Update()
	ka.Slow.Update()
}

func (ka *Params) ShouldDisplay(field string) bool {
	switch field {
	case "Rate":
		return ka.On
	default:
		return true
	}
}

// GcFromSpike updates all time scales of KNa adaptation from spiking
func (ka *Params) GcFromSpike(gKNaF, gKNaM, gKNaS *float32, spike bool) {
	ka.Fast.GcFromSpike(gKNaF, spike)
	ka.Med.GcFromSpike(gKNaM, spike)
	ka.Slow.GcFromSpike(gKNaS, spike)
}

// GcFromRate updates all time scales of KNa adaptation from rate code activation
func (ka *Params) GcFromRate(gKNaF, gKNaM, gKNaS *float32, act float32) {
	act *= ka.Rate
	ka.Fast.GcFromRate(gKNaF, act)
	ka.Med.GcFromRate(gKNaM, act)
	ka.Slow.GcFromRate(gKNaS, act)
}

/*
class STATE_CLASS(KNaAdaptMiscSpec) : public STATE_CLASS(SpecMemberBase) {
  // ##INLINE ##NO_TOKENS ##CAT_Leabra extra params associated with sodium-gated potassium channel adaptation mechanism
INHERITED(SpecMemberBase)
public:
  bool          clamp;          // #DEF_true apply adaptation even to clamped layers -- only happens if kna_adapt.on is true
  bool          invert_nd;      // #DEF_true invert the adaptation effect for the act_nd (non-depressed) value that is typically used for learning-drivng averages (avg_ss, _s, _m) -- only happens if kna_adapt.on is true
  float         max_gc;         // #CONDSHOW_ON_clamp||invert_nd #DEF_0.2 for clamp or invert_nd, maximum k_na conductance that we expect to get (prior to multiplying by g_bar.k) -- apply a proportional reduction in clamped activation and/or enhancement of act_nd based on current k_na conductance -- default is appropriate for default kna_adapt params
  float         max_adapt;      // #CONDSHOW_ON_clamp||invert_nd has opposite effects for clamp and invert_nd (and only operative when kna_adapt.on in addition): for clamp on clamped layers, this is the maximum amount of adaptation to apply to clamped activations when conductance is at max_gc -- biologically, values around .5 correspond generally to strong adaptation in primary visual cortex (V1) -- for invert_nd, this is the maximum amount of adaptation to invert, which is key for allowing learning to operate successfully despite the depression of activations due to adaptation -- values around .2 to .4 are good for g_bar.k = .2, depending on how strongly inputs are depressed -- need to experiment to find the best value for a given config
  bool          no_targ;        // #DEF_true automatically exclude units in TARGET layers and also TRC (Pulvinar) thalamic neurons from adaptation effects -- typically such layers should not be subject to these effects, so this makes it easier to not have to manually set those override params

  INLINE float Compute_Clamped(float clamp_act, float gc_kna_f, float gc_kna_m, float gc_kna_s) {
    float gc_kna = gc_kna_f + gc_kna_m + gc_kna_s;
    float pct_gc = fminf(gc_kna / max_gc, 1.0f);
    return clamp_act * (1.0f - pct_gc * max_adapt);
  }
  // apply adaptation directly to a clamped activation value, reducing in proportion to amount of k_na current

  INLINE float Compute_ActNd(float act, float gc_kna_f, float gc_kna_m, float gc_kna_s) {
    float gc_kna = gc_kna_f + gc_kna_m + gc_kna_s;
    float pct_gc = fminf(gc_kna / max_gc, 1.0f);
    return act * (1.0f + pct_gc * max_adapt);
  }
  // apply inverse of adaptation to activation value, increasing in proportion to amount of k_na current

  STATE_DECO_KEY("UnitSpec");
  STATE_TA_STD_CODE_SPEC(KNaAdaptMiscSpec);

  // STATE_UAE( UpdateDts(); );

private:
  void        Initialize()      { Defaults_init(); }
  void        Defaults_init() {
    clamp = true;  invert_nd = true;  max_gc = .2f;  max_adapt = 0.3f;  no_targ = true;
  }
};

*/
