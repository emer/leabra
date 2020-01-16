// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/bitflag"
)

///////////////////////////////////////////////////////////////////////
//  params.go contains the DeepLeabra parameters and functions

// BurstParams are parameters determining how the DeepBurst activation is computed
// from the superficial layer activation values.
type BurstParams struct {
	On          bool            `desc:"Enable the computation of Burst from superficial activation state -- if this is off, then Burst is 0 and no Burst* projection signals are sent"`
	BurstQtr    leabra.Quarters `viewif:"On" desc:"Quarter(s) when bursting occurs -- typically Q4 but can also be Q2 and Q4 for beta-frequency updating.  Note: this is a bitflag and must be accessed using bitflag.Set / Has etc routines, 32 bit versions."`
	FmActNoAttn bool            `viewif:"On" desc:"Use the ActNoAttn activation state to compute Burst activation (otherwise Act) -- if DeepAttn attentional modulation is applied to Act, then this creates a positive feedback loop that can be problematic, so using the non-modulated activation value can be better."`
	ThrRel      float32         `viewif:"On" max:"1" def:"0.1,0.2,0.5" desc:"Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds."`
	ThrAbs      float32         `viewif:"On" min:"0" max:"1" def:"0.1,0.2,0.5" desc:"Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  Overall effective threshold is MAX of relative and absolute thresholds."`
}

func (db *BurstParams) Update() {
}

func (db *BurstParams) Defaults() {
	db.On = true
	db.SetBurstQtr(leabra.Q4)
	db.FmActNoAttn = true
	db.ThrRel = 0.1
	db.ThrAbs = 0.1
}

// SetBurstQtr sets given burst quarter (adds to any existing) -- Q4 by default
func (db *BurstParams) SetBurstQtr(qtr leabra.Quarters) {
	bitflag.Set32((*int32)(&db.BurstQtr), int(qtr))
}

// IsBurstQtr returns true if the given quarter (0-3) is set as a Bursting quarter.
func (db *BurstParams) IsBurstQtr(qtr int) bool {
	qmsk := leabra.Quarters(1 << uint(qtr))
	if db.BurstQtr&qmsk != 0 {
		return true
	}
	return false
}

// NextIsBurstQtr returns true if the quarter after given quarter (0-3)
// is set as a Bursting quarter according to BurstQtr settings.
// wraps around -- if qtr=3 and qtr=0 is a burst qtr, then it is true
func (db *BurstParams) NextIsBurstQtr(qtr int) bool {
	nqt := (qtr + 1) % 4
	return db.IsBurstQtr(nqt)
}

// PrevIsBurstQtr returns true if the quarter before given quarter (0-3)
// is set as a Bursting quarter according to BurstQtr settings.
// wraps around -- if qtr=0 and qtr=3 is a burst qtr, then it is true
func (db *BurstParams) PrevIsBurstQtr(qtr int) bool {
	pqt := (qtr - 1)
	if pqt < 0 {
		pqt += 4
	}
	return db.IsBurstQtr(pqt)
}

// TRCParams provides parameters for how the plus-phase (outcome) state of thalamic relay cell
// (e.g., Pulvinar) neurons is computed from the BurstTRC projections that drive TRCBurstGe
// excitatory conductance.
type TRCParams struct {
	MaxInhib  float32 `def:"0.2" min:"0.01" desc:"Level of pooled TRCBurstGe.Max at which the predictive non-burst inputs are fully inhibited (see InhibPool for option on what level of pooling this is computed over).  Computationally, it is essential that burst inputs inhibit effect of predictive non-burst (deep layer) inputs, so that the plus phase is not always just the minus phase plus something extra (the error will never go to zero then).  When max burst input exceeds this value, predictive non-burst inputs are fully suppressed.  If there is only weak burst input however, then the predictive inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning."`
	InhibPool bool    `desc:"For the MaxInhib mechanism, if this is true then the TRCBurstGe.Max value comes from the specific pool (if sub-pools exist in layer) -- otherwise it comes from the entire layer."`
	Binarize  bool    `desc:"Apply threshold to TRCBurstGe input for computing plus-phase activations -- above BinThr, then Act = BinOn, below = BinOff.  Typically used for one-to-one BurstTRC prjns with fixed wts = 1, so threshold is in terms of sending activation.  This is beneficial for layers with weaker graded activations, such as V1 or other perceptual inputs."`
	BinThr    float32 `viewif:"Binarize" desc:"Threshold for binarizing -- typically used for one-to-one BurstTRC prjns with fixed wts = 1, so threshold is in terms of sending activation"`
	BinOn     float32 `def:"0.3" viewif:"Binarize" desc:"Effective value for units above threshold -- lower value around 0.3 or so seems best."`
	BinOff    float32 `def:"0" viewif:"Binarize" desc:"Effective value for units below threshold -- typically 0."`
	//	POnlyM   bool    `desc:"TRC plus-phase for TRC units only occurs if the minus phase max activation for given unit group Pool is above .1 -- this reduces 'main effect' positive weight changes that can drive hogging."`
}

func (tp *TRCParams) Update() {
}

func (tp *TRCParams) Defaults() {
	tp.MaxInhib = 0.2
	tp.InhibPool = false
	tp.Binarize = false
	tp.BinThr = 0.4
	tp.BinOn = 0.3
	tp.BinOff = 0
	// tp.POnlyM = false
}

// BurstGe returns effective excitatory conductance to use for burst-quarter time in TRC layer.
func (tp *TRCParams) BurstGe(burstGe float32) float32 {
	if tp.Binarize {
		if burstGe >= tp.BinThr {
			return tp.BinOn
		} else {
			return tp.BinOff
		}
	} else {
		return burstGe
	}
}

// AttnParams are parameters determining how the DeepAttn and DeepLrn attentional modulation
// is computed from the AttnGe inputs received via DeepAttn projections
type AttnParams struct {
	On  bool    `desc:"Enable the computation of DeepAttn, DeepLrn from AttnGe (otherwise, DeepAttn and DeepLrn = 1)"`
	Min float32 `viewif:"On" min:"0" max:"1" def:"0.8" desc:"Minimum DeepAttn value, which can provide a non-zero baseline for attentional modulation (typical biological attentional modulation levels are roughly 30% or so at the neuron-level, e.g., in V4)"`
	Thr float32 `viewif:"On" min:"0" desc:"Threshold on AttnGe before DeepAttn is compute -- if not receiving even this amount of overall input from deep layer senders, then just set DeepAttn and DeepLrn to 1 for all neurons, as there isn't enough of a signal to go on yet"`

	Range float32 `view:"-" inactive:"+" desc:"Range = 1 - Min.  This is the range for the AttnGe to modulate value of DeepAttn, between Min and 1"`
}

func (db *AttnParams) Update() {
	db.Range = 1 - db.Min
}

func (db *AttnParams) Defaults() {
	db.On = false
	db.Min = 0.8
	db.Thr = 0.1
	db.Update()
}

// DeepLrnFmG returns the DeepLrn value computed from AttnGe and MAX(AttnGe) across layer.
// As simply the max-normalized value.
func (db *AttnParams) DeepLrnFmG(attnG, attnMax float32) float32 {
	return attnG / attnMax
}

// DeepAttnFmG returns the DeepAttn value computed from DeepLrn value
func (db *AttnParams) DeepAttnFmG(lrn float32) float32 {
	return db.Min + db.Range*lrn
}
