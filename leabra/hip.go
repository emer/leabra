// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
)

// Contrastive Hebbian Learning (CHL) parameters
type CHLParams struct {

	// if true, use CHL learning instead of standard XCAL learning -- allows easy exploration of CHL vs. XCAL
	On bool

	// amount of hebbian learning (should be relatively small, can be effective at .0001)
	Hebb float32 `default:"0.001" min:"0" max:"1"`

	// amount of error driven learning, automatically computed to be 1-Hebb
	Err float32 `default:"0.999" min:"0" max:"1" edit:"-"`

	// if true, use ActQ1 as the minus phase -- otherwise ActM
	MinusQ1 bool

	// proportion of correction to apply to sending average activation for hebbian learning component (0=none, 1=all, .5=half, etc)
	SAvgCor float32 `default:"0.4:0.8" min:"0" max:"1"`

	// threshold of sending average activation below which learning does not occur (prevents learning when there is no input)
	SAvgThr float32 `default:"0.001" min:"0"`
}

func (ch *CHLParams) Defaults() {
	ch.On = true
	ch.Hebb = 0.001
	ch.SAvgCor = 0.4
	ch.SAvgThr = 0.001
	ch.Update()
}

func (ch *CHLParams) Update() {
	ch.Err = 1 - ch.Hebb
}

// MinusAct returns the minus-phase activation to use based on settings (ActM vs. ActQ1)
func (ch *CHLParams) MinusAct(actM, actQ1 float32) float32 {
	if ch.MinusQ1 {
		return actQ1
	}
	return actM
}

// HebbDWt computes the hebbian DWt value from sending, recv acts, savgCor, and linear Wt
func (ch *CHLParams) HebbDWt(sact, ract, savgCor, linWt float32) float32 {
	return ract * (sact*(savgCor-linWt) - (1-sact)*linWt)
}

// ErrDWt computes the error-driven DWt value from sending,
// recv acts in both phases, and linear Wt, which is used
// for soft weight bounding (always applied here, separate from hebbian
// which has its own soft weight bounding dynamic).
func (ch *CHLParams) ErrDWt(sactP, sactM, ractP, ractM, linWt float32) float32 {
	err := (ractP * sactP) - (ractM * sactM)
	if err > 0 {
		err *= (1 - linWt)
	} else {
		err *= linWt
	}
	return err
}

// DWt computes the overall dwt from hebbian and error terms
func (ch *CHLParams) DWt(hebb, err float32) float32 {
	return ch.Hebb*hebb + ch.Err*err
}

func (pt *Path) CHLDefaults() {
	pt.Learn.Norm.On = false     // off by default
	pt.Learn.Momentum.On = false // off by default
	pt.Learn.WtBal.On = false    // todo: experiment
}

// SAvgCor computes the sending average activation, corrected according to the SAvgCor
// correction factor (typically makes layer appear more sparse than it is)
func (pt *Path) SAvgCor(slay *Layer) float32 {
	savg := .5 + pt.CHL.SAvgCor*(slay.Pools[0].ActAvg.ActPAvgEff-0.5)
	savg = math32.Max(pt.CHL.SAvgThr, savg) // keep this computed value within bounds
	return 0.5 / savg
}

// DWtCHL computes the weight change (learning) for CHL
func (pt *Path) DWtCHL() {
	slay := pt.Send
	rlay := pt.Recv
	if slay.Pools[0].ActP.Avg < pt.CHL.SAvgThr { // inactive, no learn
		return
	}
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pt.SConN[si])
		st := int(pt.SConIndexSt[si])
		syns := pt.Syns[st : st+nc]
		scons := pt.SConIndex[st : st+nc]
		snActM := pt.CHL.MinusAct(sn.ActM, sn.ActQ1)

		savgCor := pt.SAvgCor(slay)

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			rnActM := pt.CHL.MinusAct(rn.ActM, rn.ActQ1)

			hebb := pt.CHL.HebbDWt(sn.ActP, rn.ActP, savgCor, sy.LWt)
			err := pt.CHL.ErrDWt(sn.ActP, snActM, rn.ActP, rnActM, sy.LWt)

			dwt := pt.CHL.DWt(hebb, err)
			norm := float32(1)
			if pt.Learn.Norm.On {
				norm = pt.Learn.Norm.NormFromAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pt.Learn.Momentum.On {
				dwt = norm * pt.Learn.Momentum.MomentFromDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pt.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pt.Learn.Norm.On {
			maxNorm := float32(0)
			for ci := range syns {
				sy := &syns[ci]
				if sy.Norm > maxNorm {
					maxNorm = sy.Norm
				}
			}
			for ci := range syns {
				sy := &syns[ci]
				sy.Norm = maxNorm
			}
		}
	}
}

func (pt *Path) EcCa1Defaults() {
	pt.Learn.Norm.On = false     // off by default
	pt.Learn.Momentum.On = false // off by default
	pt.Learn.WtBal.On = false    // todo: experiment
}

// DWt computes the weight change (learning) -- on sending pathways
// Delta version
func (pt *Path) DWtEcCa1() {
	slay := pt.Send
	rlay := pt.Recv
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pt.SConN[si])
		st := int(pt.SConIndexSt[si])
		syns := pt.Syns[st : st+nc]
		scons := pt.SConIndex[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]

			err := (sn.ActP * rn.ActP) - (sn.ActQ1 * rn.ActQ1)
			bcm := pt.Learn.BCMdWt(sn.AvgSLrn, rn.AvgSLrn, rn.AvgL)
			bcm *= pt.Learn.XCal.LongLrate(rn.AvgLLrn)
			err *= pt.Learn.XCal.MLrn
			dwt := bcm + err

			norm := float32(1)
			if pt.Learn.Norm.On {
				norm = pt.Learn.Norm.NormFromAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pt.Learn.Momentum.On {
				dwt = norm * pt.Learn.Momentum.MomentFromDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pt.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pt.Learn.Norm.On {
			maxNorm := float32(0)
			for ci := range syns {
				sy := &syns[ci]
				if sy.Norm > maxNorm {
					maxNorm = sy.Norm
				}
			}
			for ci := range syns {
				sy := &syns[ci]
				sy.Norm = maxNorm
			}
		}
	}
}

// ConfigLoopsHip configures the hippocampal looper and should be included in ConfigLoops
// in model to make sure hip loops is configured correctly.
// see hip.go for an instance of implementation of this function.
func (net *Network) ConfigLoopsHip(ctx *Context, ls *looper.Stacks) {
	var tmpValues []float32
	ecout := net.LayerByName("ECout")
	ecin := net.LayerByName("ECin")
	ca1 := net.LayerByName("CA1")
	ca3 := net.LayerByName("CA3")
	ca1FromECin := errors.Log1(ca1.RecvPathBySendName("ECin")).(*Path)
	ca1FromCa3 := errors.Log1(ca1.RecvPathBySendName("CA3")).(*Path)
	ca3FromDg := errors.Log1(ca3.RecvPathBySendName("DG")).(*Path)

	dgPjScale := ca3FromDg.WtScale.Rel

	ls.AddEventAllModes(etime.Cycle, "HipMinusPhase:Start", 0, func() {
		ca1FromECin.WtScale.Abs = 1
		ca1FromCa3.WtScale.Abs = 0
		ca3FromDg.WtScale.Rel = 0
		net.GScaleFromAvgAct()
		net.InitGInc()
	})
	ls.AddEventAllModes(etime.Cycle, "Hip:Quarter1", 25, func() {
		ca1FromECin.WtScale.Abs = 0
		ca1FromCa3.WtScale.Abs = 1
		if ctx.Mode == etime.Test {
			ca3FromDg.WtScale.Rel = 1 // weaker
		} else {
			ca3FromDg.WtScale.Rel = dgPjScale
		}
		net.GScaleFromAvgAct()
		net.InitGInc()
	})
	ls.AddEventAllModes(etime.Cycle, "HipPlusPhase:Start", 75, func() {
		ca1FromECin.WtScale.Abs = 1
		ca1FromCa3.WtScale.Abs = 0
		if ctx.Mode == etime.Train {
			ecin.UnitValues(&tmpValues, "Act", 0)
			ecout.ApplyExt1D32(tmpValues)
		}
		net.GScaleFromAvgAct()
		net.InitGInc()
	})
}
