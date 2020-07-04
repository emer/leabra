// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// GPiLayer represents the GPi / SNr output nucleus of the BG.
// It gets inhibited by the MtxGo and GPeIn layers, and its minimum
// activation during this inhibition is measured for gating.
// Typically just a single unit per Pool representing a given stripe.
type GPiLayer struct {
	GPLayer
	GateThr float32   `def:"0.2" desc:"threshold on activation to count for gating"`
	MinAct  []float32 `desc:"per-pool minimum activation value during alpha cycle"`
}

var KiT_GPiLayer = kit.Types.AddType(&GPiLayer{}, leabra.LayerProps)

func (ly *GPiLayer) Defaults() {
	ly.GPLayer.Defaults()
	ly.GateThr = 0.2
}

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *GPiLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

// Build constructs the layer state, including calling Build on the projections
// you MUST have properly configured the Inhib.Pool.On setting by this point
// to properly allocate Pools for the unit groups if necessary.
func (ly *GPiLayer) Build() error {
	err := ly.GPLayer.Build()
	if err != nil {
		return err
	}
	np := len(ly.Pools)
	ly.MinAct = make([]float32, np)
	return nil
}

func (ly *GPiLayer) InitActs() {
	ly.GPLayer.InitActs()
}

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
// should already have presented the external input to the network at this point.
func (ly *GPiLayer) AlphaCycInit() {
	ly.GPLayer.AlphaCycInit()
	for pi := range ly.MinAct {
		ly.MinAct[pi] = 0
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act.
// GP extends to compute MinAct
func (ly *GPiLayer) ActFmG(ltime *leabra.Time) {
	ly.GPLayer.ActFmG(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		min := &ly.MinAct[nrn.SubPool]
		*min = math32.Min(*min, nrn.Act)
	}
}

//////////////////////////////////////////////////////////////////////
// GPiPrjn

// GPiPrjn must be used with GPi recv layer, from MtxGo, GPeIn senders.
// Learns from DA and gating status.
type GPiPrjn struct {
	leabra.Prjn
	GateLrate float32 `desc:"extra learning rate multiplier for gated pools"`
}

var KiT_GPiPrjn = kit.Types.AddType(&GPiPrjn{}, leabra.PrjnProps)

func (pj *GPiPrjn) Defaults() {
	pj.Prjn.Defaults()
	// no additional factors
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
	pj.Learn.Lrate = 0.01
	pj.GateLrate = 0.2

}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *GPiPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(*GPiLayer)

	da := rlay.DA

	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			mingpi := rlay.MinAct[rn.SubPool]

			// gpi is off when gating happens, MtxGo and GPeIn are both active for gating
			// if gated: if < da increase inhib, else decrease
			// if not gated: if < da decrease weights, else increase
			var dwt float32
			if mingpi < rlay.GateThr { // gated
				if da < 0 { // only learn on errors here
					dwt = -pj.GateLrate * da * (1 - mingpi) * sn.Act
				}
			} else { // didn't gate -- if turned out bad, try gating -- also needs bidir case
				dwt = da * mingpi * sn.Act
			}

			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFmAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pj.Learn.Momentum.On {
				dwt = norm * pj.Learn.Momentum.MomentFmDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pj.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pj.Learn.Norm.On {
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
