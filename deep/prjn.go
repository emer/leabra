// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// deep.Prjn is the DeepLeabra projection, based on basic rate-coded leabra.Prjn
type Prjn struct {
	leabra.Prjn             // access as .Prjn
	CtxtGeInc     []float32 `desc:"local per-recv unit accumulator for Ctxt excitatory conductance from sending units -- not a delta -- the full value"`
	TRCBurstGeInc []float32 `desc:"local per-recv unit increment accumulator for TRCBurstGe excitatory conductance from sending units -- this will be thread-safe"`
	AttnGeInc     []float32 `desc:"local per-recv unit increment accumulator for AttnGe excitatory conductance from sending units -- this will be thread-safe"`
}

var KiT_Prjn = kit.Types.AddType(&Prjn{}, PrjnProps)

func (pj *Prjn) Defaults() {
	pj.Prjn.Defaults()
}

func (pj *Prjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

func (pj *Prjn) PrjnTypeName() string {
	if pj.Typ < emer.PrjnTypeN {
		return pj.Typ.String()
	}
	ptyp := PrjnType(pj.Typ)
	ts := ptyp.String()
	sz := len(ts)
	if sz > 0 {
		return ts[:sz-1] // cut off trailing _
	}
	return ""
}

func (pj *Prjn) Build() error {
	err := pj.Prjn.Build()
	if err != nil {
		return err
	}
	rsh := pj.Recv.Shape()
	rlen := rsh.Len()
	pj.CtxtGeInc = make([]float32, rlen)
	pj.TRCBurstGeInc = make([]float32, rlen)
	pj.AttnGeInc = make([]float32, rlen)
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (pj *Prjn) InitGInc() {
	pj.Prjn.InitGInc()
	for ri := range pj.CtxtGeInc {
		pj.CtxtGeInc[ri] = 0
		pj.TRCBurstGeInc[ri] = 0
		pj.AttnGeInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendCtxtGe sends the full Burst activation from sending neuron index si,
// to integrate CtxtGe excitatory conductance on receivers
func (pj *Prjn) SendCtxtGe(si int, dburst float32) {
	scdb := dburst * pj.GScale
	nc := pj.SConN[si]
	st := pj.SConIdxSt[si]
	syns := pj.Syns[st : st+nc]
	scons := pj.SConIdx[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pj.CtxtGeInc[ri] += scdb * syns[ci].Wt
	}
}

// SendTRCBurstGeDelta sends the delta-Burst activation from sending neuron index si,
// to integrate TRCBurstGe excitatory conductance on receivers
func (pj *Prjn) SendTRCBurstGeDelta(si int, delta float32) {
	scdel := delta * pj.GScale
	nc := pj.SConN[si]
	st := pj.SConIdxSt[si]
	syns := pj.Syns[st : st+nc]
	scons := pj.SConIdx[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pj.TRCBurstGeInc[ri] += scdel * syns[ci].Wt
	}
}

// SendAttnGeDelta sends the delta-activation from sending neuron index si,
// to integrate into AttnGeInc excitatory conductance on receivers
func (pj *Prjn) SendAttnGeDelta(si int, delta float32) {
	scdel := delta * pj.GScale
	nc := pj.SConN[si]
	st := pj.SConIdxSt[si]
	syns := pj.Syns[st : st+nc]
	scons := pj.SConIdx[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pj.AttnGeInc[ri] += scdel * syns[ci].Wt
	}
}

// RecvCtxtGeInc increments the receiver's CtxtGe from that of all the projections
func (pj *Prjn) RecvCtxtGeInc() {
	rlay := pj.Recv.(DeepLayer).AsDeep()
	for ri := range rlay.DeepNeurs {
		rn := &rlay.DeepNeurs[ri]
		rn.CtxtGe += pj.CtxtGeInc[ri]
		pj.CtxtGeInc[ri] = 0
	}
}

// RecvTRCBurstGeInc increments the receiver's TRCBurstGe from that of all the projections
func (pj *Prjn) RecvTRCBurstGeInc() {
	rlay := pj.Recv.(DeepLayer).AsDeep()
	for ri := range rlay.DeepNeurs {
		rn := &rlay.DeepNeurs[ri]
		rn.TRCBurstGe += pj.TRCBurstGeInc[ri]
		pj.TRCBurstGeInc[ri] = 0
	}
}

// RecvAttnGeInc increments the receiver's AttnGe from that of all the projections
func (pj *Prjn) RecvAttnGeInc() {
	rlay := pj.Recv.(DeepLayer).AsDeep()
	for ri := range rlay.DeepNeurs {
		rn := &rlay.DeepNeurs[ri]
		rn.AttnGe += pj.AttnGeInc[ri]
		pj.AttnGeInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) -- on sending projections
// Deep version supports DeepCtxt temporal learning option
func (pj *Prjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	if pj.Typ == BurstCtxt {
		pj.LeabraPrj.(DeepPrjn).DWtDeepCtxt()
	} else {
		pj.Prjn.DWt()
	}
}

// DWtDeepCtxt computes the weight change (learning) -- for DeepCtxt projections
func (pj *Prjn) DWtDeepCtxt() {
	slay := pj.Send.(DeepLayer).AsDeep()
	rlay := pj.Recv.(DeepLayer).AsDeep()
	for si := range slay.Neurons {
		dsn := &slay.DeepNeurs[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			// following line should be ONLY diff: BurstPrv for *both* short and medium *sender*
			// activations, which are first two args:
			err, bcm := pj.Learn.CHLdWt(dsn.BurstPrv, dsn.BurstPrv, rn.AvgSLrn, rn.AvgM, rn.AvgL)

			bcm *= pj.Learn.XCal.LongLrate(rn.AvgLLrn)
			err *= pj.Learn.XCal.MLrn
			dwt := bcm + err
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

//////////////////////////////////////////////////////////////////////////////////////
//  PrjnType

// PrjnType has the DeepLeabra extensions to the emer.PrjnType types, for gui
type PrjnType emer.PrjnType

//go:generate stringer -type=PrjnType

var KiT_PrjnType = kit.Enums.AddEnumExt(emer.KiT_PrjnType, PrjnTypeN, kit.NotBitFlag, nil)

// The DeepLeabra prjn types
const (
	// BurstCtxt are projections from Superficial layers to Deep layers that
	// send Burst activations drive updating of DeepCtxt excitatory conductance,
	// at end of a DeepBurst quarter.  These projections also use a special learning
	// rule that takes into account the temporal delays in the activation states.
	BurstCtxt emer.PrjnType = emer.PrjnTypeN + iota

	// BurstTRC are projections from Superficial layers to TRC (thalamic relay cell)
	// neurons (e.g., in the Pulvinar) that send Burst activation continuously
	// during the DeepBurst quarter(s), driving the TRCBurstGe value, which then drives
	// the 	plus-phase activation state of the TRC representing the "outcome" against
	// which prior predictions are (implicitly) compared via the temporal difference
	// in TRC activation state.
	BurstTRC

	// DeepAttn are projections from Deep layers (representing layer 6 regular-spiking
	// CT corticothalamic neurons) up to corresponding Superficial layer neurons, that drive
	// the attentional modulation of activations there (i.e., DeepAttn and DeepLrn values).
	// This is sent continuously all the time from deep layers using the standard delta-based
	// Ge computation, and aggregated into the AttnGe variable on Super neurons.
	DeepAttn
)

// gui versions
const (
	BurstCtxt_ PrjnType = PrjnType(emer.PrjnTypeN) + iota
	BurstTRC_
	DeepAttn_
	PrjnTypeN
)

var PrjnProps = ki.Props{
	"EnumType:Typ": KiT_PrjnType,
}
