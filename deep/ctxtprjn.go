// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"cogentcore.org/core/mat32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

// CtxtSender is an interface for layers that implement the SendCtxtGe method
// (SuperLayer, CTLayer)
type CtxtSender interface {
	leabra.LeabraLayer

	// SendCtxtGe sends activation over CTCtxtPrjn projections to integrate
	// CtxtGe excitatory conductance on CT layers.
	// This must be called at the end of the Burst quarter for this layer.
	SendCtxtGe(ltime *leabra.Time)
}

// CTCtxtPrjn is the "context" temporally-delayed projection into CTLayer,
// (corticothalamic deep layer 6) where the CtxtGe excitatory input
// is integrated only at end of Burst Quarter.
// Set FmSuper for the main projection from corresponding Super layer.
type CTCtxtPrjn struct {
	leabra.Prjn // access as .Prjn

	// if true, this is the projection from corresponding Superficial layer -- should be OneToOne prjn, with Learn.Learn = false, WtInit.Var = 0, Mean = 0.8 -- these defaults are set if FmSuper = true
	FmSuper bool

	// local per-recv unit accumulator for Ctxt excitatory conductance from sending units -- not a delta -- the full value
	CtxtGeInc []float32
}

func (pj *CTCtxtPrjn) Defaults() {
	pj.Prjn.Defaults()
	if pj.FmSuper {
		pj.Learn.Learn = false
		pj.WtInit.Mean = 0.5 // .5 better than .8 in several cases..
		pj.WtInit.Var = 0
	}
}

func (pj *CTCtxtPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

func (pj *CTCtxtPrjn) Type() emer.PrjnType {
	return CTCtxt
}

func (pj *CTCtxtPrjn) PrjnTypeName() string {
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

func (pj *CTCtxtPrjn) Build() error {
	err := pj.Prjn.Build()
	if err != nil {
		return err
	}
	rsh := pj.Recv.Shape()
	rlen := rsh.Len()
	pj.CtxtGeInc = make([]float32, rlen)
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (pj *CTCtxtPrjn) InitGInc() {
	pj.Prjn.InitGInc()
	for ri := range pj.CtxtGeInc {
		pj.CtxtGeInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendGDelta: disabled for this type
func (pj *CTCtxtPrjn) SendGDelta(si int, delta float32) {
}

// RecvGInc: disabled for this type
func (pj *CTCtxtPrjn) RecvGInc() {
}

// SendCtxtGe sends the full Burst activation from sending neuron index si,
// to integrate CtxtGe excitatory conductance on receivers
func (pj *CTCtxtPrjn) SendCtxtGe(si int, dburst float32) {
	scdb := dburst * pj.GScale
	nc := pj.SConN[si]
	st := pj.SConIndexSt[si]
	syns := pj.Syns[st : st+nc]
	scons := pj.SConIndex[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pj.CtxtGeInc[ri] += scdb * syns[ci].Wt
	}
}

// RecvCtxtGeInc increments the receiver's CtxtGe from that of all the projections
func (pj *CTCtxtPrjn) RecvCtxtGeInc() {
	rlay, ok := pj.Recv.(*CTLayer)
	if !ok {
		return
	}
	for ri := range rlay.CtxtGes {
		rlay.CtxtGes[ri] += pj.CtxtGeInc[ri]
		pj.CtxtGeInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) for Ctxt projections
func (pj *CTCtxtPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	sslay, issuper := pj.Send.(*SuperLayer)
	rlay := pj.Recv.(leabra.LeabraLayer).AsLeabra()
	for si := range slay.Neurons {
		sact := float32(0)
		if issuper {
			sact = sslay.SuperNeurs[si].BurstPrv
		} else {
			sact = slay.Neurons[si].ActQ0
		}
		nc := int(pj.SConN[si])
		st := int(pj.SConIndexSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIndex[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			// following line should be ONLY diff: sact for *both* short and medium *sender*
			// activations, which are first two args:
			err, bcm := pj.Learn.CHLdWt(sact, sact, rn.AvgSLrn, rn.AvgM, rn.AvgL)

			bcm *= pj.Learn.XCal.LongLrate(rn.AvgLLrn)
			err *= pj.Learn.XCal.MLrn
			dwt := bcm + err
			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFmAbsDWt(&sy.Norm, mat32.Abs(dwt))
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
type PrjnType emer.PrjnType //enums:enum

// The DeepLeabra prjn types
const (
	// CTCtxt are projections from Superficial layers to CT layers that
	// send Burst activations drive updating of CtxtGe excitatory conductance,
	// at end of a DeepBurst quarter.  These projections also use a special learning
	// rule that takes into account the temporal delays in the activation states.
	// Can also add self context from CT for deeper temporal context.
	CTCtxt emer.PrjnType = emer.PrjnTypeN + iota
)

// gui versions
const (
	CTCtxt_ PrjnType = PrjnType(emer.PrjnTypeN) + iota
)

// var PrjnProps = ki.Props{
// 	"EnumType:Typ": KiT_PrjnType,
// }
