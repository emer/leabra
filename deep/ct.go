// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// CTLayer implements the corticothalamic projecting layer 6 deep neurons
// that project to the TRC pulvinar neurons, to generate the predictions.
// They receive phasic input representing 5IB bursting via CTCtxtPrjn inputs
// from SuperLayer and also from self projections.
type CTLayer struct {
	leabra.Layer                 // access as .Layer
	BurstQtr     leabra.Quarters `desc:"Quarter(s) when bursting occurs -- typically Q4 but can also be Q2 and Q4 for beta-frequency updating.  Note: this is a bitflag and must be accessed using its Set / Has etc routines, 32 bit versions."`
	CtxtGes      []float32       `desc:"slice of context (temporally delayed) excitatory conducances."`
}

var KiT_CTLayer = kit.Types.AddType(&CTLayer{}, LayerProps)

func (ly *CTLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Init.Decay = 0 // deep doesn't decay!
	ly.BurstQtr.Set(int(leabra.Q4))
	ly.Typ = CT
}

func (ly *CTLayer) Class() string {
	return "CT " + ly.Cls
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *CTLayer) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *CTLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = NeuronVarByName(varNm)
	if err != nil {
		return vidx, err
	}
	vidx += len(leabra.NeuronVars)
	return vidx, err
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *CTLayer) UnitVal1D(varIdx int, idx int) float32 {
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	if varIdx < 0 || varIdx >= len(NeuronVarsAll) {
		return math32.NaN()
	}
	nn := len(leabra.NeuronVars)
	if varIdx < nn {
		nrn := &ly.Neurons[idx]
		return nrn.VarByIndex(varIdx)
	}
	varIdx -= nn
	if varIdx == int(CtxtGeVar) {
		return ly.CtxtGes[idx]
	}
	return math32.NaN()
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *CTLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.CtxtGes = make([]float32, len(ly.Neurons))
	return nil
}

func (ly *CTLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.CtxtGes {
		ly.CtxtGes[ni] = 0
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *CTLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.GRawFmInc(nrn) // std integ other inputs

		geRaw := nrn.GeRaw + ly.CtxtGes[ni]
		ly.Act.GeFmRaw(nrn, geRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
}

// SendCtxtGe sends activation over CTCtxtPrjn projections to integrate
// CtxtGe excitatory conductance on CT layers.
// This must be called at the end of the Burst quarter for this layer.
// Satisfies the CtxtSender interface.
func (ly *CTLayer) SendCtxtGe(ltime *leabra.Time) {
	if !ly.BurstQtr.Has(ltime.Quarter) {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.Act > ly.Act.OptThresh.Send {
			for _, sp := range ly.SndPrjns {
				if sp.IsOff() {
					continue
				}
				ptyp := sp.Type()
				if ptyp != CTCtxt {
					continue
				}
				pj, ok := sp.(*CTCtxtPrjn)
				if !ok {
					continue
				}
				pj.SendCtxtGe(ni, nrn.Act)
			}
		}
	}
}

// CtxtFmGe integrates new CtxtGe excitatory conductance from projections, and computes
// overall Ctxt value, only on Deep layers.
// This must be called at the end of the DeepBurst quarter for this layer, after SendCtxtGe.
func (ly *CTLayer) CtxtFmGe(ltime *leabra.Time) {
	if !ly.BurstQtr.Has(ltime.Quarter) {
		return
	}
	for ni := range ly.CtxtGes {
		ly.CtxtGes[ni] = 0
	}
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		ptyp := p.Type()
		if ptyp != CTCtxt {
			continue
		}
		pj, ok := p.(*CTCtxtPrjn)
		if !ok {
			continue
		}
		pj.RecvCtxtGeInc()
	}
}
