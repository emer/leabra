// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"log"
	"math/rand"

	"cogentcore.org/core/enums"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/erand"
)

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWeights initializes the weight values in the network,
// i.e., resetting learning Also calls InitActs.
func (ly *Layer) InitWeights() {
	ly.UpdateParams()
	for _, pt := range ly.SendPaths {
		if pt.Off {
			continue
		}
		pt.InitWeights()
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActAvg.ActMAvg = ly.Inhib.ActAvg.Init
		pl.ActAvg.ActPAvg = ly.Inhib.ActAvg.Init
		pl.ActAvg.ActPAvgEff = ly.Inhib.ActAvg.EffInit()
	}
	ly.InitActAvg()
	ly.InitActs()
	ly.CosDiff.Init()
}

// InitActAvg initializes the running-average activation
// values that drive learning.
func (ly *Layer) InitActAvg() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Learn.InitActAvg(nrn)
	}
}

// InitActs fully initializes activation state.
// only called automatically during InitWeights.
func (ly *Layer) InitActs() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Act.InitActs(nrn)
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Init()
		pl.ActM.Init()
		pl.ActP.Init()
	}
}

// InitWeightsSym initializes the weight symmetry.
// higher layers copy weights from lower layers.
func (ly *Layer) InitWtSym() {
	for _, pt := range ly.SendPaths {
		if pt.Off {
			continue
		}
		if !(pt.WtInit.Sym) {
			continue
		}
		// key ordering constraint on which way weights are copied
		if pt.Recv.Index < pt.Send.Index {
			continue
		}
		rpt, has := ly.RecipToSendPath(pt)
		if !has {
			continue
		}
		if !(rpt.WtInit.Sym) {
			continue
		}
		pt.InitWtSym(rpt)
	}
}

// InitExt initializes external input state -- called prior to apply ext
func (ly *Layer) InitExt() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Ext = 0
		nrn.Targ = 0
		nrn.SetFlag(false, NeurHasExt, NeurHasTarg, NeurHasCmpr)
	}
}

// ApplyExtFlags gets the flags that should cleared and set for updating neuron flags
// based on layer type, and whether input should be applied to Targ (else Ext)
func (ly *Layer) ApplyExtFlags() (clear, set []enums.BitFlag, toTarg bool) {
	clear = []enums.BitFlag{NeurHasExt, NeurHasTarg, NeurHasCmpr}
	toTarg = false
	if ly.Type == emer.Target {
		set = []enums.BitFlag{NeurHasTarg}
		toTarg = true
	} else if ly.Type == emer.Compare {
		set = []enums.BitFlag{NeurHasCmpr}
		toTarg = true
	} else {
		set = []enums.BitFlag{NeurHasExt}
	}
	return
}

// ApplyExt applies external input in the form of an tensor.Float32.  If
// dimensionality of tensor matches that of layer, and is 2D or 4D, then each dimension
// is iterated separately, so any mismatch preserves dimensional structure.
// Otherwise, the flat 1D view of the tensor is used.
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt(ext tensor.Tensor) {
	switch {
	case ext.NumDims() == 2 && ly.Shape.NumDims() == 4: // special case
		ly.ApplyExt2Dto4D(ext)
	case ext.NumDims() != ly.Shape.NumDims() || !(ext.NumDims() == 2 || ext.NumDims() == 4):
		ly.ApplyExt1DTsr(ext)
	case ext.NumDims() == 2:
		ly.ApplyExt2D(ext)
	case ext.NumDims() == 4:
		ly.ApplyExt4D(ext)
	}
}

// ApplyExt2D applies 2D tensor external input
func (ly *Layer) ApplyExt2D(ext tensor.Tensor) {
	clear, set, toTarg := ly.ApplyExtFlags()
	ymx := min(ext.Dim(0), ly.Shape.Dim(0))
	xmx := min(ext.Dim(1), ly.Shape.Dim(1))
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			vl := float32(ext.FloatValue(idx))
			i := ly.Shape.Offset(idx)
			nrn := &ly.Neurons[i]
			if nrn.Off {
				continue
			}
			if toTarg {
				nrn.Targ = vl
			} else {
				nrn.Ext = vl
			}
			nrn.SetFlag(false, clear...)
			nrn.SetFlag(true, set...)
		}
	}
}

// ApplyExt2Dto4D applies 2D tensor external input to a 4D layer
func (ly *Layer) ApplyExt2Dto4D(ext tensor.Tensor) {
	clear, set, toTarg := ly.ApplyExtFlags()
	lNy, lNx, _, _ := tensor.Path2DShape(&ly.Shape, false)

	ymx := min(ext.Dim(0), lNy)
	xmx := min(ext.Dim(1), lNx)
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			vl := float32(ext.FloatValue(idx))
			ui := tensor.Path2DIndex(&ly.Shape, false, y, x)
			nrn := &ly.Neurons[ui]
			if nrn.Off {
				continue
			}
			if toTarg {
				nrn.Targ = vl
			} else {
				nrn.Ext = vl
			}
			nrn.SetFlag(false, clear...)
			nrn.SetFlag(true, set...)
		}
	}
}

// ApplyExt4D applies 4D tensor external input
func (ly *Layer) ApplyExt4D(ext tensor.Tensor) {
	clear, set, toTarg := ly.ApplyExtFlags()
	ypmx := min(ext.Dim(0), ly.Shape.Dim(0))
	xpmx := min(ext.Dim(1), ly.Shape.Dim(1))
	ynmx := min(ext.Dim(2), ly.Shape.Dim(2))
	xnmx := min(ext.Dim(3), ly.Shape.Dim(3))
	for yp := 0; yp < ypmx; yp++ {
		for xp := 0; xp < xpmx; xp++ {
			for yn := 0; yn < ynmx; yn++ {
				for xn := 0; xn < xnmx; xn++ {
					idx := []int{yp, xp, yn, xn}
					vl := float32(ext.FloatValue(idx))
					i := ly.Shape.Offset(idx)
					nrn := &ly.Neurons[i]
					if nrn.Off {
						continue
					}
					if toTarg {
						nrn.Targ = vl
					} else {
						nrn.Ext = vl
					}
					nrn.SetFlag(false, clear...)
					nrn.SetFlag(true, set...)
				}
			}
		}
	}
}

// ApplyExt1DTsr applies external input using 1D flat interface into tensor.
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1DTsr(ext tensor.Tensor) {
	clear, set, toTarg := ly.ApplyExtFlags()
	mx := min(ext.Len(), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.Off {
			continue
		}
		vl := float32(ext.FloatValue1D(i))
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.SetFlag(false, clear...)
		nrn.SetFlag(true, set...)
	}
}

// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D(ext []float64) {
	clear, set, toTarg := ly.ApplyExtFlags()
	mx := min(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.Off {
			continue
		}
		vl := float32(ext[i])
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.SetFlag(false, clear...)
		nrn.SetFlag(true, set...)
	}
}

// ApplyExt1D32 applies external input in the form of
//
//	a flat 1-dimensional slice of float32s.
//
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D32(ext []float32) {
	clear, set, toTarg := ly.ApplyExtFlags()
	mx := min(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.Off {
			continue
		}
		vl := ext[i]
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.SetFlag(false, clear...)
		nrn.SetFlag(true, set...)
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (ly *Layer) UpdateExtFlags() {
	clear, set, _ := ly.ApplyExtFlags()
	for i := range ly.Neurons {
		nrn := &ly.Neurons[i]
		if nrn.Off {
			continue
		}
		nrn.SetFlag(false, clear...)
		nrn.SetFlag(true, set...)
	}
}

// ActAvgFmAct updates the running average ActMAvg, ActPAvg, and ActPAvgEff
// values from the current pool-level averages.
// The ActPAvgEff value is used for updating the conductance scaling parameters,
// if these are not set to Fixed, so calling this will change the scaling of
// pathways in the network!
func (ly *Layer) ActAvgFmAct() {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		ly.Inhib.ActAvg.AvgFmAct(&pl.ActAvg.ActMAvg, pl.ActM.Avg)
		ly.Inhib.ActAvg.AvgFmAct(&pl.ActAvg.ActPAvg, pl.ActP.Avg)
		ly.Inhib.ActAvg.EffFmAvg(&pl.ActAvg.ActPAvgEff, pl.ActAvg.ActPAvg)
	}
}

// ActQ0FmActP updates the neuron ActQ0 value from prior ActP value
func (ly *Layer) ActQ0FmActP() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		nrn.ActQ0 = nrn.ActP
	}
}

// AlphaCycInit handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// If updtActAvg is true, this includes updating the running-average
// activations for each layer / pool, and the AvgL running average used
// in BCM Hebbian learning.
// The input scaling is updated  based on the layer-level running average acts,
// and this can then change the behavior of the network,
// so if you want 100% repeatable testing results, set this to false to
// keep the existing scaling factors (e.g., can pass a train bool to
// only update during training).  This flag also affects the AvgL learning
// threshold
func (ly *Layer) AlphaCycInit(updtActAvg bool) {
	ly.ActQ0FmActP()
	if updtActAvg {
		ly.AvgLFmAvgM()
		ly.ActAvgFmAct()
	}
	ly.GScaleFmAvgAct() // need to do this always, in case hasn't been done at all yet
	if ly.Act.Noise.Type != NoNoise && ly.Act.Noise.Fixed && ly.Act.Noise.Dist != erand.Mean {
		ly.GenNoise()
	}
	ly.DecayState(ly.Act.Init.Decay)
	ly.InitGInc()
	if ly.Act.Clamp.Hard && ly.Type == emer.Input {
		ly.HardClamp()
	}
}

// AvgLFmAvgM updates AvgL long-term running average activation that drives BCM Hebbian learning
func (ly *Layer) AvgLFmAvgM() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ly.Learn.AvgLFmAvgM(nrn)
		if ly.Learn.AvgL.ErrMod {
			nrn.AvgLLrn *= ly.CosDiff.ModAvgLLrn
		}
	}
}

// GScaleFmAvgAct computes the scaling factor for synaptic input conductances G,
// based on sending layer average activation.
// This attempts to automatically adjust for overall differences in raw activity
// coming into the units to achieve a general target of around .5 to 1
// for the integrated Ge value.
func (ly *Layer) GScaleFmAvgAct() {
	totGeRel := float32(0)
	totGiRel := float32(0)
	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		slay := pt.Send.(LeabraLayer).AsLeabra()
		slpl := &slay.Pools[0]
		savg := slpl.ActAvg.ActPAvgEff
		snu := len(slay.Neurons)
		ncon := pt.RConNAvgMax.Avg
		pt.GScale = pt.WtScale.FullScale(savg, float32(snu), ncon)
		// reverting this change: if you want to eliminate a path, set the Off flag
		// if you want to negate it but keep the relative factor in the denominator
		// then set the scale to 0.
		// if pj.GScale == 0 {
		// 	continue
		// }
		if pt.Type == InhibPath {
			totGiRel += pt.WtScale.Rel
		} else {
			totGeRel += pt.WtScale.Rel
		}
	}

	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		if pt.Type == InhibPath {
			if totGiRel > 0 {
				pt.GScale /= totGiRel
			}
		} else {
			if totGeRel > 0 {
				pt.GScale /= totGeRel
			}
		}
	}
}

// GenNoise generates random noise for all neurons
func (ly *Layer) GenNoise() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Noise = float32(ly.Act.Noise.Gen(-1))
	}
}

// DecayState decays activation state by given proportion (default is on ly.Act.Init.Decay).
// This does *not* call InitGInc -- must call that separately at start of AlphaCyc
func (ly *Layer) DecayState(decay float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ly.Act.DecayState(nrn, decay)
	}
	for pi := range ly.Pools { // decaying average act is essential for inhib
		pl := &ly.Pools[pi]
		pl.Inhib.Decay(decay)
	}
}

// DecayStatePool decays activation state by given proportion
// in given sub-pool index (0 based).
func (ly *Layer) DecayStatePool(pool int, decay float32) {
	pi := int32(pool + 1) // 1 based
	pl := &ly.Pools[pi]
	for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ly.Act.DecayState(nrn, decay)
	}
	pl.Inhib.Decay(decay)
}

// HardClamp hard-clamps the activations in the layer.
// called during AlphaCycInit for hard-clamped Input layers.
func (ly *Layer) HardClamp() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ly.Act.HardClamp(nrn)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// InitGinc initializes the Ge excitatory and Gi inhibitory conductance accumulation states
// including ActSent and G*Raw values.
// called at start of trial always, and can be called optionally
// when delta-based Ge computation needs to be updated (e.g., weights
// might have changed strength)
func (ly *Layer) InitGInc() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ly.Act.InitGInc(nrn)
	}
	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		pt.InitGInc()
	}
}

// SendGDelta sends change in activation since last sent, to increment recv
// synaptic conductances G, if above thresholds
func (ly *Layer) SendGDelta(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		if nrn.Act > ly.Act.OptThresh.Send {
			delta := nrn.Act - nrn.ActSent
			if math32.Abs(delta) > ly.Act.OptThresh.Delta {
				for _, sp := range ly.SendPaths {
					if sp.Off {
						continue
					}
					sp.SendGDelta(ni, delta)
				}
				nrn.ActSent = nrn.Act
			}
		} else if nrn.ActSent > ly.Act.OptThresh.Send {
			delta := -nrn.ActSent // un-send the last above-threshold activation to get back to 0
			for _, sp := range ly.SendPaths {
				if sp.Off {
					continue
				}
				sp.SendGDelta(ni, delta)
			}
			nrn.ActSent = 0
		}
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *Layer) GFmInc(ltime *Time) {
	ly.RecvGInc(ltime)
	ly.GFmIncNeur(ltime)
}

// RecvGInc calls RecvGInc on receiving pathways to collect Neuron-level G*Inc values.
// This is called by GFmInc overall method, but separated out for cases that need to
// do something different.
func (ly *Layer) RecvGInc(ltime *Time) {
	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		pt.RecvGInc()
	}
}

// GFmIncNeur is the neuron-level code for GFmInc that integrates overall Ge, Gi values
// from their G*Raw accumulators.
func (ly *Layer) GFmIncNeur(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		// note: each step broken out here so other variants can add extra terms to Raw
		ly.Act.GeFmRaw(nrn, nrn.GeRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (ly *Layer) AvgMaxGe(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Ge.Init()
		for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.Off {
				continue
			}
			pl.Inhib.Ge.UpdateValue(nrn.Ge, int32(ni))
		}
		pl.Inhib.Ge.CalcAvg()
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) InhibFmGeAct(ltime *Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime)
	ly.InhibFmPool(ltime)
}

// PoolInhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) PoolInhibFmGeAct(ltime *Time) {
	np := len(ly.Pools)
	if np == 1 {
		return
	}
	lpl := &ly.Pools[0]
	lyInhib := ly.Inhib.Layer.On
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		ly.Inhib.Pool.Inhib(&pl.Inhib)
		if lyInhib {
			pl.Inhib.LayGi = lpl.Inhib.Gi
			pl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi) // pool is max of layer
		} else {
			lpl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi) // update layer from pool
		}
	}
	if !lyInhib {
		lpl.Inhib.GiOrig = lpl.Inhib.Gi // effective GiOrig
	}
}

// InhibFmPool computes inhibition Gi from Pool-level aggregated inhibition, including self and syn
func (ly *Layer) InhibFmPool(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
		nrn.Gi = pl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *Layer) ActFmG(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		ly.Learn.AvgsFmAct(nrn)
	}
}

// AvgMaxAct computes the average and max Act stats, used in inhibition
func (ly *Layer) AvgMaxAct(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Act.Init()
		for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.Off {
				continue
			}
			pl.Inhib.Act.UpdateValue(nrn.Act, int32(ni))
		}
		pl.Inhib.Act.CalcAvg()
	}
}

// CyclePost is called after the standard Cycle update, as a separate
// network layer loop.
// This is reserved for any kind of special ad-hoc types that
// need to do something special after Act is finally computed.
// For example, sending a neuromodulatory signal such as dopamine.
func (ly *Layer) CyclePost(ltime *Time) {
}

//////////////////////////////////////////////////////////////////////////////////////
//  Quarter

// QuarterFinal does updating after end of a quarter
func (ly *Layer) QuarterFinal(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		switch ltime.Quarter {
		case 2:
			pl.ActM = pl.Inhib.Act
		case 3:
			pl.ActP = pl.Inhib.Act
		}
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		switch ltime.Quarter {
		case 0:
			nrn.ActQ1 = nrn.Act
		case 1:
			nrn.ActQ2 = nrn.Act
		case 2:
			nrn.ActM = nrn.Act
			if nrn.HasFlag(NeurHasTarg) { // will be clamped in plus phase
				nrn.Ext = nrn.Targ
				nrn.SetFlag(true, NeurHasExt)
			}
		case 3:
			nrn.ActP = nrn.Act
			nrn.ActDif = nrn.ActP - nrn.ActM
			nrn.ActAvg += ly.Act.Dt.AvgDt * (nrn.Act - nrn.ActAvg)
		}
	}
	if ltime.Quarter == 3 {
		ly.CosDiffFmActs()
	}
}

// CosDiffFmActs computes the cosine difference in activation state between minus and plus phases.
// this is also used for modulating the amount of BCM hebbian learning
func (ly *Layer) CosDiffFmActs() {
	lpl := &ly.Pools[0]
	avgM := lpl.ActM.Avg
	avgP := lpl.ActP.Avg
	cosv := float32(0)
	ssm := float32(0)
	ssp := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		ap := nrn.ActP - avgP // zero mean
		am := nrn.ActM - avgM
		cosv += ap * am
		ssm += am * am
		ssp += ap * ap
	}

	dist := math32.Sqrt(ssm * ssp)
	if dist != 0 {
		cosv /= dist
	}
	ly.CosDiff.Cos = cosv

	ly.Learn.CosDiff.AvgVarFmCos(&ly.CosDiff.Avg, &ly.CosDiff.Var, ly.CosDiff.Cos)

	if ly.IsTarget() {
		ly.CosDiff.AvgLrn = 0 // no BCM for non-hidden layers
		ly.CosDiff.ModAvgLLrn = 0
	} else {
		ly.CosDiff.AvgLrn = 1 - ly.CosDiff.Avg
		ly.CosDiff.ModAvgLLrn = ly.Learn.AvgL.ErrModFmLayErr(ly.CosDiff.AvgLrn)
	}
}

// IsTarget returns true if this layer is a Target layer.
// By default, returns true for layers of Type == emer.Target
// Other Target layers include the TRCLayer in deep predictive learning.
// This is used for turning off BCM hebbian learning,
// in CosDiffFmActs to set the CosDiff.ModAvgLLrn value
// for error-modulated level of hebbian learning.
// It is also used in WtBal to not apply it to target layers.
// In both cases, Target layers are purely error-driven.
func (ly *Layer) IsTarget() bool {
	return ly.Type == emer.Target
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

// DWt computes the weight change (learning) -- calls DWt method on sending pathways
func (ly *Layer) DWt() {
	for _, pt := range ly.SendPaths {
		if pt.Off {
			continue
		}
		pt.DWt()
	}
}

// WtFmDWt updates the weights from delta-weight changes -- on the sending pathways
func (ly *Layer) WtFmDWt() {
	for _, pt := range ly.SendPaths {
		if pt.Off {
			continue
		}
		pt.WtFmDWt()
	}
}

// WtBalFmWt computes the Weight Balance factors based on average recv weights
func (ly *Layer) WtBalFmWt() {
	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		pt.WtBalFmWt()
	}
}

// LrateMult sets the new Lrate parameter for Paths to LrateInit * mult.
// Useful for implementing learning rate schedules.
func (ly *Layer) LrateMult(mult float32) {
	for _, pt := range ly.RecvPaths {
		// if p.Off { // keep all sync'd
		// 	continue
		// }
		pt.LrateMult(mult)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Threading / Reports

// CostEst returns the estimated computational cost associated with this layer,
// separated by neuron-level and synapse-level, in arbitrary units where
// cost per synapse is 1.  Neuron-level computation is more expensive but
// there are typically many fewer neurons, so in larger networks, synaptic
// costs tend to dominate.  Neuron cost is estimated from TimerReport output
// for large networks.
func (ly *Layer) CostEst() (neur, syn, tot int) {
	perNeur := 300 // cost per neuron, relative to synapse which is 1
	neur = len(ly.Neurons) * perNeur
	syn = 0
	for _, pt := range ly.SendPaths {
		ns := len(pt.Syns)
		syn += ns
	}
	tot = neur + syn
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Stats

// note: use float64 for stats as that is best for logging

// MSE returns the sum-squared-error and mean-squared-error
// over the layer, in terms of ActP - ActM (valid even on non-target layers FWIW).
// Uses the given tolerance per-unit to count an error at all
// (e.g., .5 = activity just has to be on the right side of .5).
func (ly *Layer) MSE(tol float32) (sse, mse float64) {
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0, 0
	}
	sse = 0.0
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.Off {
			continue
		}
		var d float32
		if ly.Type == emer.Compare {
			d = nrn.Targ - nrn.ActM
		} else {
			d = nrn.ActP - nrn.ActM
		}
		if math32.Abs(d) < tol {
			continue
		}
		sse += float64(d * d)
	}
	return sse, sse / float64(nn)
}

// SSE returns the sum-squared-error over the layer, in terms of ActP - ActM
// (valid even on non-target layers FWIW).
// Uses the given tolerance per-unit to count an error at all
// (e.g., .5 = activity just has to be on the right side of .5).
// Use this in Python which only allows single return values.
func (ly *Layer) SSE(tol float32) float64 {
	sse, _ := ly.MSE(tol)
	return sse
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion

// UnLesionNeurons unlesions (clears the Off flag) for all neurons in the layer
func (ly *Layer) UnLesionNeurons() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.SetFlag(false, NeurOff)
	}
}

// LesionNeurons lesions (sets the Off flag) for given proportion (0-1) of neurons in layer
// returns number of neurons lesioned.  Emits error if prop > 1 as indication that percent
// might have been passed
func (ly *Layer) LesionNeurons(prop float32) int {
	ly.UnLesionNeurons()
	if prop > 1 {
		log.Printf("LesionNeurons got a proportion > 1 -- must be 0-1 as *proportion* (not percent) of neurons to lesion: %v\n", prop)
		return 0
	}
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0
	}
	p := rand.Perm(nn)
	nl := int(prop * float32(nn))
	for i := 0; i < nl; i++ {
		nrn := &ly.Neurons[p[i]]
		nrn.SetFlag(true, NeurOff)
	}
	return nl
}

//////////////////////////////////////////////////////////////////////////////////////
//  Layer props for gui

// var LayerProps = tree.Props{
// "ToolBar": tree.PropSlice{
// 	{"Defaults", tree.Props{
// 		"icon": "reset",
// 		"desc": "return all parameters to their intial default values",
// 	}},
// 	{"InitWeights", tree.Props{
// 		"icon": "update",
// 		"desc": "initialize the layer's weight values according to path parameters, for all *sending* pathways out of this layer",
// 	}},
// 	{"InitActs", tree.Props{
// 		"icon": "update",
// 		"desc": "initialize the layer's activation values",
// 	}},
// 	{"sep-act", tree.BlankProp{}},
// 	{"LesionNeurons", tree.Props{
// 		"icon": "close",
// 		"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
// 		"Args": tree.PropSlice{
// 			{"Proportion", tree.Props{
// 				"desc": "proportion (0 -- 1) of neurons to lesion",
// 			}},
// 		},
// 	}},
// 	{"UnLesionNeurons", tree.Props{
// 		"icon": "reset",
// 		"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
// 	}},
// },
// }
