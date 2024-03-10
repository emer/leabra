// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"strconv"

	"cogentcore.org/core/mat32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

type MSNLayer struct {
	ModLayer

	// patch or matrix
	Compartment StriatalCompartment `inactive:"+"`

	// slice of delayed inhibition state for this layer.
	DIState []DelInhState

	DIParams DelayedInhibParams `view:"no-inline add-fields"`
}

type IMSNLayer interface {
	AsMSNLayer() *MSNLayer
}

func (ly *MSNLayer) AsMSNLayer() *MSNLayer {
	return ly
}

func (ly *MSNLayer) AsMod() *ModLayer {
	return &ly.ModLayer
}

// Parameters for Dorsal Striatum Medium Spiny Neuron computation
type MSNParams struct {

	// patch or matrix
	Compartment StriatalCompartment `inactive:"+"`
}

type StriatalCompartment int //enums:enum

const (
	PATCH StriatalCompartment = iota
	MATRIX
	NSComp
)

// Delayed inhibition for matrix compartment layers
type DelayedInhibParams struct {

	// add in a portion of inhibition from previous time period
	Active bool

	// proportion of per-unit net input on previous gamma-frequency quarter to add in as inhibition
	PrvQ float32

	// proportion of per-unit net input on previous trial to add in as inhibition
	PrvTrl float32
}

// Params for for trace-based learning
type MSNTraceParams struct {

	// use the sigmoid derivative factor 2 * act * (1-act) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)
	Deriv bool `def:"true"`

	// multiplier on trace activation for decaying prior traces -- new trace magnitude drives decay of prior trace -- if gating activation is low, then new trace can be low and decay is slow, so increasing this factor causes learning to be more targeted on recent gating changes
	Decay float32 `def:"1" min:"0"`

	// learning rate scale factor, if
	GateLRScale float32
}

// DelInhState contains extra variables for MSNLayer neurons -- stored separately
type DelInhState struct {

	// netin from previous quarter, used for delayed inhibition
	GePrvQ float32

	// netin from previous "trial" (alpha cycle), used for delayed inhibition
	GePrvTrl float32 `desc:"netin from previous \"trial\" (alpha cycle), used for delayed inhibition"`
}

func (ly *MSNLayer) GetMonitorVal(data []string) float64 {
	var val float32
	valType := data[0]
	unitIdx, _ := strconv.Atoi(data[1])
	switch valType {
	case "TotalAct":
		val = TotalAct(ly)
	case "Act":
		val = ly.Neurons[unitIdx].Act
	case "Inet":
		val = ly.Neurons[unitIdx].Inet
	case "ModAct":
		val = ly.ModNeurs[unitIdx].ModAct
	case "ModLevel":
		val = ly.ModNeurs[unitIdx].ModLevel
	case "PVAct":
		val = ly.ModNeurs[unitIdx].PVAct
	default:
		idx, err := ly.UnitVarIdx(valType)
		if err != nil {
			fmt.Printf("Unit value name \"%v\" unknown\n", valType)
			val = 0
		} else {
			val = ly.UnitVal1D(idx, unitIdx, 0)
		}
	}
	return float64(val)
}

// AddMatrixLayer adds a MSNLayer of given size, with given name.
// nY = number of pools in Y dimension, nX is pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func AddMSNLayer(nt *Network, name string, nY, nX, nNeurY, nNeurX int, cpmt StriatalCompartment, da DaRType) *MSNLayer {
	ly := &MSNLayer{}
	nt.AddLayerInit(ly, name, []int{nY, nX, nNeurY, nNeurX}, emer.Hidden)
	ly.ModLayer.Init()
	ly.DaMod.RecepType = da
	ly.Compartment = cpmt
	return ly
}

func (tp *MSNTraceParams) Defaults() {
	tp.Deriv = true
	tp.Decay = 1
	tp.GateLRScale = 0.7
}

// LrnFactor returns multiplicative factor for level of msn activation.  If Deriv
// is 2 * act * (1-act) -- the factor of 2 compensates for otherwise reduction in
// learning from these factors.  Otherwise is just act.
func (tp *MSNTraceParams) MSNActLrnFactor(act float32) float32 {
	if !tp.Deriv {
		return act
	}
	return 2 * act * (1 - act)
}

// Defaults in param.Sheet format
// Sel: "MSNLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.Gi":     "1.9",
// 		"Layer.Inhib.Layer.FB":     "0.5",
// 		"Layer.Inhib.Pool.On":      "true",
// 		"Layer.Inhib.Pool.Gi":      "1.9",
// 		"Layer.Inhib.Pool.FB":      "0",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.3",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.ActAvg.Init":  "0.2",
// 	}}

func (ly *MSNLayer) Defaults() {
	ly.ModLayer.Defaults()
	ly.DaMod.On = true
	ly.Inhib.Layer.Gi = 1.9
	ly.Inhib.Layer.FB = 0.5
	ly.Inhib.Pool.On = true
	ly.Inhib.Pool.Gi = 1.9
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.3
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 0.2
	ly.DIParams.Active = ly.Compartment == MATRIX
	if ly.DIParams.Active {
		ly.DIParams.PrvQ = 0
		ly.DIParams.PrvTrl = 6
	}
	for ni := range ly.DIState {
		dis := &ly.DIState[ni]
		mnr := &ly.ModNeurs[ni]
		dis.GePrvQ = 0
		dis.GePrvTrl = 0
		mnr.DA = 0
		mnr.ACh = 0
	}
}

func (ly *MSNLayer) GetDA() float32 {
	return ly.DA
}

func (ly *MSNLayer) SetDA(da float32) {
	for ni := range ly.ModNeurs {
		mnr := &ly.ModNeurs[ni]
		mnr.DA = da
	}
	ly.DA = da
}

func (ly *MSNLayer) QuarterInitPrvs(ltime *leabra.Time) {
	for ni := range ly.DIState {
		dis := &ly.DIState[ni]
		if ltime.Quarter == 0 {
			dis.GePrvQ = dis.GePrvTrl
		} else {
			nrn := &ly.Neurons[ni]
			dis.GePrvQ = nrn.Ge
		}
	}
}

func (ly *MSNLayer) ClearMSNTrace() {
	for pi := range ly.RcvPrjns {
		pj := ly.RcvPrjns[pi]
		mpj, ok := pj.(*MSNPrjn)
		if ok {
			mpj.ClearTrace()
		}
	}
}

// Build constructs the layer state, including calling Build on the projections
// you MUST have properly configured the Inhib.Pool.On setting by this point
// to properly allocate Pools for the unit groups if necessary.
func (ly *MSNLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.ModLayer.Build()
	ly.DIState = make([]DelInhState, len(ly.Neurons))
	return err
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *MSNLayer) InitActs() {
	ly.ModLayer.InitActs()
	for ni := range ly.Neurons {
		dis := &ly.DIState[ni]
		dis.GePrvQ = 0
		dis.GePrvTrl = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

func (ly *MSNLayer) PoolDelayedInhib(pl *leabra.Pool) {
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		dis := &ly.DIState[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
		nrn.Gi = pl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
		nrn.Gi += ly.DIParams.PrvTrl*dis.GePrvTrl + ly.DIParams.PrvQ*dis.GePrvQ
	}
}

func (ly *MSNLayer) ModsFmInc(_ *leabra.Time) {
	plMax := ly.ModPools[0].ModNetStats.Max
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		mnr := &ly.ModNeurs[ni]
		if ly.Compartment == MATRIX {
			if mnr.ModNet <= ly.ModNetThreshold {
				mnr.ModLrn = 0
				mnr.ModLevel = 1
			} else {
				newLrn := mnr.ModNet / plMax
				if mat32.IsInf(newLrn, 0) || mat32.IsNaN(newLrn) {
					mnr.ModLrn = 1
				} else {
					mnr.ModLrn = newLrn
				}
			}
		} else { // PATCH
			if mnr.ModNet <= ly.ModNetThreshold { // not enough yet
				mnr.ModLrn = 0 // default is 0
				if ly.ActModZero {
					mnr.ModLevel = 0
				} else {
					mnr.ModLevel = 1
				}
			} else {
				newLrn := mnr.ModNet / plMax
				if mat32.IsInf(newLrn, 1) || mat32.IsNaN(newLrn) {
					mnr.ModLrn = 1
				} else if mat32.IsInf(newLrn, -1) {
					mnr.ModLrn = -1
				} else {
					mnr.ModLrn = newLrn
				}
				mnr.ModLevel = 1 // do not modulate!
			}
		}
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
// this is here for matrix delyed inhibition, not needed otherwise
func (ly *MSNLayer) InhibFmGeAct(ltime *leabra.Time) {
	if ly.DIParams.Active {
		lpl := &ly.Pools[0]
		ly.Inhib.Layer.Inhib(&lpl.Inhib)
		np := len(ly.Pools)
		if np > 1 {
			for pi := 1; pi < np; pi++ {
				pl := &ly.Pools[pi]
				ly.Inhib.Pool.Inhib(&pl.Inhib)
				pl.Inhib.Gi = mat32.Max(pl.Inhib.Gi, lpl.Inhib.Gi)
				ly.PoolDelayedInhib(pl)
			}
		} else {
			ly.PoolDelayedInhib(lpl)
		}
	} else {
		ly.ModLayer.InhibFmGeAct(ltime)
	}
}

func (ly *MSNLayer) AlphaCycInit(updtActAvg bool) {
	if ly.DIParams.Active {
		for ni := range ly.DIState {
			dis := &ly.DIState[ni]
			nrn := &ly.Neurons[ni]
			dis.GePrvTrl = nrn.Ge
		}
	}
	ly.ModLayer.AlphaCycInit(updtActAvg)
}
