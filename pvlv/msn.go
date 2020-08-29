// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"strconv"
)

type MSNLayer struct {
	ModLayer
	MSN      MSNParams          `view:"inline" desc:"striatal layer parameters"`
	MSNNeurs []MSNeuron         `desc:"slice of MSNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values."`
	DelInh   DelayedInhibParams `view:"no-inline add-fields"`
}

var KiT_MSNLayer = kit.Types.AddType(&MSNLayer{}, leabra.LayerProps)

type IMSNLayer interface {
	//IModLayer
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
	Compartment StriatalCompartment `inactive:"+" desc:"patch or matrix"`
	PatchShunt  float32             `def:"0.2,0.5" min:"0" max:"1" desc:"how much the patch shunt activation multiplies the dopamine values -- 0 = complete shunting, 1 = no shunting -- should be a factor < 1.0"`
	ShuntACh    bool                `def:"true" desc:"also shunt the ACh value driven from TAN units -- this prevents clearing of MSNConSpec traces -- more plausibly the patch units directly interfere with the effects of TAN's rather than through ach, but it is easier to implement with ach shunting here."`
	OutAChInhib float32             `def:"0,0.3" desc:"how much does the LACK of ACh from the TAN units drive extra inhibition to output-gating MSN units -- gi += out_ach_inhib * (1-ach) -- provides a bias for output gating on reward trials -- do NOT apply to NoGo, only Go -- this is a key param -- between 0.1-0.3 usu good -- see how much output gating happening and change accordingly"`
}

type StriatalCompartment int

const (
	PATCH StriatalCompartment = iota
	MATRIX
	NSComp
)

var KiT_StriatalCompartment = kit.Enums.AddEnum(NSComp, kit.NotBitFlag, nil)

// Delayed inhibition for matrix compartment layers
type DelayedInhibParams struct {
	Active bool    `desc:"add in a portion of inhibition from previous time period"`
	PrvQ   float32 `desc:"proportion of per-unit net input on previous gamma-frequency quarter to add in as inhibition"`
	PrvTrl float32 `desc:"proportion of per-unit net input on previous trial to add in as inhibition"`
}

// Params for for trace-based learning
type MSNTraceParams struct {
	Deriv bool    `def:"true" desc:"use the sigmoid derivative factor 2 * act * (1-act) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)"`
	Decay float32 `def:"1" min:"0" desc:"multiplier on trace activation for decaying prior traces -- new trace magnitude drives decay of prior trace -- if gating activation is low, then new trace can be low and decay is slow, so increasing this factor causes learning to be more targeted on recent gating changes"`
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
			val = ly.UnitVal1D(idx, unitIdx)
		}
	}
	return float64(val)
}

// AddMatrixLayer adds a MSNLayer of given size, with given name.
// nY = number of pools in Y dimension, nX is pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func AddMSNLayer(nt *Network, name string, nY, nX, nNeurY, nNeurX int, cpmt StriatalCompartment, da DaRType) *MSNLayer {
	stri := &MSNLayer{}
	nt.AddLayerInit(stri, name, []int{nY, nX, nNeurY, nNeurX}, emer.Hidden)
	stri.ModLayer.Init()
	stri.DaRType = da
	stri.MSN.Compartment = cpmt
	return stri
}

func (tp *MSNTraceParams) Defaults() {
	tp.Deriv = true
	tp.Decay = 1
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

type MSNVariant struct {
	//DV          DorsalVentral       `inactive:"+" desc:"dorsal or ventral"`
	Compartment StriatalCompartment `inactive:"+" desc:"patch or matrix"`
	// DaR is hidden because it runs into a widget config bug
	//DaR         DaRType             `view:"-" desc:"dominant type of dopamine receptor -- D1R for Go pathway, D2R for NoGo"`
}

// MSNeuron contains extra variables for MSNLayer neurons -- stored separately
type MSNeuron struct {
	//DALrn    float32 `desc:"dopamine value for learning"`
	GePrvQ   float32 `desc:"netin from previous quarter, used for delayed inhibition"`
	GePrvTrl float32 `desc:"netin from previous \"trial\" (alpha cycle), used for delayed inhibition"`
	//GiEx	float32 `desc:"extra inhibition value, used by matrix delayed inhibition"`
}

//func (ly *MSNLayer) SendMods(ltime *leabra.Time) {}

// Defaults in param.Sheet format
// Sel: "MSNLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.Gi":     "1.9",
// 		"Layer.Inhib.Layer.FB":     "0.5",
// 		"Layer.Inhib.Pool.On":      "true",
// 		"Layer.Inhib.Pool.Gi":      "1.9",
// 		"Layer.Inhib.Pool.FB":      "0",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "1.3",
// 		"Layer.Inhib.ActAvg.Init":  "0.2",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 	}}

func (ly *MSNLayer) Defaults() {
	ly.ModLayer.Defaults()
	ly.MSN.Defaults()
	ly.DaOn = true
	ly.Inhib.Layer.Gi = 1.9
	ly.Inhib.Layer.FB = 0.5
	ly.Inhib.Pool.On = true
	ly.Inhib.Pool.Gi = 1.9
	ly.Inhib.Pool.FB = 0
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.3
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 0.2
	ly.DelInh.Active = ly.MSN.Compartment == MATRIX
	if ly.DelInh.Active {
		ly.DelInh.PrvQ = 0
		ly.DelInh.PrvTrl = 6
	}
	for ni := range ly.MSNNeurs {
		msn := &ly.MSNNeurs[ni]
		mnr := &ly.ModNeurs[ni]
		msn.GePrvQ = 0
		msn.GePrvTrl = 0
		//msn.GiEx = 0
		mnr.DA = 0
		mnr.ACh = 0
	}
}

func (mp *MSNParams) Defaults() {
	mp.PatchShunt = 0.2
	mp.ShuntACh = true
	mp.OutAChInhib = 0.3
}

func (ly *MSNLayer) GetDA() float32 {
	return ly.DA
}

func (ly *MSNLayer) SetDA(da float32) {
	for ni := range ly.MSNNeurs {
		mnr := &ly.ModNeurs[ni]
		mnr.DA = da
	}
	ly.DA = da
}

func (ly *MSNLayer) QuarterInitPrvs(ltime *leabra.Time) {
	for ni := range ly.MSNNeurs {
		msn := &ly.MSNNeurs[ni]
		nrn := &ly.Neurons[ni]
		if ltime.Quarter == 0 {
			msn.GePrvQ = msn.GePrvTrl
		} else {
			msn.GePrvQ = nrn.Ge
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
	ly.MSNNeurs = make([]MSNeuron, len(ly.Neurons))
	return err
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *MSNLayer) InitActs() {
	ly.ModLayer.InitActs()
	for ni := range ly.Neurons {
		msn := &ly.MSNNeurs[ni]
		msn.GePrvQ = 0
		msn.GePrvTrl = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

func (ly *MSNLayer) PoolDelayedInhib(pl *leabra.Pool) {
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		msn := &ly.MSNNeurs[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
		nrn.Gi = pl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
		nrn.Gi += ly.DelInh.PrvTrl*msn.GePrvTrl + ly.DelInh.PrvQ*msn.GePrvQ
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
// this is here for matrix delyed inhibition, not needed otherwise
func (ly *MSNLayer) InhibFmGeAct(ltime *leabra.Time) {
	if ly.DelInh.Active {
		lpl := &ly.Pools[0]
		ly.Inhib.Layer.Inhib(&lpl.Inhib)
		np := len(ly.Pools)
		if np > 1 {
			for pi := 1; pi < np; pi++ {
				pl := &ly.Pools[pi]
				ly.Inhib.Pool.Inhib(&pl.Inhib)
				pl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi)
				ly.PoolDelayedInhib(pl)
			}
		} else {
			ly.PoolDelayedInhib(lpl)
		}
	} else {
		ly.ModLayer.InhibFmGeAct(ltime)
	}
}

func (ly *MSNLayer) AlphaCycInit() {
	if ly.DelInh.Active {
		for ni := range ly.MSNNeurs {
			msn := &ly.MSNNeurs[ni]
			nrn := &ly.Neurons[ni]
			msn.GePrvTrl = nrn.Ge
		}
	}
	ly.ModLayer.AlphaCycInit()
}

func (ly *MSNLayer) ModsFmInc(_ *leabra.Time) {
	ly.DaAChFmLay()
	ly.SetModLevels()
}

// DaAChFmLay computes Da and ACh from layer and Shunt received from PatchLayer units
func (ly *MSNLayer) DaAChFmLay() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		//msnr := &ly.MSNNeurs[ni]
		mnr := &ly.ModNeurs[ni]
		mnr.DA = ly.DA
		mnr.ACh = ly.ACh
		//msnr.DALrn = ly.DALrnFmDA(mnr.DA)
	}
}

func (ly *MSNLayer) ActFmG(_ *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mnr := &ly.ModNeurs[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		if math32.IsNaN(nrn.Act) {
			fmt.Println("NaN in MSN ActFmG")
		}
		if !ly.IsModSender {
			newAct := nrn.Act * mnr.ModLevel
			newDel := nrn.Act - newAct
			nrn.Act = newAct
			nrn.ActDel -= newDel
		}
		mnr.ModAct = nrn.Act // ModAct is used in DWt. Don't want to modulate Act with DA, or things get weird very quickly
		daVal := ly.DALrnFmDA(mnr.DA)
		mnr.ModAct *= 1 + daVal
		if mnr.PVAct > 0.01 { //&& ltime.PlusPhase {
			mnr.ModAct = mnr.PVAct // this gives results that look more like the CEmer model (rather than setting Act directly from PVAct)
		}
		ly.Learn.AvgsFmAct(nrn)
	}
}
