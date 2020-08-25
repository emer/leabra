// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"strconv"
)

type MSNLayer struct {
	ModLayer
	MSNVariant `inactive:"+" desc:"specific type of medium spiny neuron"`
	MSN        MSNParams          `view:"inline" desc:"striatal layer parameters"`
	MSNNeurs   []MSNeuron         `desc:"slice of MSNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values."`
	DelInh     DelayedInhibParams `view:"no-inline add-fields"`
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

// MSNParams has parameters for Dorsal Striatum Medium Spiny Neuron computation
// These are the main Go / NoGo gating units in BG driving updating of PFC WM in PBWM
type MSNParams struct {
	PatchShunt  float32 `def:"0.2,0.5" min:"0" max:"1" desc:"how much the patch shunt activation multiplies the dopamine values -- 0 = complete shunting, 1 = no shunting -- should be a factor < 1.0"`
	ShuntACh    bool    `def:"true" desc:"also shunt the ACh value driven from TAN units -- this prevents clearing of MSNConSpec traces -- more plausibly the patch units directly interfere with the effects of TAN's rather than through ach, but it is easier to implement with ach shunting here."`
	OutAChInhib float32 `def:"0,0.3" desc:"how much does the LACK of ACh from the TAN units drive extra inhibition to output-gating MSN units -- gi += out_ach_inhib * (1-ach) -- provides a bias for output gating on reward trials -- do NOT apply to NoGo, only Go -- this is a key param -- between 0.1-0.3 usu good -- see how much output gating happening and change accordingly"`
}

/*
  float         ach_reset_thr;  // #MIN_0 #DEF_0.5 threshold on receiving unit ach value, sent by TAN units, for reseting the trace -- only applicable for trace-based learning
  bool          msn_deriv;      // #DEF_true use the sigmoid derivative factor msn * (1-msn) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)
  float         max_vs_deep_mod; // for VS matrix TRACE_NO_THAL_VS and DA_HEBB_VS learning rules, this is the maximum value that the deep_mod_net modulatory inputs from the basal amygdala (up state enabling signal) can contribute to learning

*/

type StriatalCompartment int

const (
	PATCH StriatalCompartment = iota
	MATRIX
	NSComp
)

var KiT_StriatalCompartment = kit.Enums.AddEnum(NSComp, kit.NotBitFlag, nil)

////go:generate stringer -type=StriatalCompartment // moved to stringers.go

//type DorsalVentral int
//const (
//	DORSAL DorsalVentral = iota
//	VENTRAL
//	NDV
//)
//var KiT_DorsalVentral = kit.Enums.AddEnum(NDV, kit.NotBitFlag, nil)
////go:generate stringer -type=DorsalVentral

type DelayedInhibParams struct {
	Active bool    `desc:"add in a portion of inhibition from previous time period"`
	PrvQ   float32 `desc:"proportion of per-unit net input on previous gamma-frequency quarter to add in as inhibition"`
	PrvTrl float32 `desc:"proportion of per-unit net input on previous trial to add in as inhibition"`
}

// Params for for trace-based learning, including DA_HEBB_VS
type MSNTraceParams struct {
	//NotGatedLR    float32 `def:"0.7" min:"0" desc:"learning rate for all not-gated stripes, which learn in the opposite direction to the gated stripes, and typically with a slightly lower learning rate -- although there are different learning logics associated with each of these different not-gated cases, in practice the same learning rate for all works best, and is simplest"`
	//GateNoGoPosLR float32 `def:"0.1" min:"0" desc:"learning rate for gated, NoGo (D2), positive dopamine (weights decrease) -- this is the single most important learning parameter here -- by making this relatively small (but non-zero), an asymmetry in the role of Go vs. NoGo is established, whereby the NoGo pathway focuses largely on punishing and preventing actions associated with negative outcomes, while those assoicated with positive outcomes only very slowly get relief from this NoGo pressure -- this is critical for causing the model to explore other possible actions even when a given action SOMETIMES produces good results -- NoGo demands a very high, consistent level of good outcomes in order to have a net decrease in these avoidance weights.  Note that the gating signal applies to both Go and NoGo MSN's for gated stripes, ensuring learning is about the action that was actually selected (see not_ cases for logic for actions that were close but not taken)"`
	//AChResetThr   float32 `min:"0" def:"0.5" desc:"threshold on receiving unit ACh value, sent by TAN units, for reseting the trace"`
	Deriv bool    `def:"true" desc:"use the sigmoid derivative factor 2 * act * (1-act) in modulating learning -- otherwise just multiply by msn activation directly -- this is generally beneficial for learning to prevent weights from continuing to increase when activations are already strong (and vice-versa for decreases)"`
	Decay float32 `def:"1" min:"0" desc:"multiplier on trace activation for decaying prior traces -- new trace magnitude drives decay of prior trace -- if gating activation is low, then new trace can be low and decay is slow, so increasing this factor causes learning to be more targeted on recent gating changes"`
}

func (ly *MSNLayer) GetMonitorVal(data []string) float64 {
	var val float32
	valType := data[0]
	unitIdx, _ := strconv.Atoi(data[1])
	switch valType {
	case "TotalAct":
		val = GlobalTotalActFn(ly)
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

func (tp *MSNTraceParams) Defaults() {
	//tp.NotGatedLR = 0.7
	//tp.GateNoGoPosLR = 0.1
	//tp.AChResetThr = 0.5
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
	DALrn    float32 `desc:"dopamine value for learning"`
	Shunt    float32 `desc:"shunting input received from Patch neurons (in reality flows through SNc DA pathways)"`
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
	ly.DelInh.Active = ly.Compartment == MATRIX
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
		msn.Shunt = 0
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

// Quarter2DWt is optional Q2 DWt -- define where relevant
func (ly *MSNLayer) Quarter2DWt() {
	for pi := range ly.SndPrjns {
		pj := ly.SndPrjns[pi]
		if pj.IsOff() {
			continue
		}
		rly := pj.RecvLay()
		imrly, ok := rly.(IMSNLayer)
		if ok {
			mrly := imrly.AsMSNLayer()
			if mrly.DoQuarter2DWt() {
				pj.(leabra.LeabraPrjn).DWt()
			}
		}
	}
}

func (ly *MSNLayer) QuarterFinal(ltime *leabra.Time) {
	ly.Layer.QuarterFinal(ltime)
	if ltime.Quarter == 1 {
		ly.Quarter2DWt()
	}
	//if ly.DelInh.Active {
	//	for ni := range ly.MSNNeurs {
	//		snr := &ly.MSNNeurs[ni]
	//		nrn := &ly.Neurons[ni]
	//		snr.GePrvQ = nrn.Ge
	//		if ltime.Quarter == 3 {
	//			snr.GePrvTrl = snr.GePrvQ
	//		}
	//	}
	//}
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
		msn.Shunt = 0
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
		//msn.GiEx += ly.DelInh.PrvTrl*msn.GePrvTrl + ly.DelInh.PrvQ*msn.GePrvQ
		nrn.Gi = pl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn + ly.DelInh.PrvTrl*msn.GePrvTrl + ly.DelInh.PrvQ*msn.GePrvQ
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
				//ly.PoolDelayedInhib(pl)
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
		} else {
			//ly.PoolDelayedInhib(lpl)
			for ni := lpl.StIdx; ni < lpl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				msn := &ly.MSNNeurs[ni]
				if nrn.IsOff() {
					continue
				}
				ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
				nrn.Gi = lpl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
				nrn.Gi += ly.DelInh.PrvTrl*msn.GePrvTrl + ly.DelInh.PrvQ*msn.GePrvQ
			}
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

//DoQuarter2DWt indicates whether to do optional Q2 DWt
func (ly *MSNLayer) DoQuarter2DWt() bool {
	return true
}

func (ly *MSNLayer) ModsFmInc(ltime *leabra.Time) {
	ly.DaAChFmLay(ltime)
	ly.CalcActMod()
	ly.SetModLevels(ltime)
}

//func (ly *MSNLayer) CalcActMod() {
//	pl := &ly.Pools[0].Inhib.Act
//	for ni := range ly.Neurons {
//		mnr := &ly.ModNeurs[ni]
//		//nrn := &ly.Neurons[ni]
//		if mnr.ModNet <= ly.ModNetThreshold {
//			mnr.ModLrn = 0
//			if ly.Compartment == PATCH && ly.ActModZero {
//				mnr.ModLevel = 0
//			} else {
//				mnr.ModLevel = 1
//			}
//		} else {
//			//pl := &ly.Pools[nrn.SubPool].Inhib.Act
//			if pl.Max != 0 {
//				//	mnr.ModLrn = 1
//				//} else {
//				mnr.ModLrn = math32.Min(1, mnr.ModNet/pl.Max)
//			} else {
//				mnr.ModLrn = 0
//			}
//			mnr.ModLevel = 1
//		}
//	}
//}

func (ly *MSNLayer) CalcActMod() {
	plMax := ly.ModPools[0].ModNetStats.Max
	for ni := range ly.Neurons {
		mnr := &ly.ModNeurs[ni]
		//nrn := &ly.Neurons[ni]
		if ly.Compartment == MATRIX {
			if mnr.ModNet <= ly.ModNetThreshold {
				mnr.ModLrn = 0
				mnr.ModLevel = 1
			} else {
				if plMax != 0 {
					mnr.ModLrn = math32.Min(1, mnr.ModNet/plMax)
				}
				mnr.ModLevel = 1
			}

		} else { // PATCH
			if mnr.ModNet <= ly.ModNetThreshold {
				mnr.ModLrn = 0
				if ly.ActModZero {
					mnr.ModLevel = 0
				} else {
					mnr.ModLevel = 1
				}
			} else {
				if plMax != 0 {
					mnr.ModLrn = math32.Min(1, mnr.ModNet/plMax)
				} else {
					mnr.ModLrn = 0
				}
				mnr.ModLevel = 1
			}

		}
	}
}

func (ly *MSNLayer) SetModLevels(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mnr := &ly.ModNeurs[ni]
		mnr.ModAct = nrn.Act
	}
}

// DaAChFmLay computes Da and ACh from layer and Shunt received from PatchLayer units
func (ly *MSNLayer) DaAChFmLay(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		msnr := &ly.MSNNeurs[ni]
		mnr := &ly.ModNeurs[ni]
		mnr.DA = ly.DA
		mnr.ACh = ly.ACh
		//if msnr.Shunt > 0 { // note: treating Shunt as binary variable -- could multiply
		//	mnr.DA *= ly.MSN.PatchShunt
		//	if ly.MSN.ShuntACh {
		//		mnr.ACh *= ly.MSN.PatchShunt
		//	}
		//}
		msnr.DALrn = ly.DALrnFmDA(mnr.DA)
	}
}

func (ly *MSNLayer) ActFmG(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mnr := &ly.ModNeurs[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		if !ly.IsModSender {
			newAct := nrn.Act * mnr.ModLevel
			newDel := nrn.Act - newAct
			nrn.Act = newAct
			nrn.ActDel -= newDel
		}
		ly.Learn.AvgsFmAct(nrn)
	}
}
