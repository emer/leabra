package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	_ "github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

type ICElAmygLayer interface {
	AsCElAmygLayer() *CElAmygLayer
}

func (ly *CElAmygLayer) AsCElAmygLayer() *CElAmygLayer {
	return ly
}

type AcqExt int

const (
	Acq AcqExt = iota
	Ext
	NAcqExt
)

////go:generate stringer -type=AcqExt // moved to stringers.go
var KiT_AcqExt = kit.Enums.AddEnum(NAcqExt, kit.NotBitFlag, nil)

type CElAmygLayer struct {
	AmygdalaLayer
	CElTyp     CElAmygLayerType `desc:"basic parameters determining what type CEl layer this is"`
	AcqDeepMod bool             `desc:"use deep_mod_net for value from acquisition / go units, instead of inhibition current (otherwise use gi_syn) -- allows simpler parameter setting without titrating inhibition and this learning modulation signal"`
}

var KiT_CElAmygLayer = kit.Types.AddType(&CElAmygLayer{}, nil)

type ICElAmygPrjn interface {
	AsCElAmygPrjn() *CElAmygPrjn
	IModPrjn
}

func (pj *CElAmygPrjn) AsCElAmygPrjn() *CElAmygPrjn {
	return pj
}

func (pj *CElAmygPrjn) AsModPrjn() *ModHebbPrjn {
	return &pj.ModHebbPrjn
}

type CElAmygLayerType struct {
	AcqExt  AcqExt  `desc:"acquisition or extinction"`
	Valence Valence `desc:"positive or negative DA valence"`
}

type CElAmygPrjn struct {
	ModHebbPrjn
}

var KiT_CElAmygPrjn = kit.Types.AddType(&CElAmygPrjn{}, nil)

// Defaults in param.Sheet format
// Sel: "CElAmygLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.CElLearn.DALRGain":    "1.0",
// 		"Layer.CElLearn.DALRBase":    "0.0",
// 		"Layer.CElLearn.DALrnThr":    "0.01",
// 		"Layer.CElLearn.ActDeltaThr": "0.0",
// 		"Layer.CElDAMod.ActModZero": "false",
// 		"Layer.CElDAMod.AcqDeepMod":  "true",
// 		"Layer.CElDAMod.LrnModAct":   "false",
// 		"Layer.CElDAMod.BurstDAGain": "0.04",
// 		"Layer.CElDAMod.DipDAGain":   "0.1",
// 		"Layer.CElDAMod.USClampAvg":  "0.2",
// 		"Layer.Act.Init.Vm":          "0.55",
// 		"Layer.Act.Gbar.L":           "0.6",
// 		"Layer.Act.Erev.L":           "0.55",
// 		"Layer.Act.Erev.I":           "0.4",
// 		"Layer.DeepBurst.On":         "true",
// 		"Layer.DeepBurst.Qtr":        "8", //Q4
// 		"Layer.DeepBurst.ThrRel":     "0.1",
// 		"Layer.DeepBurst.ThrAbs":     "0.1",
// 		"Layer.DeepAttn.Thr":         "0.01",
// 	}}

func (ly *CElAmygLayer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.Neurons = make([]leabra.Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	if err != nil {
		return err
	}
	err = ly.AsMod().Build()
	if err != nil {
		return err
	}
	ly.Defaults()
	return err
}

func (ly *CElAmygLayer) Defaults() {
	ly.DALrnThr = 0.01
	ly.ActDeltaThr = 0.0
	ly.Act.Init.Vm = 0.55
	ly.ActModZero = false
	ly.AcqDeepMod = true
	ly.LrnModAct = false
	ly.BurstDAGain = 0.04 // differs between CEl layers
	ly.DipDAGain = 0.1    // differs between CEl layers
	ly.USClampAvg = 0.2   // differs between CEl layers
	ly.ModLayer.Defaults()
	ly.DebugVal = -1
}

func (pj *CElAmygPrjn) Build() error {
	err := pj.Prjn.Build()
	if err != nil {
		return err
	}
	return nil
}

// Compute DA-modulated weight changes for CElAmyg layers
func (pj *CElAmygPrjn) DWt() {
	// CON_STATE* scg, ss *Sim, thrNo int
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlayi := pj.Recv.(ICElAmygLayer)
	rlay := rlayi.AsCElAmygLayer()
	clRate := pj.Learn.Lrate // * rlay.CosDiff.ModAvgLLrn
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		snAct := sn.ActQ0
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			mn := &rlay.ModNeurs[ri]

			if rn.IsOff() {
				continue
			}
			// screen out spurious da signals due to tiny VSPatch-to-LHb signals
			if math32.Abs(mn.DA) < rlay.DALrnThr {
				mn.DA = 0
			}

			// inlined from CComputeDWtCElDelta
			rnActDelta := mn.ModAct - rn.ActQ0
			if math32.Abs(rnActDelta) < rlay.ActDeltaThr {
				rnActDelta = 0.0
			}
			delta := clRate * snAct * rnActDelta
			// dopamine signal further modulates learning
			daLRate := pj.DALRBase + pj.DALRGain*math32.Abs(mn.DA)
			if rlay.DebugVal > 0 && delta != 0 {
				fmt.Printf("%v->%v[%v]: delta=%v, daLRate=%v, prevDWt=%v\n", slay.Name(), rlay.Name(), ri, delta, daLRate, sy.DWt)
			}
			sy.DWt += daLRate * delta
			//rlayi.CComputeDWtCElDelta(dwts[i], su_act, ru->act_eq, ru->act_q0, ru_da_p, clrate)
		}
	}
}

// Compute_DeepMod in CEmer. Called early in Compute_Act_Rate, if deep.on
// Called (if at all) before Compute_ApplyInhib, Compute_Vm, and Compute_ActFun_Rate
// Calculate activity-based modulation values
func (ly *CElAmygLayer) CalcActMod() {
	mpl := &ly.ModPools[0].ModNetStats
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		mnr := &ly.ModNeurs[ni]
		if ly.IsModSender { // record what we send!
			mnr.ModLrn = nrn.Act
			mnr.ModLevel = nrn.Act
		} else if mnr.ModNet <= ly.ModNetThreshold { // not enough yet
			mnr.ModLrn = 0 // default is 0!
			if ly.ActModZero {
				mnr.ModLevel = 0
			} else {
				mnr.ModLevel = 1
			}
		} else {
			if mpl.Max != 0 {
				mnr.ModLrn = mnr.ModNet / mpl.Max
			} else {
				mnr.ModLrn = 0
			}
			mnr.ModLevel = 1 // do not modulate with deep_mod!
		}
	}
}

func (ly *CElAmygLayer) ModsFmInc(ltime *leabra.Time) {
	ly.CalcActMod()
}

func (ly *CElAmygLayer) ActFmG(ltime *leabra.Time) {
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
		mnr.ModAct = nrn.Act // ModAct is used in DWt. Don't want to modulate Act with DA, or things get weird very quickly
		daVal := ly.DALrnFmDA(mnr.DA)
		mnr.ModAct *= 1 + daVal
		if mnr.PVAct > 0.01 { //&& ltime.PlusPhase {
			mnr.ModAct = mnr.PVAct // this gives results that look more like the CEmer model (rather than setting Act)
		}
		ly.Learn.AvgsFmAct(nrn)
	}
}
