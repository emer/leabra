package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/interinhib"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"strconv"
)

type IBlAmygLayer interface {
	AsBlAmygLayer() *BlAmygLayer
}

func (ly *BlAmygLayer) AsBlAmygLayer() *BlAmygLayer {
	return ly
}

type BLAchMod struct {
	On   bool    `desc:"whether to modulate activations as a function of ach levels"`
	Min  float32 `desc:"minimum ach modulation factor -- net modulation is BLAchMod.Min + ach * (1 - BLAchMod.Min)"`
	MinC float32 `desc:"1 - BLAchMod.Min"`
}

var KiT_BLAchMod = kit.Types.AddType(&BLAchMod{}, nil)

type BlAmygLayer struct {
	AmygdalaLayer `desc:"amygdala-specific"`
	Valence       Valence               `desc:"positive or negative valence"`
	AChMod        BLAchMod              `desc:"cholinergic modulation"`
	ILI           interinhib.InterInhib `desc:"inter-layer inhibition parameters and state"`
}

var KiT_BlAmygLayer = kit.Types.AddType(&BlAmygLayer{}, nil)

type BlAmygPrjn struct {
	ModHebbPrjn
}

type IBLAmygPrjn interface {
	AsBlAmygPrjn() *BlAmygPrjn
	IModPrjn
}

func (pj *BlAmygPrjn) AsBlAmygPrjn() *BlAmygPrjn {
	return pj
}

var KiT_BlAmygPrjn = kit.Types.AddType(&BlAmygPrjn{}, nil)

func (pj *BlAmygPrjn) AsModPrjn() *ModHebbPrjn {
	return &pj.ModHebbPrjn
}

func (ly *BlAmygLayer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.Neurons = make([]leabra.Neuron, nu)
	err := ly.AsMod().Build()
	if err != nil {
		return err
	}
	err = ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	if err != nil {
		return err
	}
	ly.Defaults()
	return err
}

func (ly *BlAmygLayer) Defaults() {
	ly.AChMod.On = true
	ly.AChMod.Min = 0.8
	ly.AChMod.MinC = 1.0 - ly.AChMod.Min

	ly.ActModZero = true
	ly.Act.Init.Vm = 0.55
	ly.Act.Gbar.L = 0.6
	ly.Act.Erev.L = 0.55
	ly.Act.Erev.I = 0.4
	ly.BurstDAGain = 0.04
	ly.DipDAGain = 0.1
	ly.LrnModAct = false
	ly.USClampAvg = 0.2

	ly.DaOn = true
	ly.Minus = 1.0
	ly.Plus = 1.0
	ly.NegGain = 0.1
	ly.PosGain = 0.1
	ly.ModLayer.Defaults()
	ly.DebugVal = -1
}

func (ly *BlAmygLayer) GetMonitorVal(data []string) float64 {
	var val float32
	var err error
	valType := data[0]
	unitIdx, _ := strconv.Atoi(data[1])
	switch valType {
	case "TotalAct":
		val = GlobalTotalActFn(ly)
	case "PoolActAvg":
		val = ly.Pools[unitIdx].Inhib.Act.Avg
	case "PoolActMax":
		val = ly.Pools[unitIdx].Inhib.Act.Max
	case "Act":
		val = ly.Neurons[unitIdx].Act
	case "ActDiff":
		val = ly.Neurons[unitIdx].Act - ly.ModNeurs[unitIdx].ModAct
	default:
		mnr := &ly.ModNeurs[unitIdx]
		val, err = mnr.VarByName(valType)
		if err != nil {
			nrn := &ly.Neurons[unitIdx]
			val, err = nrn.VarByName(valType)
			if err != nil {
				fmt.Printf("VarByName error: %v\n", err)
			}
		}
	}
	return float64(val)
}

func (pj *BlAmygPrjn) InitWts() {
	if pj.SetScale {
		pj.SetScalesFunc(pj.GaussScale)
		pj.SetWtsFunc(func(_, _ int, _, _ *etensor.Shape) float32 {
			return pj.InitWtVal
		})
		for si := range pj.Syns {
			sy := &pj.Syns[si]
			sy.DWt = 0
			sy.Norm = 0
			sy.Moment = 0
		}
	} else {
		pj.Prjn.InitWts()
	}
}

// GaussScale returns gaussian weight value for given unit indexes in
// given send and recv layers according to Gaussian Sigma and MaxWt.
func (pj *BlAmygPrjn) GaussScale(_, _ int, _, _ *etensor.Shape) float32 {
	scale := float32(pj.WtInit.Gen(-1))
	scale = math32.Max(pj.SetScaleMin, scale)
	scale = math32.Min(pj.SetScaleMax, scale)
	return scale
}

func (pj *BlAmygPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.DALrnThr = 0.0
	pj.ActDeltaThr = 0.05
	pj.ActLrnThr = 0.05
}

func (pj *BlAmygPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlayi := pj.Recv.(IBlAmygLayer)
	rlay := rlayi.AsBlAmygLayer()

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
			// filter any tiny spurious da signals on t2 & t4 trials - best for ext guys since
			// they have zero dalr_base value
			if math32.Abs(mn.DA) < pj.DALrnThr {
				mn.DA = 0
			}

			lRateEff := clRate
			// learning dependent on non-zero deep_lrn
			if pj.ActLrnMod {
				var effActLrn float32
				if mn.ModLrn > pj.ActLrnThr {
					effActLrn = 1
				} else {
					effActLrn = 0 // kills all learning
				}
				lRateEff *= effActLrn
			}
			//pj.CComputeDWtBlDelta(&sy.DWt, snAct, rn.Act, rn.ActQ0, rlay.DA, lRateEff)

			// inlined from CComputeDWtBlDelta
			rnActDelta := mn.ModAct - rn.ActQ0
			if math32.Abs(rnActDelta) < pj.ActDeltaThr {
				rnActDelta = 0
			}
			delta := lRateEff * snAct * rnActDelta
			// dopamine signal further modulates learning
			daLRate := pj.DALRBase + pj.DALRGain*math32.Abs(mn.DA)
			if rlay.DebugVal > 0 && delta != 0 {
				fmt.Printf("%v->%v[%v]: delta=%v, daLRate=%v, prevDWt=%v, wt=%v\n", slay.Name(), rlay.Name(), ri, delta, daLRate, sy.DWt, sy.Wt)
			}
			sy.DWt += daLRate * delta
		}
	}
}

// Compute_DeepMod in CEmer. Called early in Compute_Act_Rate, if deep.on
// Called (if at all) before Compute_ApplyInhib, Compute_Vm, and Compute_ActFun_Rate
// Calculate activity-based modulation values
func (ly *BlAmygLayer) CalcActMod() {
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
		//if ly.AChMod.On {
		//	mnr.ModLevel = ly.AChMod.Min + ly.AChMod.Min*ly.ACh
		//}
	}
}

func (ly *BlAmygLayer) ModsFmInc(ltime *leabra.Time) {
	ly.CalcActMod()
}

// InhibiFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *BlAmygLayer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.ILI.Inhib(&ly.Layer) // does inter-layer inhibition
	ly.PoolInhibFmGeAct(ltime)
	ly.InhibFmPool(ltime)
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *BlAmygLayer) ActFmG(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mnr := &ly.ModNeurs[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		if !ly.IsModSender { // is receiver
			newAct := nrn.Act * mnr.ModLevel
			newDel := nrn.Act - newAct
			nrn.Act = newAct
			nrn.ActDel -= newDel
		}
		mnr.ModAct = nrn.Act // ModAct is used in DWt
		daVal := ly.DALrnFmDA(mnr.DA)
		mnr.ModAct *= 1 + daVal
		ly.Learn.AvgsFmAct(nrn)
	}
}
