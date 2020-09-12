// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// layers that receive modulatory projections--(original code from SendDeepMod in cemer)

package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"strconv"
)

type IModLayer interface {
	AsMod() *ModLayer
}

type AvgMaxModLayer interface {
	AvgMaxMod(*leabra.Time)
}

type ModReceiver interface {
	ModsFmInc(ltime *leabra.Time)
	ReceiveMods(sender ModSender, scale float32)
}

type ModSender interface {
	SendMods(ltime *leabra.Time)
	ModSendValue(ni int32) float32
}

type ModLayer struct {
	leabra.Layer
	ModNeurs     []ModNeuron     `desc:"neuron-level modulation state"`
	ModPools     []ModPool       `desc:"pools for maintaining aggregate values"`
	ModReceivers []ModRcvrParams `desc:"layer names and scale values for mods sent from this layer"`
	ModParams    `desc:"parameters shared by all modulator receiver layers"`
	Modulators   `desc:"layer-level neuromodulator levels"`
}

var _ IModLayer = (*ModLayer)(nil)

var KiT_ModLayer = kit.Types.AddType(&ModLayer{}, nil)

type ModPool struct {
	ModNetStats      minmax.AvgMax32
	ModSent          float32 `desc:"modulation level transmitted to receiver layers"`
	ModSendThreshold float32 `desc:"threshold for sending modulation. values below this are not added to the pool-level total"`
}

type ModParams struct {
	DaOn             bool    `desc:"whether to use dopamine modulation"`
	DaRType          DaRType `inactive:"+" desc:"dopamine receptor type, D1 or D2"`
	LrnModAct        bool    `desc:"if true, phasic dopamine values effect learning by modulating net_syn values (Compute_NetinExtras() - and thus unit activations; - CAUTION - very brittle and hard to use due to unintended consequences!"`
	PctAct           float32 `desc:"if LrnModAct is true, proportion of activation used for computing dopamine modulation value -- 1-pct_act comes from net-input -- activation is more differentiated and leads to more differentiated representations, but if there is no activation then dopamine modulation has no effect, so it depends on having that activation signal"`
	Minus            float32 `viewif:"On" desc:"how much to multiply Da in the minus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`
	Plus             float32 `viewif:"On" desc:"how much to multiply Da in the plus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`
	NegGain          float32 `viewif:"DaOn&&ModGain" desc:"for negative dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * NegNain) -- da is multiplied by minus or plus depending on phase"`
	PosGain          float32 `viewif:"DaOn&&ModGain" desc:"for positive dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * PosGain) -- da is multiplied by minus or plus depending on phase"`
	ActModZero       bool    `desc:"for modulation coming from the BLA via deep_mod_net -- when this modulation signal is below zero, does it have the ability to zero out the patch activations?  i.e., is the modulation required to enable patch firing?"`
	ModNetThreshold  float32 `desc:"threshold on deep_mod_net before deep mod is applied -- if not receiving even this amount of overall input from deep_mod sender, then do not use the deep_mod_net to drive deep_mod and deep_lrn values -- only for SUPER units -- based on LAYER level maximum for base LeabraLayerSpec, PVLV classes are based on actual deep_mod_net for each unit"`
	ModSendThreshold float32 `desc:"threshold for including neuron activation in total to send (for ModNet)"`
	IsModSender      bool    `desc:"does this layer send modulation to other layers?"`
	IsPVReceiver     bool    `desc:"does this layer receive a direct PV input?"`
	BurstDAGain      float32 `desc:"multiplicative gain factor applied to positive dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign!"`
	DipDAGain        float32 `desc:"multiplicative gain factor applied to negative dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign! should be small for acq, but roughly equal to burst_da_gain for ext"`
}

type ModRcvrParams struct {
	RcvName string  `desc:"name of receiving layer"`
	Scale   float32 `desc:"scale factor for modulation to this receiver"`
}

var KiT_ModParams = kit.Types.AddType(&ModParams{}, nil)

type Modulators struct {
	DA  float32 `desc:"current dopamine level for this layer"`
	ACh float32 `desc:"current acetylcholine level for this layer"`
	SE  float32 `desc:"current serotonin level for this layer"`
}

var KiT_Modulators = kit.Types.AddType(&Modulators{}, nil)

type ModNeuron struct {
	Modulators `desc:"neuron-level modulator activation"`
	ModAct     float32 `desc:"activity level for modulation"`
	ModLevel   float32 `desc:"degree of full modulation to apply"`
	ModNet     float32 `desc:"modulation input from sender"`
	ModLrn     float32 `desc:"multiplier for DA modulation of learning rate"`
	PVAct      float32 `desc:"direct activation from US"`
}

var KiT_ModNeuron = kit.Types.AddType(&ModNeuron{}, nil)

// AsMod returns a pointer to the ModLayer portion of the layer
func (ly *ModLayer) AsMod() *ModLayer {
	return ly
}

// Get a pointer to the generic Leabra portion of the layer
func (ly *ModLayer) AsLeabra() *leabra.Layer {
	return &ly.Layer
}

// Ge returns da-modulated ge value
func (dm *ModParams) Ge(da, ge float32, plusPhase bool) float32 {
	if plusPhase {
		return dm.Plus * da * ge
	} else {
		return dm.Minus * da * ge
	}
}

// Gain returns da-modulated gain value
func (dm *ModParams) Gain(da, gain float32, plusPhase bool) float32 {
	if plusPhase {
		da *= dm.Plus
	} else {
		da *= dm.Minus
	}
	if da < 0 {
		return gain * (1 + da*dm.NegGain)
	} else {
		return gain * (1 + da*dm.PosGain)
	}
}

// DaRType for D1R and D2R dopamine receptors
type DaRType int

const (
	// D1R primarily expresses Dopamine D1 Receptors -- dopamine is excitatory and bursts of dopamine lead to increases in synaptic weight, while dips lead to decreases -- direct pathway in dorsal striatum
	D1R DaRType = iota

	// D2R primarily expresses Dopamine D2 Receptors -- dopamine is inhibitory and bursts of dopamine lead to decreases in synaptic weight, while dips lead to increases -- indirect pathway in dorsal striatum
	D2R

	DaRTypeN
)

var KiT_DaRType = kit.Enums.AddEnum(DaRTypeN, kit.NotBitFlag, nil)

func (ev DaRType) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *DaRType) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// Retrieve a value for a trace of some quantity, possibly more than just a variable
func (ly *ModLayer) GetMonitorVal(data []string) float64 {
	var val float32
	var err error
	valType := data[0]
	unitIdx, _ := strconv.Atoi(data[1])
	switch valType {
	case "TotalAct":
		val = TotalAct(ly)
	case "ModAct":
		val = ly.ModNeurs[unitIdx].ModAct
	case "ModLevel":
		val = ly.ModNeurs[unitIdx].ModLevel
	case "ModNet":
		val = ly.ModNeurs[unitIdx].ModNet
	case "ModLrn":
		val = ly.ModNeurs[unitIdx].ModLrn
	case "PVAct":
		val = ly.ModNeurs[unitIdx].PVAct
	case "PoolActAvg":
		val = ly.Pools[unitIdx].Inhib.Act.Avg
	case "PoolActMax":
		val = ly.Pools[unitIdx].Inhib.Act.Max
	case "ModPoolAvg":
		val = ly.ModPools[unitIdx].ModNetStats.Avg
	case "DA":
		val = ly.ModNeurs[unitIdx].DA
	case "DALrn":
		val = ly.DALrnFmDA(ly.ModNeurs[unitIdx].DA)
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

// UnitValByIdx returns value of given variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *ModLayer) UnitValByIdx(vidx ModNeuronVar, idx int) float32 {
	switch vidx {
	case DA:
		return ly.ModNeurs[idx].DA
	case ACh:
		return ly.ModNeurs[idx].ACh
	case SE:
		return ly.ModNeurs[idx].SE
	case ModAct:
		return ly.ModNeurs[idx].ModAct
	case ModLevel:
		return ly.ModNeurs[idx].ModLevel
	case ModNet:
		return ly.ModNeurs[idx].ModNet
	case ModLrn:
		return ly.ModNeurs[idx].ModLrn
	case PVAct:
		return ly.ModNeurs[idx].PVAct
	default:
		return math32.NaN()
	}
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *ModLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	vidx, err = NeuronVarIdxByName(varNm)
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
func (ly *ModLayer) UnitVal1D(varIdx int, idx int) float32 {
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	if varIdx < 0 || varIdx >= len(ModNeuronVarsAll) {
		return math32.NaN()
	}
	nn := len(leabra.NeuronVars)
	if varIdx < nn {
		nrn := &ly.Neurons[idx]
		return nrn.VarByIndex(varIdx)
	}
	varIdx -= nn
	mnr := &ly.ModNeurs[idx]
	return mnr.VarByIndex(varIdx)
}

func (ly *ModLayer) Init() {
	ly.ModReceivers = []ModRcvrParams{}
}

func (ly *ModLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.ModPools = make([]ModPool, len(ly.Layer.Pools)) // uses regular Pools for indexes
	ly.ModNeurs = make([]ModNeuron, len(ly.Layer.Neurons))
	err = ly.BuildPrjns()
	if err != nil {
		return err
	}
	return nil
}

func (ly *ModLayer) Defaults() {
	ly.Layer.Defaults()
	ly.IsPVReceiver = false
	ly.ModSendThreshold = 0.1
	ly.Layer.UpdateParams()
	for pi := range ly.ModPools {
		mpl := &ly.ModPools[pi]
		mpl.ModNetStats.Init()
		mpl.ModSent = 0
		mpl.ModSendThreshold = 0
	}
}

func (ly *ModLayer) UpdateParams() {
	ly.Layer.UpdateParams()
}

func (ly *ModLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.ModNeurs {
		mnr := &ly.ModNeurs[ni]
		mnr.InitActs()
	}
	for pi := range ly.ModPools {
		mpl := &ly.ModPools[pi]
		mpl.ModNetStats.Init()
		mpl.ModSent = 0
	}
	ly.Modulators.InitActs()
}

func (ml *Modulators) InitActs() {
	ml.ACh = 0
	ml.DA = 0
	ml.SE = 0
}

func (mnr *ModNeuron) InitActs() {
	mnr.ModAct = 0
	mnr.ModLevel = 1
	mnr.ModNet = 0
	mnr.ModLrn = 1
	mnr.PVAct = 0
	mnr.Modulators.InitActs()
}

func (ly *ModLayer) ClearModLevels() {
	for ni := range ly.ModNeurs {
		mnr := &ly.ModNeurs[ni]
		mnr.ModAct = 0
		mnr.ModLevel = 1
		mnr.ModNet = 0
		mnr.ModLrn = 1
		mnr.PVAct = 0
	}
}

func (ly *ModLayer) AddModReceiver(rcvr ModReceiver, scale float32) {
	ly.IsModSender = true
	rly := rcvr.(IModLayer).AsMod()
	ly.ModReceivers = append(ly.ModReceivers, ModRcvrParams{rly.Name(), scale})
}

func (ly *ModLayer) ModSendValue(ni int32) float32 {
	return ly.ModPools[ni].ModSent
}

func (ly *ModLayer) SendMods(_ *leabra.Time) {
	for pi := range ly.ModPools {
		mpl := &ly.ModPools[pi]
		mpl.ModSent = 0
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mpl := &ly.ModPools[nrn.SubPool]
		if math32.Abs(nrn.Act) > ly.Act.OptThresh.Send {
			mpl.ModSent += nrn.Act
		}
	}
	for _, mr := range ly.ModReceivers {
		rl := ly.Network.LayerByName(mr.RcvName).(ModReceiver)
		rl.ReceiveMods(ly, mr.Scale)
	}
}

func (ly *ModLayer) ReceiveMods(sender ModSender, scale float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mnr := &ly.ModNeurs[ni]
		mnr.ModNet = sender.ModSendValue(nrn.SubPool) * scale
	}
}

func (ly *ModLayer) ModsFmInc(_ *leabra.Time) {
	plMax := ly.ModPools[0].ModNetStats.Max
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
			mnr.ModLrn = 0 // default is 0
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
			mnr.ModLevel = 1 // do not modulate!
		}
	}
}

func (ly *ModLayer) ClearModActs() {
	for ni := range ly.ModNeurs {
		mnr := &ly.ModNeurs[ni]
		mnr.InitActs()
	}
	for pi := range ly.ModPools {
		mpl := &ly.ModPools[pi]
		mpl.ModSent = 0
		mpl.ModNetStats.Init()
	}
}

func (ly *ModLayer) GScaleFmAvgAct() {
	totGeRel := float32(0)
	totGiRel := float32(0)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(leabra.LeabraPrjn).AsLeabra()
		slay := p.SendLay().(leabra.LeabraLayer).AsLeabra()
		slpl := &slay.Pools[0]
		savg := slpl.ActAvg.ActPAvgEff
		snu := len(slay.Neurons)
		ncon := pj.RConNAvgMax.Avg
		pj.GScale = pj.WtScale.FullScale(savg, float32(snu), ncon)
		switch pj.Typ {
		case emer.Inhib:
			totGiRel += pj.WtScale.Rel
		default:
			totGeRel += pj.WtScale.Rel
		}
		if ly.IsPVReceiver {
			totGeRel += 1
		}
	}

	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(leabra.LeabraPrjn).AsLeabra()
		switch pj.Typ {
		case emer.Inhib:
			if totGiRel > 0 {
				pj.GScale /= totGiRel
			}
		default:
			if totGeRel > 0 {
				pj.GScale /= totGeRel
			}
		}
	}
}

// DALrnFmDA returns effective learning dopamine value from given raw DA value
// applying Burst and Dip Gain factors, and then reversing sign for D2R.
// GetDa in cemer
func (ly *ModLayer) DALrnFmDA(da float32) float32 {
	if da > 0 {
		da *= ly.BurstDAGain
	} else {
		da *= ly.DipDAGain
	}
	if ly.DaRType == D2R {
		da = -da
	}
	return da
}

// Functions for rl.DALayer
func (ly *ModLayer) GetDA() float32 {
	return ly.DA
}

func (ly *ModLayer) SetDA(da float32) {
	ly.DA = da
	for ni := range ly.ModNeurs {
		mnr := &ly.ModNeurs[ni]
		mnr.DA = da
	}
}

// end rl.DALayer

func (ly *ModLayer) AvgMaxMod(_ *leabra.Time) {
	for pi := range ly.ModPools {
		mpl := &ly.ModPools[pi]
		pl := &ly.Pools[pi]
		mpl.ModNetStats.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			mnr := &ly.ModNeurs[ni]
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			mpl.ModNetStats.UpdateVal(mnr.ModNet, ni)
		}
		mpl.ModNetStats.CalcAvg()
		if mpl.ModNetStats.Max == 0 { // HACK!!!
			mpl.ModNetStats.Max = math32.SmallestNonzeroFloat32
		}
	}
}

func (ly *ModLayer) ActFmG(_ *leabra.Time) {
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
		mnr.ModAct = nrn.Act // ModAct is used in DWt. Don't want to modulate Act with DA, or things get weird very quickly
		daVal := ly.DALrnFmDA(mnr.DA)
		mnr.ModAct *= 1 + daVal
		if mnr.PVAct > 0.01 { //&& ltime.PlusPhase {
			mnr.ModAct = mnr.PVAct // this gives results that look more like the CEmer model (rather than setting Act directly from PVAct)
		}
		ly.Learn.AvgsFmAct(nrn)
	}
}
