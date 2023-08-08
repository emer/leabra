// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// layers that receive modulatory projections--(original code from SendDeepMod in cemer)

package pvlv

import (
	"fmt"
	"strconv"

	"github.com/emer/emergent/emer"
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

type IModLayer interface {
	AsMod() *ModLayer
}

type AvgMaxModLayer interface {
	AvgMaxMod(*leabra.Time)
}

// ModSender has methods for sending modulation, and setting the value to be sent.
type ModSender interface {
	SendMods(ltime *leabra.Time)
	ModSendValue(ni int32) float32
}

// ModReceiver has one method to integrate incoming modulation, and another
type ModReceiver interface {
	ReceiveMods(sender ModSender, scale float32) // copy incoming modulation values into the layer's own ModNet variable
	ModsFmInc(ltime *leabra.Time)                // set modulation levels
}

// ModLayer is a layer that RECEIVES modulatory input
type ModLayer struct {
	leabra.Layer

	// neuron-level modulation state
	ModNeurs []ModNeuron `desc:"neuron-level modulation state"`

	// pools for maintaining aggregate values
	ModPools []ModPool `desc:"pools for maintaining aggregate values"`

	// layer names and scale values for mods sent from this layer
	ModReceivers []ModRcvrParams `desc:"layer names and scale values for mods sent from this layer"`

	// parameters shared by all modulator receiver layers
	ModParams `desc:"parameters shared by all modulator receiver layers"`

	// parameters for dopaminergic modulation
	DaMod DaModParams `desc:"parameters for dopaminergic modulation"`

	// layer-level neuromodulator levels
	Modulators `desc:"layer-level neuromodulator levels"`
}

var _ IModLayer = (*ModLayer)(nil)

var KiT_ModLayer = kit.Types.AddType(&ModLayer{}, nil)

// ModPool is similar to a standard Pool structure, and uses the same code to compute running statistics.
type ModPool struct {
	ModNetStats minmax.AvgMax32

	// modulation level transmitted to receiver layers
	ModSent float32 `desc:"modulation level transmitted to receiver layers"`

	// threshold for sending modulation. values below this are not added to the pool-level total
	ModSendThreshold float32 `desc:"threshold for sending modulation. values below this are not added to the pool-level total"`
}

// DaModParams specifies parameters shared by all layers that receive dopaminergic modulatory input.
type DaModParams struct {

	// whether to use dopamine modulation
	On bool `desc:"whether to use dopamine modulation"`

	// dopamine receptor type, D1 or D2
	RecepType DaRType `inactive:"+" desc:"dopamine receptor type, D1 or D2"`

	// multiplicative gain factor applied to positive dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign!
	BurstGain float32 `desc:"multiplicative gain factor applied to positive dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign!"`

	// multiplicative gain factor applied to negative dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign! should be small for acq, but roughly equal to burst_da_gain for ext
	DipGain float32 `desc:"multiplicative gain factor applied to negative dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign! should be small for acq, but roughly equal to burst_da_gain for ext"`
}

// ModParams contains values that control a receiving layer's response to modulatory inputs
type ModParams struct {

	// [viewif: On] how much to multiply Da in the minus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons
	Minus float32 `viewif:"On" desc:"how much to multiply Da in the minus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`

	// [viewif: On] how much to multiply Da in the plus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons
	Plus float32 `viewif:"On" desc:"how much to multiply Da in the plus phase to add to Ge input -- use negative values for NoGo/indirect pathway/D2 type neurons"`

	// [viewif: DaMod.On&&ModGain] for negative dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * NegNain) -- da is multiplied by minus or plus depending on phase
	NegGain float32 `viewif:"DaMod.On&&ModGain" desc:"for negative dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * NegNain) -- da is multiplied by minus or plus depending on phase"`

	// [viewif: DaMod.On&&ModGain] for positive dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * PosGain) -- da is multiplied by minus or plus depending on phase
	PosGain float32 `viewif:"DaMod.On&&ModGain" desc:"for positive dopamine, how much to change the default gain value as a function of dopamine: gain = gain * (1 + da * PosGain) -- da is multiplied by minus or plus depending on phase"`

	// for modulation coming from the BLA via deep_mod_net -- when this modulation signal is below zero, does it have the ability to zero out the patch activations?  i.e., is the modulation required to enable patch firing?
	ActModZero bool `desc:"for modulation coming from the BLA via deep_mod_net -- when this modulation signal is below zero, does it have the ability to zero out the patch activations?  i.e., is the modulation required to enable patch firing?"`

	// threshold on deep_mod_net before deep mod is applied -- if not receiving even this amount of overall input from deep_mod sender, then do not use the deep_mod_net to drive deep_mod and deep_lrn values -- only for SUPER units -- based on LAYER level maximum for base LeabraLayerSpec, PVLV classes are based on actual deep_mod_net for each unit
	ModNetThreshold float32 `desc:"threshold on deep_mod_net before deep mod is applied -- if not receiving even this amount of overall input from deep_mod sender, then do not use the deep_mod_net to drive deep_mod and deep_lrn values -- only for SUPER units -- based on LAYER level maximum for base LeabraLayerSpec, PVLV classes are based on actual deep_mod_net for each unit"`

	// threshold for including neuron activation in total to send (for ModNet)
	ModSendThreshold float32 `desc:"threshold for including neuron activation in total to send (for ModNet)"`

	// does this layer send modulation to other layers?
	IsModSender bool `desc:"does this layer send modulation to other layers?"`

	// does this layer receive modulation from other layers?
	IsModReceiver bool `desc:"does this layer receive modulation from other layers?"`

	// does this layer receive a direct PV input?
	IsPVReceiver bool `desc:"does this layer receive a direct PV input?"`
}

// ModRcvrParams specifies the name of a layer that receives modulatory input, and a scale factor--critical for inputs from
// large layers such as BLA.
type ModRcvrParams struct {

	// name of receiving layer
	RcvName string `desc:"name of receiving layer"`

	// scale factor for modulation to this receiver
	Scale float32 `desc:"scale factor for modulation to this receiver"`
}

var KiT_ModParams = kit.Types.AddType(&ModParams{}, nil)

// Modulators are modulatory neurotransmitters. Currently ACh and SE are only placeholders.
type Modulators struct {

	// current dopamine level for this layer
	DA float32 `desc:"current dopamine level for this layer"`

	// current acetylcholine level for this layer
	ACh float32 `desc:"current acetylcholine level for this layer"`

	// current serotonin level for this layer
	SE float32 `desc:"current serotonin level for this layer"`
}

var KiT_Modulators = kit.Types.AddType(&Modulators{}, nil)

// ModNeuron encapsulates the variables used by all layers that receive modulatory input
type ModNeuron struct {

	// neuron-level modulator activation
	Modulators `desc:"neuron-level modulator activation"`

	// activity level for modulation
	ModAct float32 `desc:"activity level for modulation"`

	// degree of full modulation to apply
	ModLevel float32 `desc:"degree of full modulation to apply"`

	// modulation input from sender
	ModNet float32 `desc:"modulation input from sender"`

	// multiplier for DA modulation of learning rate
	ModLrn float32 `desc:"multiplier for DA modulation of learning rate"`

	// direct activation from US
	PVAct float32 `desc:"direct activation from US"`
}

var KiT_ModNeuron = kit.Types.AddType(&ModNeuron{}, nil)

// AsMod returns a pointer to the ModLayer portion of the layer
func (ly *ModLayer) AsMod() *ModLayer {
	return ly
}

// AsLeabra gets a pointer to the generic Leabra portion of the layer
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

// Dopamine receptor type, for D1R and D2R dopamine receptors
type DaRType int

const (
	// D1R: primarily expresses Dopamine D1 Receptors -- dopamine is excitatory and bursts of dopamine lead to increases in synaptic weight, while dips lead to decreases -- direct pathway in dorsal striatum
	D1R DaRType = iota

	// D2R: primarily expresses Dopamine D2 Receptors -- dopamine is inhibitory and bursts of dopamine lead to decreases in synaptic weight, while dips lead to increases -- indirect pathway in dorsal striatum
	D2R

	DaRTypeN
)

var KiT_DaRType = kit.Enums.AddEnum(DaRTypeN, kit.NotBitFlag, nil)

// GetMonitorVal retrieves a value for a trace of some quantity, possibly more than just a variable
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
		return mat32.NaN()
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
		return mat32.NaN()
	}
	if varIdx < 0 || varIdx >= len(ModNeuronVarsAll) {
		return mat32.NaN()
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

// UpdateParams passes on an UpdateParams call to the layer's underlying Leabra layer.
func (ly *ModLayer) UpdateParams() {
	ly.Layer.UpdateParams()
}

// InitActs sets modulation state variables to their default values for a layer, including its pools.
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

// InitActs zeroes activation levels for a set of modulator variables.
func (ml *Modulators) InitActs() {
	ml.ACh = 0
	ml.DA = 0
	ml.SE = 0
}

// InitActs sets modulation state variables to their default values for one neuron.
func (mnr *ModNeuron) InitActs() {
	mnr.ModAct = 0
	mnr.ModLevel = 1
	mnr.ModNet = 0
	mnr.ModLrn = 1
	mnr.PVAct = 0
	mnr.Modulators.InitActs()
}

// ClearModLevels resets modulation state variables to their default values for an entire layer.
func (ly *ModLayer) ClearModLevels() {
	for ni := range ly.ModNeurs {
		ly.ModNeurs[ni].InitActs()
	}
}

// AddModReceiver adds a receiving layer to the list of modulatory target layers for a sending layer.
func (ly *ModLayer) AddModReceiver(rcvr ModReceiver, scale float32) {
	ly.IsModSender = true
	rly := rcvr.(IModLayer).AsMod()
	rly.IsModReceiver = true
	ly.ModReceivers = append(ly.ModReceivers, ModRcvrParams{rly.Name(), scale})
}

// ModSendValue returns the value of ModSent for one modulatory pool, specified by ni.
func (ly *ModLayer) ModSendValue(ni int32) float32 {
	return ly.ModPools[ni].ModSent
}

// SendMods calculates the level of modulation to send to receivers, based on subpool activations, and calls
// ReceiveMods for the receivers to process sent values.
func (ly *ModLayer) SendMods(_ *leabra.Time) {
	for pi := range ly.ModPools {
		mpl := &ly.ModPools[pi]
		mpl.ModSent = 0
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mpl := &ly.ModPools[nrn.SubPool]
		if mat32.Abs(nrn.Act) > ly.Act.OptThresh.Send {
			mpl.ModSent += nrn.Act
		}
	}
	for _, mr := range ly.ModReceivers {
		rl := ly.Network.LayerByName(mr.RcvName).(ModReceiver)
		rl.ReceiveMods(ly, mr.Scale)
	}
}

// ReceiveMods computes ModNet, based on the value from the sender, times a scale value.
func (ly *ModLayer) ReceiveMods(sender ModSender, scale float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mnr := &ly.ModNeurs[ni]
		modVal := sender.ModSendValue(nrn.SubPool)
		//if ly.Debug > 0 && ni == 1 && modVal != 0 && ly.Nm == "VSMatrixNegD2" {
		//	fmt.Printf("%s: modVal:%f, Modnet:%f, scale:%f\n", ly.Nm, modVal, mnr.ModNet, scale)
		//}
		mnr.ModNet = modVal * scale
	}
}

// ModsFmInc sets ModLrn and ModLevel based on individual neuron activation and incoming ModNet values.
//
// If ModNet is below threshold, ModLrn is set to 0, and ModLevel is set to either 0 or 1 depending on the value of the
// ModNetThreshold parameter.
//
// If ModNet is above threshold, ModLrn for each neuron is set to the ratio of its ModNet input to its subpool
// activation value, with special cases for extreme values.
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
		} else {
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

// ClearModActs clears modulatory activation values. This is critical for getting clean results from one trial to
// the next.
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

// GScaleFmAvgAct sets the value of GScale on incoming projections, based on sending layer subpool activations.
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
		da *= ly.DaMod.BurstGain
	} else {
		da *= ly.DaMod.DipGain
	}
	if ly.DaMod.RecepType == D2R {
		da = -da
	}
	return da
}

// Functions for rl.DALayer

// GetDA returns the level of dopaminergic activation for an entire layer.
func (ly *ModLayer) GetDA() float32 {
	return ly.DA
}

// SetDA sets the level of dopaminergic activation for an entire layer.
func (ly *ModLayer) SetDA(da float32) {
	ly.DA = da
	for ni := range ly.ModNeurs {
		mnr := &ly.ModNeurs[ni]
		mnr.DA = da
	}
}

// end rl.DALayer

// AvgMaxMod runs the standard activation statistics calculation as used for other pools on a layer's ModPools.
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
			mpl.ModNetStats.UpdateVal(mnr.ModNet, int32(ni))
		}
		mpl.ModNetStats.CalcAvg()
		if mpl.ModNetStats.Max == 0 { // HACK!!!
			mpl.ModNetStats.Max = mat32.SmallestNonzeroFloat32
		}
	}
}

// ActFmG calculates activation from net input, applying modulation values.
func (ly *ModLayer) ActFmG(_ *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		mnr := &ly.ModNeurs[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		if ly.IsModReceiver {
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
