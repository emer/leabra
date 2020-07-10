// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"fmt"
	"log"
	"strings"
	"unsafe"

	"github.com/chewxy/math32"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// CaParams control the calcium dynamics in STN neurons.
// Gillies & Willshaw, 2006 provide a biophysically detailed simulation,
// and we use their logistic function for computing KCa conductance based on Ca,
// but we use a simpler approximation with burst and act threshold.
// KCa are Calcium-gated potassium channels that drive the long
// afterhyperpolarization of STN neurons.  Auto reset at each AlphaCycle.
// The conductance is applied to KNa channels to take advantage
// of the existing infrastructure.
type CaParams struct {
	BurstThr float32 `def:"0.9" desc:"activation threshold for bursting that drives strong influx of Ca to turn on KCa channels -- there is a complex de-inactivation dynamic involving the volley of excitation and inhibition from GPe, but we can just use a threshold"`
	ActThr   float32 `def:"0.7" desc:"activation threshold for increment in activation above baseline that drives lower influx of Ca"`
	BurstCa  float32 `desc:"Ca level for burst level activation"`
	ActCa    float32 `def:"0.2" desc:"Ca increment from regular sub-burst activation -- drives slower inhibition of firing over time -- for stop-type STN dynamics that initially put hold on GPi and then decay"`
	GbarKCa  float32 `def:"20" desc:"maximal KCa conductance (actual conductance is applied to KNa channels)"`
	KCaTau   float32 `def:"40" desc:"KCa conductance time constant -- 40 from Gillies & Willshaw, 2006"`
	CaTau    float32 `def:"185.7" desc:"Ca time constant of decay to baseline -- 185.7 from Gillies & Willshaw, 2006"`
}

func (kc *CaParams) Defaults() {
	kc.BurstThr = 0.9
	kc.ActThr = 0.7
	kc.BurstCa = 200
	kc.ActCa = 0.2
	kc.GbarKCa = 20
	kc.KCaTau = 40
	kc.CaTau = 185.7
}

// KCaGFmCa returns the driving conductance for KCa channels based on given Ca level.
// This equation comes from Gillies & Willshaw, 2006.
func (kc *CaParams) KCaGFmCa(ca float32) float32 {
	return 0.81 / (1 + math32.Exp(-(math32.Log(ca)+0.3))/0.46)
}

///////////////////////////////////////////////////////////////////////////
// STNLayer

// STNLayer represents the pausing subtype of STN neurons.
// These open the gating window.
type STNLayer struct {
	leabra.Layer
	Ca       CaParams    `view:"inline" desc:"parameters for calcium and calcium-gated potassium channels that drive the afterhyperpolarization that open the gating window in STN neurons (Hallworth et al., 2003)"`
	DA       float32     `inactive:"+" desc:"dopamine value for this layer"`
	STNNeurs []STNNeuron `desc:"slice of extra STNNeuron state for this layer -- flat list of len = Shape.Len(). You must iterate over index and use pointer to modify values."`
}

var KiT_STNLayer = kit.Types.AddType(&STNLayer{}, leabra.LayerProps)

// Defaults in param.Sheet format
// Sel: "STNLayer", Desc: "defaults",
// 	Params: params.Params{
// 		"Layer.Act.Init.Vm":   "0.9",
// 		"Layer.Act.Init.Act":  "0.5",
// 		"Layer.Act.Erev.L":    "0.8",
// 		"Layer.Act.Gbar.L":    "0.3",
// 		"Layer.Inhib.Layer.On":     "false",
// 		"Layer.Inhib.ActAvg.Init":  "0.25",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 		"Layer.Inhib.Self.On":      "true",
// 		"Layer.Inhib.Self.Gi":      "0.4",
// 		"Layer.Inhib.Self.Tau":     "3.0",
// 		"Layer.Act.XX1.Gain":       "20", // more graded -- still works with 40 but less Rt distrib
// 		"Layer.Act.Dt.VmTau":       "3.3",
// 		"Layer.Act.Dt.GTau":        "3",
// 		"Layer.Act.Init.Decay":     "0",
// }}

func (ly *STNLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Ca.Defaults()
	ly.DA = 0

	// STN is tonically self-active and has no FFFB inhibition

	ly.Act.Init.Vm = 0.9
	ly.Act.Init.Act = 0.5
	ly.Act.Erev.L = 0.8
	ly.Act.Gbar.L = 0.3
	ly.Inhib.Layer.On = false
	ly.Inhib.Pool.On = false
	ly.Inhib.Self.On = true
	ly.Inhib.Self.Gi = 0.4 // 0.4 in localist one
	ly.Inhib.ActAvg.Init = 0.25
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.Self.Tau = 3.0
	ly.Act.XX1.Gain = 20 // more graded -- still works with 40 but less Rt distrib
	ly.Act.Dt.VmTau = 3.3
	ly.Act.Dt.GTau = 3 // fastest
	ly.Act.Init.Decay = 0

	for _, pji := range ly.RcvPrjns {
		pj := pji.(leabra.LeabraPrjn).AsLeabra()
		pj.Learn.Learn = false
		pj.Learn.WtSig.Gain = 1
		pj.WtInit.Mean = 0.9
		pj.WtInit.Var = 0
		pj.WtInit.Sym = false
		if strings.HasSuffix(ly.Nm, "STNp") {
			if _, ok := pj.Send.(*GPLayer); ok { // GPeIn -- others are PFC, 1.5 in orig
				pj.WtScale.Abs = 0.1
			}
		} else { // STNs
			if _, ok := pj.Send.(*GPLayer); ok { // GPeIn -- others are PFC, 1.5 in orig
				pj.WtScale.Abs = 0.1
			} else {
				pj.WtScale.Abs = 0.2 // weaker inputs
			}
		}
	}

	ly.UpdateParams()
}

// DALayer interface:

func (ly *STNLayer) GetDA() float32   { return ly.DA }
func (ly *STNLayer) SetDA(da float32) { ly.DA = da }

/*
// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *STNLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	switch vidx {
	case DA:
		return ly.DA
	}
	return 0
}
*/

func (ly *STNLayer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
// should already have presented the external input to the network at this point.
func (ly *STNLayer) AlphaCycInit() {
	ly.Layer.AlphaCycInit()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.Gk = 0
		snr := &ly.STNNeurs[ni]
		snr.Ca = 0
		snr.KCa = 0
	}
}

func (ly *STNLayer) ActFmG(ltime *leabra.Time) {
	for ni := range ly.Neurons { // note: copied from leabra ActFmG, not calling it..
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)

		snr := &ly.STNNeurs[ni]
		snr.KCa += (ly.Ca.KCaGFmCa(snr.Ca) - snr.KCa) / ly.Ca.KCaTau
		dCa := -snr.Ca / ly.Ca.CaTau
		if nrn.Act >= ly.Ca.BurstThr {
			dCa += ly.Ca.BurstCa
			snr.KCa = 1 // burst this too
		} else if nrn.Act >= ly.Ca.ActThr {
			dCa += (nrn.Act - ly.Ca.ActThr) * ly.Ca.ActCa
		}
		snr.Ca += dCa
		nrn.Gk = ly.Ca.GbarKCa * snr.KCa

		ly.Learn.AvgsFmAct(nrn)
	}
}

//////////////////////////////////////////////////////////////////////
// STNNeurs management

// UnitVals fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *STNLayer) UnitVals(vals *[]float32, varNm string) error {
	vidx, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitVals(vals, varNm)
	}
	vidx, err = STNNeuronVarByName(varNm)
	if err != nil {
		return err
	}
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	for i := range ly.STNNeurs {
		dnr := &ly.STNNeurs[i]
		(*vals)[i] = dnr.VarByIndex(vidx)
	}
	return nil
}

// UnitValsTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *STNLayer) UnitValsTensor(tsr etensor.Tensor, varNm string) error {
	if tsr == nil {
		err := fmt.Errorf("leabra.UnitValsTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	vidx, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitValsTensor(tsr, varNm)
	}
	vidx, err = STNNeuronVarByName(varNm)
	if err != nil {
		return err
	}
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	for i := range ly.STNNeurs {
		dnr := &ly.STNNeurs[i]
		tsr.SetFloat1D(i, float64(dnr.VarByIndex(vidx)))
	}
	return nil
}

// UnitValTry returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *STNLayer) UnitValTry(varNm string, idx []int) (float32, error) {
	_, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitValTry(varNm, idx)
	}
	fidx := ly.Shp.Offset(idx)
	nn := len(ly.STNNeurs)
	if fidx < 0 || fidx >= nn {
		return 0, fmt.Errorf("STNLayer UnitVal index: %v out of range, N = %v", fidx, nn)
	}
	dnr := &ly.STNNeurs[fidx]
	return dnr.VarByName(varNm)
}

// UnitVal1DTry returns value of given variable name on given unit,
// using 1-dimensional index.
func (ly *STNLayer) UnitVal1DTry(varNm string, idx int) (float32, error) {
	_, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitVal1DTry(varNm, idx)
	}
	nn := len(ly.STNNeurs)
	if idx < 0 || idx >= nn {
		return 0, fmt.Errorf("STNLayer UnitVal1D index: %v out of range, N = %v", idx, nn)
	}
	dnr := &ly.STNNeurs[idx]
	return dnr.VarByName(varNm)
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *STNLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.STNNeurs = make([]STNNeuron, len(ly.Neurons))
	return nil
}

//////////////////////////////////////////////////////////////////////
// STN neurons

// STNNeuron holds the extra neuron (unit) level variables for STN computation.
type STNNeuron struct {
	Ca  float32 `desc:"intracellular Calcium concentration -- increased by bursting and elevated levels of activation, drives KCa currents that result in hyperpolarization / inhibition."`
	KCa float32 `desc:"Calcium-gated potassium channel conductance level, computed using function from gillies & Willshaw 2006 as function of Ca."`
}

var (
	STNNeuronVars    = []string{"Ca", "KCa"}
	STNNeuronVarsMap map[string]int
)

func init() {
	STNNeuronVarsMap = make(map[string]int, len(STNNeuronVars))
	for i, v := range STNNeuronVars {
		STNNeuronVarsMap[v] = i
	}
}

func (nrn *STNNeuron) VarNames() []string {
	return STNNeuronVars
}

// STNNeuronVarByName returns the index of the variable in the STNNeuron, or error
func STNNeuronVarByName(varNm string) (int, error) {
	i, ok := STNNeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("STNNeuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in STNNeuronVars list)
func (nrn *STNNeuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *STNNeuron) VarByName(varNm string) (float32, error) {
	i, err := STNNeuronVarByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}
