// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	_ "fmt"
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"reflect"
)

// TraceSyn holds extra synaptic state for trace projections
type TraceSyn struct {
	NTr float32 `desc:"new trace -- drives updates to trace value -- su * (1-ru_msn) for gated, or su * ru_msn for not-gated (or for non-thalamic cases)"`
	Tr  float32 `desc:" current ongoing trace of activations, which drive learning -- adds ntr and clears after learning on current values -- includes both thal gated (+ and other nongated, - inputs)"`
}

type DALrnRule int

const (
	DAHebbVS DALrnRule = iota
	TraceNoThalVS
	DALrnRuleN
)

var KiT_DALrnRule = kit.Enums.AddEnum(DALrnRuleN, kit.NotBitFlag, nil)

////go:generate stringer -type=DALrnRule // moved to stringers.go

//var TraceSynVars = []string{"NTr", "Tr"}

//////////////////////////////////////////////////////////////////////////////////////
//  MSNPrjn

// MSNPrjn does dopamine-modulated, for striatum-like layers
type MSNPrjn struct {
	ModHebbPrjn
	LearningRule DALrnRule
	Trace        MSNTraceParams `view:"inline" desc:"special parameters for striatum trace learning"`
	TrSyns       []TraceSyn     `desc:"trace synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIdx array"`
	ActVar       string         `desc:"activation variable name"`
	MaxVSActMod  float32        `def:"0.7" min:"0" desc:"for VS matrix TRACE_NO_THAL_VS and DA_HEBB_VS learning rules, this is the maximum value that the deep_mod_net modulatory inputs from the basal amygdala (up state enabling signal) can contribute to learning"`
	DebugVal     int
}

type IMSNPrjn interface {
	AsMSNPrjn() *MSNPrjn
}

func (pj *MSNPrjn) AsMSNPrjn() *MSNPrjn {
	return pj
}

func (pj *MSNPrjn) AsModPrjn() *ModHebbPrjn {
	return &pj.ModHebbPrjn
}

func (pj *MSNPrjn) Defaults() {
	pj.Trace.Defaults()
	pj.Prjn.Defaults()
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
	pj.ActVar = "ActP"
	pj.DebugVal = -1
}

func (pj *MSNPrjn) Build() error {
	err := pj.Prjn.Build()
	pj.TrSyns = make([]TraceSyn, len(pj.SConIdx))
	return err
}

func (pj *MSNPrjn) ClearTrace() {
	for si := range pj.TrSyns {
		sy := &pj.TrSyns[si]
		sy.Tr = 0
	}
}

func (pj *MSNPrjn) InitWts() {
	pj.Prjn.InitWts()
	pj.ClearTrace()
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *MSNPrjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(*MSNLayer)
	var effLRate float32
	sActIdx, _ := slay.UnitVarIdx(pj.ActVar)
	rActIdx, _ := rlay.UnitVarIdx(pj.ActVar)
	//rlay := rlayi.AsMSNLayer() // note: won't work if derived
	if rlay.IsOff() {
		return
	}
	for si := range slay.Neurons {
		snAct := slay.UnitVal1D(sActIdx, si)
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		trsyns := pj.TrSyns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			trsy := &trsyns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			mn := &rlay.ModNeurs[ri]

			if rn.IsOff() {
				continue
			}

			// dwt += lRateEff * da * rn.Act * sn.Act
			da, _ := mn.VarByName("DA")
			daLrn := rlay.DALrnFmDA(da)
			// ACh mod should be Q4 only. For now, just not doing it
			//ach := rlay.UnitValByIdx(ACh, int(ri))
			//rnAct, _ := mn.VarByName(pj.ActVar)
			//rnAct := mn.ModAct
			//rnAct := rn.ActP
			rnAct := rlay.UnitVal1D(rActIdx, int(ri))
			effModLevel := mn.ModNet
			//if effModLevel < 0.1 {
			//	effModLevel = 0
			//}
			effRnAct := math32.Max(rnAct, math32.Min(effModLevel, pj.MaxVSActMod))
			rawDWt := float32(0)
			switch pj.LearningRule {
			case TraceNoThalVS:
				tr := trsy.Tr
				effLRate = pj.Learn.Lrate
				rawDWt = daLrn * tr // multiplied by learning rate below

				//if ach >= pj.Trace.AChResetThr { tr = 0 }

				newNTr := pj.Trace.MSNActLrnFactor(effRnAct) * snAct
				decay := math32.Abs(newNTr) // decay is function of new trace
				if decay > 1 {
					decay = 1
				}
				trInc := newNTr - decay*tr
				oldTr := tr
				tr += trInc
				if rlay.DebugVal > 0 && ri == 0 && math32.Abs(rawDWt) > 0.1 { // || newNTr > 1e-9) {
					fmt.Printf("%v[%v]->%v[%v]: rnAct=%8f, mln=%8f, effRnAct=%8f, snAct=%8f, da=%8f, daLrn=%8f, oldTr=%8f, trInc=%8f, newNTr=%8f, lr=%8f, rawDWt=%8f\n",
						slay.Name(), si, rlay.Name(), ri, rnAct, effModLevel, effRnAct, snAct, da, daLrn, oldTr, trInc, newNTr, effLRate, rawDWt)
				}
				trsy.Tr = tr
				trsy.NTr = newNTr
			case DAHebbVS:
				rawDWt = daLrn * effRnAct * snAct
				effLRate = pj.Learn.Lrate * mn.ModLrn
				if rlay.DebugVal > 0 && rawDWt != 0 {
					fmt.Printf("%v[%v]->%v[%v]: rnAct=%v, mln=%v, effRnAct=%v, snAct=%v, da=%v, delta=%v, lr=%v, prevDWt=%v\n",
						slay.Name(), si, rlay.Name(), ri, rnAct, effModLevel, effRnAct, snAct, da, rawDWt, pj.Learn.Lrate, sy.DWt)
				}
			}
			sy.DWt += effLRate * rawDWt
		}
	}
}

var (
	TraceVars       = []string{"NTr", "Tr"}
	SynapseVarProps = map[string]string{
		"NTr": `auto-scale:"+"`,
		"Tr":  `auto-scale:"+"`,
	}
	TraceVarsMap   map[string]int
	SynapseVarsAll []string
)

func init() {
	TraceVarsMap = make(map[string]int, len(TraceVars)+len(leabra.SynapseVars))
	for i, v := range leabra.SynapseVars {
		TraceVarsMap[v] = i
	}
	for i, v := range TraceVars {
		TraceVarsMap[v] = i + len(leabra.SynapseVars)
	}
	for k, v := range leabra.SynapseVarProps {
		SynapseVarProps[k] = v
	}
	ln := len(leabra.SynapseVars)
	SynapseVarsAll = make([]string, len(TraceVars)+ln)
	copy(SynapseVarsAll, leabra.SynapseVars)
	copy(SynapseVarsAll[ln:], TraceVars)
}

func (tr *TraceSyn) VarNames() []string {
	return TraceVars
}

// copied from SynapseVarByName
func TraceVarByName(varNm string) (int, error) {
	i, ok := TraceVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("synapse VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in SynapseVars list)
func (tr *TraceSyn) VarByIndex(idx int) float32 {
	// todo: would be ideal to avoid having to use reflect here..
	v := reflect.ValueOf(*tr)
	return v.Field(idx).Interface().(float32)
}

// VarByName returns variable by name, or error
func SynapseVarByName(varNm string) (int, error) {
	i, ok := TraceVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("synapse VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

func (tr *TraceSyn) SetVarByIndex(idx int, val float32) {
	// todo: would be ideal to avoid having to use reflect here..
	v := reflect.ValueOf(tr)
	v.Elem().Field(idx).SetFloat(float64(val))
}

// SetVarByName sets synapse variable to given value
func (tr *TraceSyn) SetVarByName(varNm string, val float32) error {
	i, err := TraceVarByName(varNm)
	if err != nil {
		return err
	}
	tr.SetVarByIndex(i, val)
	return nil
}

func (pj *MSNPrjn) SynVal(varNm string, sidx, ridx int) float32 {
	vidx, err := pj.SynVarIdx(varNm)
	if err != nil {
		return math32.NaN()
	}
	synIdx := pj.SynIdx(sidx, ridx)
	return pj.LeabraPrj.SynVal1D(vidx, synIdx)
}

func (pj *MSNPrjn) SynVarIdx(varNm string) (int, error) {
	return SynapseVarByName(varNm)
}

//func (pj *MSNPrjn) SynVarIdx(varNm string) (int, error) {
//	vidx, err := pj.Prjn.SynVarIdx(varNm)
//	if err == nil {
//		return vidx, err
//	}
//	nn := len(leabra.SynapseVars)
//	switch varNm {
//	case "NTr":
//		return nn, nil
//	case "Tr":
//		return nn + 1, nil
//	}
//	return -1, fmt.Errorf("MatrixTracePrjn SynVarIdx: variable name: %v not valid", varNm)
//}

// SynVal1D returns value of given variable index (from SynVarIdx) on given SynIdx.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pj *MSNPrjn) SynVal1D(varIdx int, synIdx int) float32 {
	if varIdx < 0 || varIdx >= len(SynapseVarsAll) {
		return math32.NaN()
	}
	nn := len(leabra.SynapseVars)
	if varIdx < nn {
		return pj.Prjn.SynVal1D(varIdx, synIdx)
	}
	if synIdx < 0 || synIdx >= len(pj.TrSyns) {
		return math32.NaN()
	}
	varIdx -= nn
	sy := &pj.TrSyns[synIdx]
	return sy.VarByIndex(varIdx)
}

func (ly *MSNLayer) RecvPrjnVals(vals *[]float32, varNm string, sendLay emer.Layer, sendIdx1D int, prjnType string) error {
	var err error
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := math32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if sendLay == nil {
		return fmt.Errorf("sending layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = sendLay.SendPrjns().RecvNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
		}
	} else {
		pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	for ri := range ly.Neurons {
		(*vals)[ri] = pj.SynVal(varNm, sendIdx1D, ri) // this will work with any variable -- slower, but necessary
	}
	return nil
}

func (ly *MSNLayer) SendPrjnVals(vals *[]float32, varNm string, recvLay emer.Layer, recvIdx1D int, prjnType string) error {
	var err error
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := math32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if recvLay == nil {
		return fmt.Errorf("receiving layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = recvLay.RecvPrjns().SendNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
		}
	} else {
		pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	for si := range ly.Neurons {
		(*vals)[si] = pj.SynVal(varNm, si, recvIdx1D)
	}
	return nil
}
