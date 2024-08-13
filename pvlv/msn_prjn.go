// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	_ "fmt"
	"reflect"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

// TraceSyn holds extra synaptic state for trace pathways
type TraceSyn struct {

	// new trace -- drives updates to trace value -- su * (1-ru_msn) for gated, or su * ru_msn for not-gated (or for non-thalamic cases)
	NTr float32

	//  current ongoing trace of activations, which drive learning -- adds ntr and clears after learning on current values -- includes both thal gated (+ and other nongated, - inputs)
	Tr float32
}

type DALrnRule int //enums:enum

const (
	DAHebbVS DALrnRule = iota
	TraceNoThalVS
)

//////////////////////////////////////////////////////////////////////////////////////
//  MSNPath

// MSNPath does dopamine-modulated, for striatum-like layers
type MSNPath struct {
	leabra.Path
	LearningRule DALrnRule

	// special parameters for striatum trace learning
	Trace MSNTraceParams `display:"inline"`

	// trace synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIndex array
	TrSyns []TraceSyn

	// sending layer activation variable name
	SLActVar string

	// receiving layer activation variable name
	RLActVar string

	// for VS matrix TRACE_NO_THAL_VS and DA_HEBB_VS learning rules, this is the maximum value that the deep_mod_net modulatory inputs from the basal amygdala (up state enabling signal) can contribute to learning
	MaxVSActMod float32 `def:"0.7" min:"0"`

	// parameters for dopaminergic modulation
	DaMod DaModParams
}

type IMSNPath interface {
	AsMSNPath() *MSNPath
}

func (pj *MSNPath) AsMSNPath() *MSNPath {
	return pj
}

func (pj *MSNPath) Defaults() {
	pj.Trace.Defaults()
	pj.Path.Defaults()
	pj.Learn.WtSig.Gain = 1
	pj.Learn.Norm.On = false
	pj.Learn.Momentum.On = false
	pj.Learn.WtBal.On = false
	pj.MaxVSActMod = 0.5
}

func (pj *MSNPath) Build() error {
	err := pj.Path.Build()
	pj.TrSyns = make([]TraceSyn, len(pj.SConIndex))
	return err
}

func (pj *MSNPath) ClearTrace() {
	for si := range pj.TrSyns {
		sy := &pj.TrSyns[si]
		sy.Tr = 0
		sy.NTr = 0
	}
}

func (pj *MSNPath) InitWeights() {
	pj.Path.InitWeights()
	pj.ClearTrace()
}

// DWt computes the weight change (learning) -- on sending pathways.
func (pj *MSNPath) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(leabra.LeabraLayer).AsLeabra()
	rlay := pj.Recv.(*MSNLayer)
	var effLRate float32
	if rlay.Off {
		return
	}
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		snAct := sn.ActP
		nc := int(pj.SConN[si])
		st := int(pj.SConIndexSt[si])
		syns := pj.Syns[st : st+nc]
		trsyns := pj.TrSyns[st : st+nc]
		scons := pj.SConIndex[st : st+nc]

		for ci := range syns {
			sy := &syns[ci]
			trsy := &trsyns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			mn := &rlay.ModNeurs[ri]

			if rn.Off {
				continue
			}

			da, _ := mn.VarByName("DA")
			daLrn := rlay.DALrnFmDA(da)
			//rnAct := mn.ModAct // ModAct seems more correct than ActP, but doesn't match CEmer results quite as well
			rnAct := rn.ActP
			//rnAct := rn.Act
			effModLevel := mn.ModNet
			effRnAct := math32.Max(rnAct, math32.Min(effModLevel, pj.MaxVSActMod))
			rawDWt := float32(0)
			switch pj.LearningRule {
			case TraceNoThalVS:
				tr := trsy.Tr
				if mn.ModLrn == 0 {
					effLRate = pj.Learn.Lrate * pj.Trace.GateLRScale
				} else {
					effLRate = pj.Learn.Lrate
				}
				//effLRate = pj.Learn.Lrate * mn.ModLrn
				rawDWt = daLrn * tr // multiplied by learning rate below

				newNTr := pj.Trace.MSNActLrnFactor(effRnAct) * snAct
				decay := math32.Abs(newNTr) // decay is function of new trace
				if decay > 1 {
					decay = 1
				}
				//trInc := newNTr - decay*tr
				tr += newNTr - decay*tr
				trsy.Tr = tr
				trsy.NTr = newNTr
			case DAHebbVS:
				rawDWt = daLrn * effRnAct * snAct
				effLRate = pj.Learn.Lrate * mn.ModLrn
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

func (pj *MSNPath) SynValue(varNm string, sidx, ridx int) float32 {
	vidx, err := pj.SynVarIndex(varNm)
	if err != nil {
		return math32.NaN()
	}
	synIndex := pj.SynIndex(sidx, ridx)
	return pj.LeabraPrj.SynVal1D(vidx, synIndex)
}

func (pj *MSNPath) SynVarIndex(varNm string) (int, error) {
	return SynapseVarByName(varNm)
}

//func (pj *MSNPath) SynVarIndex(varNm string) (int, error) {
//	vidx, err := pj.Path.SynVarIndex(varNm)
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
//	return -1, fmt.Errorf("MatrixTracePath SynVarIndex: variable name: %v not valid", varNm)
//}

// SynVal1D returns value of given variable index (from SynVarIndex) on given SynIndex.
// Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pj *MSNPath) SynVal1D(varIndex int, synIndex int) float32 {
	if varIndex < 0 || varIndex >= len(SynapseVarsAll) {
		return math32.NaN()
	}
	nn := len(leabra.SynapseVars)
	if varIndex < nn {
		return pj.Path.SynVal1D(varIndex, synIndex)
	}
	if synIndex < 0 || synIndex >= len(pj.TrSyns) {
		return math32.NaN()
	}
	varIndex -= nn
	sy := &pj.TrSyns[synIndex]
	return sy.VarByIndex(varIndex)
}

func (ly *MSNLayer) RecvPathValues(vals *[]float32, varNm string, sendLay emer.Layer, sendIndex1D int, pathType string) error {
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
	var pj emer.Path
	if pathType != "" {
		pj, err = sendLay.RecvNameTypeTry(ly.Name, pathType)
		if pj == nil {
			pj, err = sendLay.RecvNameTry(ly.Name)
		}
	} else {
		pj, err = sendLay.RecvNameTry(ly.Name)
	}
	if pj == nil {
		return err
	}
	for ri := range ly.Neurons {
		(*vals)[ri] = pj.SynValue(varNm, sendIndex1D, ri) // this will work with any variable -- slower, but necessary
	}
	return nil
}

func (ly *MSNLayer) SendPathValues(vals *[]float32, varNm string, recvLay emer.Layer, recvIndex1D int, pathType string) error {
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
	var pj emer.Path
	if pathType != "" {
		pj, err = recvLay.SendNameTypeTry(ly.Name, pathType)
		if pj == nil {
			pj, err = recvLay.SendNameTry(ly.Name)
		}
	} else {
		pj, err = recvLay.SendNameTry(ly.Name)
	}
	if pj == nil {
		return err
	}
	for si := range ly.Neurons {
		(*vals)[si] = pj.SynValue(varNm, si, recvIndex1D)
	}
	return nil
}
