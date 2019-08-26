// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"reflect"
)

// leabra.Synapse holds state for the synaptic connection between neurons
type Synapse struct {
	Wt     float32 `desc:"synaptic weight value -- sigmoid contrast-enhanced"`
	LWt    float32 `desc:"linear (underlying) weight value -- learns according to the lrate specified in the connection spec -- this is converted into the effective weight value, Wt, via sigmoidal contrast enhancement (see WtSigParams)"`
	DWt    float32 `desc:"change in synaptic weight, from learning"`
	Norm   float32 `desc:"DWt normalization factor -- reset to max of abs value of DWt, decays slowly down over time -- serves as an estimate of variance in weight changes over time"`
	Moment float32 `desc:"momentum -- time-integrated DWt changes, to accumulate a consistent direction of weight change and cancel out dithering contradictory changes"`
	Scale  float32 `desc:"scaling parameter for this connection: effective weight value is scaled by this factor -- useful for topographic connectivity patterns e.g., to enforce more distant connections to always be lower in magnitude than closer connections.  Value defaults to 1 (cannot be exactly 0 -- otherwise is automatically reset to 1 -- use a very small number to approximate 0).  Typically set by using the prjn.Pattern Weights() values where appropriate"`
}

var SynapseVars = []string{"Wt", "LWt", "DWt", "Norm", "Moment", "Scale"}

var SynapseVarProps = map[string]string{
	"DWt":    `auto-scale:"+"`,
	"Moment": `auto-scale:"+"`,
}

var SynapseVarsMap map[string]int

func init() {
	SynapseVarsMap = make(map[string]int, len(SynapseVars))
	for i, v := range SynapseVars {
		SynapseVarsMap[v] = i
	}
}

func (sy *Synapse) VarNames() []string {
	return SynapseVars
}

// SynapseVarByName returns the index of the variable in the Synapse, or error
func SynapseVarByName(varNm string) (int, error) {
	i, ok := SynapseVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("Synapse VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in SynapseVars list)
func (sy *Synapse) VarByIndex(idx int) float32 {
	// todo: would be ideal to avoid having to use reflect here..
	v := reflect.ValueOf(*sy)
	return v.Field(idx).Interface().(float32)
}

// VarByName returns variable by name, or error
func (sy *Synapse) VarByName(varNm string) (float32, error) {
	i, err := SynapseVarByName(varNm)
	if err != nil {
		return 0, err
	}
	return sy.VarByIndex(i), nil
}

func (sy *Synapse) SetVarByIndex(idx int, val float32) {
	// todo: would be ideal to avoid having to use reflect here..
	v := reflect.ValueOf(sy)
	v.Elem().Field(idx).SetFloat(float64(val))
}

// SetVarByName sets synapse variable to given value
func (sy *Synapse) SetVarByName(varNm string, val float32) error {
	i, err := SynapseVarByName(varNm)
	if err != nil {
		return err
	}
	sy.SetVarByIndex(i, val)
	return nil
}
