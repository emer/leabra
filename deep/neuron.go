// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"unsafe"

	"github.com/emer/leabra/leabra"
)

// deep.Neuron holds the extra neuron (unit) level variables for DeepLeabra computation.
// DeepLeabra includes both attentional and predictive learning functions of the deep layers
// and thalamocortical circuitry.
// These are maintained in a separate parallel slice from the leabra.Neuron variables.
type Neuron struct {
	ActNoAttn  float32 `desc:"non-attention modulated activation of the superficial-layer neurons -- i.e., the activation prior to any modulation by the DeepAttn modulatory signal.  Using this as a driver of Burst when there is DeepAttn modulation of superficial-layer activations prevents a positive-feedback loop that can be problematic."`
	Burst      float32 `desc:"Deep layer bursting activation values, representing activity of layer 5b intrinsic bursting (5IB) neurons, which project into the thalamus (TRC) and other deep layers locally.  Somewhat confusingly, this is computed on the Superficial layer neurons, as a thresholded function of the unit activation.  Burst is only updated during the bursting quarter(s) (typically the 4th quarter) of the alpha cycle, and it is sent via BurstCtxt projections to Deep layers (representing activation of layer 6 CT corticothalamic neurons) to drive Ctxt value there, and via BurstTRC projections to TRC layers to drive the plus-phase outcome activation (e.g., in Pulvinar) for predictive learning."`
	BurstPrv   float32 `desc:"Burst from the previous alpha trial -- this is typically used for learning in the BurstCtxt projection."`
	CtxtGe     float32 `desc:"Current excitatory conductance for temporally-delayed local integration of Burst signals sent via BurstCtxt projection into separate Deep layer neurons, which represent the activation of layer 6 CT corticothalamic neurons.  CtxtGe is updated at end of a DeepBurst quarter, and thus takes effect during subsequent quarter(s) until updated again."`
	TRCBurstGe float32 `desc:"Total excitatory conductance received from Burst activations into TRC neurons, continuously updated during the bursting quarter(s).  This drives plus-phase, outcome activation of TRC neurons."`
	BurstSent  float32 `desc:"Last Burst activation value sent, for computing TRCBurstGe using efficient delta mechanism."`
	AttnGe     float32 `desc:"Total excitatory conductance received from from deep layer activations (representing layer 6 regular spiking CT corticothalamic neurons) via DeepAttn projections.  This is sent continuously all quarters from deep layers using standard delta-based Ge computation, and drives both DeepAttn and DeepLrn values."`
	DeepAttn   float32 `desc:"DeepAttn = Min + (1-Min) * (AttnGe / MAX(AttnGe)).  This is the current attention modulatory value in Super neurons, based on inputs from deep layer 6 CT corticothalamic, regular spiking neurons that represents the net attentional filter applied to the superficial layers.  This value directly multiplies the superficial layer activations (Act) (ActNoAttn represents value prior to this multiplication).  Value is computed from AttnGe received via DeepAttn projections from Deep layers."`
	DeepLrn    float32 `desc:"DeepLrn = AttnGe / MAX(AttnGe) across layer.  This version of DeepAttn  modulates learning rates instead of activations -- learning is assumed to be more strongly affected than activation, so it lacks the positive offset that DeepAttn has."`
}

var (
	NeuronVars    = []string{"ActNoAttn", "Burst", "BurstPrv", "CtxtGe", "TRCBurstGe", "BurstSent", "AttnGe", "DeepAttn", "DeepLrn"}
	NeuronVarsMap map[string]int
	NeuronVarsAll []string
)

func init() {
	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	for i, v := range NeuronVars {
		NeuronVarsMap[v] = i
	}
	ln := len(leabra.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, leabra.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)
}

func (nrn *Neuron) VarNames() []string {
	return NeuronVars
}

// NeuronVarByName returns the index of the variable in the Neuron, or error
func NeuronVarByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return 0, fmt.Errorf("Neuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in NeuronVars list)
func (nrn *Neuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *Neuron) VarByName(varNm string) (float32, error) {
	i, err := NeuronVarByName(varNm)
	if err != nil {
		return 0, err
	}
	return nrn.VarByIndex(i), nil
}
