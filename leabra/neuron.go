// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"strings"
	"unsafe"

	"cogentcore.org/core/enums"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/types"
	"github.com/emer/emergent/v2/emer"
)

// NeuronVarStart is the byte offset of fields in the Neuron structure
// where the float32 named variables start.
// Note: all non-float32 infrastructure variables must be at the start!
const NeuronVarStart = 12

// leabra.Neuron holds all of the neuron (unit) level variables -- this is the most basic version with
// rate-code only and no optional features at all.
// All variables accessible via Unit interface must be float32 and start at the top, in contiguous order
type Neuron struct {

	// bit flags for binary state variables
	Flags NeurFlags

	// index of the sub-level inhibitory pool that this neuron is in (only for 4D shapes, the pool (unit-group / hypercolumn) structure level) -- indicies start at 1 -- 0 is layer-level pool (is 0 if no sub-pools).
	SubPool int32

	////////////////////////////
	// Act

	// rate-coded activation value reflecting final output of neuron communicated to other neurons, typically in range 0-1.  This value includes adaptation and synaptic depression / facilitation effects which produce temporal contrast (see ActLrn for version without this).  For rate-code activation, this is noisy-x-over-x-plus-one (NXX1) function; for discrete spiking it is computed from the inverse of the inter-spike interval (ISI), and Spike reflects the discrete spikes.
	Act float32

	// total excitatory synaptic conductance -- the net excitatory input to the neuron -- does *not* include Gbar.E
	Ge float32

	// total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I
	Gi float32

	// total potassium conductance, typically reflecting sodium-gated potassium currents involved in adaptation effects -- does *not* include Gbar.K
	Gk float32

	// net current produced by all channels -- drives update of Vm
	Inet float32

	// membrane potential -- integrates Inet current over time
	Vm float32

	// noise value added to unit (ActNoiseParams determines distribution, and when / where it is added)
	Noise float32

	// whether neuron has spiked or not (0 or 1), for discrete spiking neurons.
	Spike float32

	// target value: drives learning to produce this activation value
	Targ float32

	// external input: drives activation of unit from outside influences (e.g., sensory input)
	Ext float32

	////////////////////////////
	// Learn

	// super-short time-scale average of ActLrn activation -- provides the lowest-level time integration -- for spiking this integrates over spikes before subsequent averaging, and it is also useful for rate-code to provide a longer time integral overall
	AvgSS float32

	// short time-scale average of ActLrn activation -- tracks the most recent activation states (integrates over AvgSS values), and represents the plus phase for learning in XCAL algorithms
	AvgS float32

	// medium time-scale average of ActLrn activation -- integrates over AvgS values, and represents the minus phase for learning in XCAL algorithms
	AvgM float32

	// long time-scale average of medium-time scale (trial level) activation, used for the BCM-style floating threshold in XCAL
	AvgL float32

	// how much to learn based on the long-term floating threshold (AvgL) for BCM-style Hebbian learning -- is modulated by level of AvgL itself (stronger Hebbian as average activation goes higher) and optionally the average amount of error experienced in the layer (to retain a common proportionality with the level of error-driven learning across layers)
	AvgLLrn float32

	// short time-scale activation average that is actually used for learning -- typically includes a small contribution from AvgM in addition to mostly AvgS, as determined by LrnActAvgParams.LrnM -- important to ensure that when unit turns off in plus phase (short time scale), enough medium-phase trace remains so that learning signal doesn't just go all the way to 0, at which point no learning would take place
	AvgSLrn float32

	// learning activation value, reflecting *dendritic* activity that is not affected by synaptic depression or adapdation channels which are located near the axon hillock.  This is the what drives the Avg* values that drive learning. Computationally, neurons strongly discount the signals sent to other neurons to provide temporal contrast, but need to learn based on a more stable reflection of their overall inputs in the dendrites.
	ActLrn float32

	////////////////////////////
	// Phase

	// the activation state at end of third quarter, which is the traditional posterior-cortical minus phase activation
	ActM float32

	// the activation state at end of fourth quarter, which is the traditional posterior-cortical plus_phase activation
	ActP float32

	// ActP - ActM -- difference between plus and minus phase acts -- reflects the individual error gradient for this neuron in standard error-driven learning terms
	ActDif float32

	// delta activation: change in Act from one cycle to next -- can be useful to track where changes are taking place
	ActDel float32

	// the activation state at start of current alpha cycle (same as the state at end of previous cycle)
	ActQ0 float32

	// the activation state at end of first quarter of current alpha cycle
	ActQ1 float32

	// the activation state at end of second quarter of current alpha cycle
	ActQ2 float32

	// average activation (of final plus phase activation state) over long time intervals (time constant = DtPars.AvgTau -- typically 200) -- useful for finding hog units and seeing overall distribution of activation
	ActAvg float32

	////////////////////////////
	// Gmisc

	// aggregated synaptic inhibition (from Inhib pathways) -- time integral of GiRaw -- this is added with computed FFFB inhibition to get the full inhibition in Gi
	GiSyn float32

	// total amount of self-inhibition -- time-integrated to avoid oscillations
	GiSelf float32

	// last activation value sent (only send when diff is over threshold)
	ActSent float32

	// raw excitatory conductance (net input) received from sending units (send delta's are added to this value)
	GeRaw float32

	// raw inhibitory conductance (net input) received from sending units (send delta's are added to this value)
	GiRaw float32

	// conductance of sodium-gated potassium channel (KNa) fast dynamics (M-type) -- produces accommodation / adaptation of firing
	GknaFast float32

	// conductance of sodium-gated potassium channel (KNa) medium dynamics (Slick) -- produces accommodation / adaptation of firing
	GknaMed float32

	// conductance of sodium-gated potassium channel (KNa) slow dynamics (Slack) -- produces accommodation / adaptation of firing
	GknaSlow float32

	// current inter-spike-interval -- counts up since last spike.  Starts at -1 when initialized.
	ISI float32

	// average inter-spike-interval -- average time interval between spikes.  Starts at -1 when initialized, and goes to -2 after first spike, and is only valid after the second spike post-initialization.
	ISIAvg float32

	// gating activation -- the activity value when gating occurred in this pool.
	ActG float32

	// per-neuron effective learning dopamine value -- gain modulated and sign reversed for D2R
	DALrn float32

	// shunting input received from Patch neurons (in reality flows through SNc DA pathways)
	Shunt float32

	// maintenance value for Deep layers = sending act at time of gating
	Maint float32

	// maintenance excitatory conductance value for Deep layers
	MaintGe float32
}

var NeuronVars = []string{
	"Act", "Ge", "Gi", "Gk", "Inet", "Vm", "Noise", "Spike", "Targ", "Ext",
	"AvgSS", "AvgS", "AvgM", "AvgL", "AvgLLrn", "AvgSLrn", "ActLrn",
	"ActM", "ActP", "ActDif", "ActDel", "ActQ0", "ActQ1", "ActQ2", "ActAvg",
	"GiSyn", "GiSelf", "ActSent", "GeRaw", "GiRaw", "GknaFast", "GknaMed", "GknaSlow", "ISI", "ISIAvg", "ActG", "DALrn", "Shunt", "Maint", "MaintGe", "DA", "ACh", "SE", "GateAct", "GateNow", "GateCnt"}

var NeuronVarsMap map[string]int

var VarCategories = []emer.VarCategory{
	{"Act", "basic activation variables, including conductances, current, Vm, spiking"},
	{"Learn", "calcium-based learning variables"},
	{"Phase", "phase-based activation state"},
	{"Gmisc", "more detailed conductance (G) variables, for specific channels and computational values"},
	{"PBWM", "prefrontal cortex basal ganglia working memory model variables, including neuromodulation"},
	{"Wts", "weights and other synaptic-level variables"},
}

var NeuronVarProps = map[string]string{
	// Act vars
	"Act":   `cat:"Act"`,
	"Ge":    `cat:"Act"`,
	"Gi":    `cat:"Act"`,
	"Gk":    `cat:"Act"`,
	"Inet":  `cat:"Act"`,
	"Vm":    `cat:"Act" min:"0" max:"1"`,
	"Noise": `cat:"Act"`,
	"Spike": `cat:"Act"`,
	"Targ":  `cat:"Act"`,
	"Ext":   `cat:"Act"`,

	// Learn vars
	"AvgSS":   `cat:"Learn"`,
	"AvgS":    `cat:"Learn"`,
	"AvgM":    `cat:"Learn"`,
	"AvgL":    `cat:"Learn"`,
	"AvgLLrn": `cat:"Learn"`,
	"AvgSLrn": `cat:"Learn"`,
	"ActLrn":  `cat:"Learn"`,

	// Phase vars
	"ActM":   `cat:"Phase"`,
	"ActP":   `cat:"Phase"`,
	"ActDif": `cat:"Phase" auto-scale:"+"`,
	"ActDel": `cat:"Phase" auto-scale:"+"`,
	"ActQ0":  `cat:"Phase"`,
	"ActQ1":  `cat:"Phase"`,
	"ActQ2":  `cat:"Phase"`,
	"ActAvg": `cat:"Phase"`,

	// Gmisc vars
	"GiSyn":    `cat:"Gmisc"`,
	"GiSelf":   `cat:"Gmisc"`,
	"ActSent":  `cat:"Gmisc"`,
	"GeRaw":    `cat:"Gmisc"`,
	"GiRaw":    `cat:"Gmisc"`,
	"GknaFast": `cat:"Gmisc"`,
	"GknaMed":  `cat:"Gmisc"`,
	"GknaSlow": `cat:"Gmisc"`,
	"ISI":      `cat:"Gmisc"`,
	"ISIAvg":   `cat:"Gmisc"`,

	"ActG":    `cat:"PBWM"`,
	"DALrn":   `cat:"PBWM"`,
	"Shunt":   `cat:"PBWM"`,
	"Maint":   `cat:"PBWM"`,
	"MaintGe": `cat:"PBWM"`,
	"DA":      `cat:"PBWM"`,
	"ACh":     `cat:"PBWM"`,
	"SE":      `cat:"PBWM"`,
	"GateAct": `cat:"PBWM"`,
	"GateNow": `cat:"PBWM"`,
	"GateCnt": `cat:"PBWM"`,
}

func init() {
	NeuronVarsMap = make(map[string]int, len(NeuronVars))
	for i, v := range NeuronVars {
		NeuronVarsMap[v] = i
	}
	ntyp := types.For[Neuron]()
	for _, fld := range ntyp.Fields {
		tag := NeuronVarProps[fld.Name]
		NeuronVarProps[fld.Name] = tag + ` doc:"` + strings.ReplaceAll(fld.Doc, "\n", " ") + `"`
	}
}

func (nrn *Neuron) VarNames() []string {
	return NeuronVars
}

// NeuronVarIndexByName returns the index of the variable in the Neuron, or error
func NeuronVarIndexByName(varNm string) (int, error) {
	i, ok := NeuronVarsMap[varNm]
	if !ok {
		return -1, fmt.Errorf("Neuron VarByName: variable name: %v not valid", varNm)
	}
	return i, nil
}

// VarByIndex returns variable using index (0 = first variable in NeuronVars list)
func (nrn *Neuron) VarByIndex(idx int) float32 {
	fv := (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(nrn)) + uintptr(NeuronVarStart+4*idx)))
	return *fv
}

// VarByName returns variable by name, or error
func (nrn *Neuron) VarByName(varNm string) (float32, error) {
	i, err := NeuronVarIndexByName(varNm)
	if err != nil {
		return math32.NaN(), err
	}
	return nrn.VarByIndex(i), nil
}

func (nrn *Neuron) HasFlag(f NeurFlags) bool {
	return nrn.Flags.HasFlag(f)
}

func (nrn *Neuron) SetFlag(on bool, f ...enums.BitFlag) {
	nrn.Flags.SetFlag(on, f...)
}

// IsOff returns true if the neuron has been turned off (lesioned)
func (nrn *Neuron) IsOff() bool {
	return nrn.HasFlag(NeurOff)
}

// NeurFlags are bit-flags encoding relevant binary state for neurons
type NeurFlags int64 //enums:bitflag

// The neuron flags
const (
	// NeurOff flag indicates that this neuron has been turned off (i.e., lesioned)
	NeurOff NeurFlags = iota

	// NeurHasExt means the neuron has external input in its Ext field
	NeurHasExt

	// NeurHasTarg means the neuron has external target input in its Targ field
	NeurHasTarg

	// NeurHasCmpr means the neuron has external comparison input in its Targ field -- used for computing
	// comparison statistics but does not drive neural activity ever
	NeurHasCmpr
)
