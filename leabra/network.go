// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"strings"
	"unsafe"

	"cogentcore.org/core/base/datasize"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/paths"
)

///////////////////////////////////////////////////////////////////////////
//  Primary Algorithmic interface.
//
//  The following methods constitute the primary user-called API during
//  AlphaCyc method to compute one complete algorithmic alpha cycle update.

// AlphaCycInit handles all initialization at start of new input pattern.
// Should already have presented the external input to the network at this point.
// If updtActAvg is true, this includes updating the running-average
// activations for each layer / pool, and the AvgL running average used
// in BCM Hebbian learning.
// The input scaling is updated  based on the layer-level running average acts,
// and this can then change the behavior of the network,
// so if you want 100% repeatable testing results, set this to false to
// keep the existing scaling factors (e.g., can pass a train bool to
// only update during training).
// This flag also affects the AvgL learning threshold.
func (nt *Network) AlphaCycInit(updtActAvg bool) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.AlphaCycInit(updtActAvg)
	}
}

// Cycle runs one cycle of activation updating:
// * Sends Ge increments from sending to receiving layers
// * Average and Max Ge stats
// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
// * Activation from Ge, Gi, and Gl
// * Average and Max Act stats
// This basic version doesn't use the time info, but more specialized types do, and we
// want to keep a consistent API for end-user code.
func (nt *Network) Cycle(ctx *Context) {
	nt.SendGDelta(ctx) // also does integ
	nt.AvgMaxGe(ctx)
	nt.InhibFromGeAct(ctx)
	nt.ActFromG(ctx)
	nt.AvgMaxAct(ctx)
	nt.GateSend(ctx)   // GateLayer (GPiThal) computes gating, sends to other layers
	nt.RecGateAct(ctx) // Record activation state at time of gating (in ActG neuron var)
	nt.SendMods(ctx)   // send neuromod
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendGeDelta sends change in activation since last sent, if above thresholds
// and integrates sent deltas into GeRaw and time-integrated Ge values
func (nt *Network) SendGDelta(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.SendGDelta(ctx)
	}
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.GFromInc(ctx)
	}
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (nt *Network) AvgMaxGe(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.AvgMaxGe(ctx)
	}
}

// InhibiFromGeAct computes inhibition Gi from Ge and Act stats within relevant Pools
func (nt *Network) InhibFromGeAct(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InhibFromGeAct(ctx)
	}
}

// ActFromG computes rate-code activation from Ge, Gi, Gl conductances
func (nt *Network) ActFromG(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.ActFromG(ctx)
	}
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (nt *Network) AvgMaxAct(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.AvgMaxAct(ctx)
	}
}

// QuarterFinal does updating after end of a quarter, for first 2
func (nt *Network) QuarterFinal(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.QuarterFinal(ctx)
	}
}

// MinusPhase is called at the end of the minus phase (quarter 3), to record state.
func (nt *Network) MinusPhase(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.MinusPhase(ctx)
	}
}

// PlusPhase is called at the end of the plus phase (quarter 4), to record state.
func (nt *Network) PlusPhase(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.PlusPhase(ctx)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) based on current
// running-average activation values
func (nt *Network) DWt() {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.DWt()
	}
}

// WtFromDWt updates the weights from delta-weight changes.
// Also calls WtBalFromWt every WtBalInterval times
func (nt *Network) WtFromDWt() {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.WtFromDWt()
	}

	nt.WtBalCtr++
	if nt.WtBalCtr >= nt.WtBalInterval {
		nt.WtBalCtr = 0
		for _, ly := range nt.Layers {
			if ly.Off {
				continue
			}

			ly.WtBalFromWt()
		}
	}
}

// LrateMult sets the new Lrate parameter for Paths to LrateInit * mult.
// Useful for implementing learning rate schedules.
func (nt *Network) LrateMult(mult float32) {
	for _, ly := range nt.Layers {
		// if ly.Off { // keep all sync'd
		// 	continue
		// }
		ly.LrateMult(mult)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWeights initializes synaptic weights and all other
// associated long-term state variables including running-average
// state values (e.g., layer running average activations etc).
func (nt *Network) InitWeights() {
	nt.WtBalCtr = 0
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitWeights()
	}
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitWtSym()
	}
}

// InitTopoScales initializes synapse-specific scale parameters from
// path types that support them, with flags set to support it,
// includes: paths.PoolTile paths.Circle.
// call before InitWeights if using Topo wts.
func (nt *Network) InitTopoScales() {
	scales := &tensor.Float32{}
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		rpjn := ly.RecvPaths
		for _, pt := range rpjn {
			if pt.Off {
				continue
			}
			pat := pt.Pattern
			switch ptn := pat.(type) {
			case *paths.PoolTile:
				if !ptn.HasTopoWeights() {
					continue
				}
				slay := pt.Send
				ptn.TopoWeights(&slay.Shape, &ly.Shape, scales)
				pt.SetScalesRPool(scales)
			case *paths.Circle:
				if !ptn.TopoWeights {
					continue
				}
				pt.SetScalesFunc(ptn.GaussWts)
			}
		}
	}
}

// DecayState decays activation state by given proportion
// e.g., 1 = decay completely, and 0 = decay not at all
// This is called automatically in AlphaCycInit, but is avail
// here for ad-hoc decay cases.
func (nt *Network) DecayState(decay float32) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.DecayState(decay)
	}
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitActs()
	}
}

// InitExt initializes external input state.
// call prior to applying external inputs to layers.
func (nt *Network) InitExt() {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitExt()
	}
}

// UpdateExtFlags updates the neuron flags for external input
// based on current layer Type field.
// call this if the Type has changed since the last
// ApplyExt* method call.
func (nt *Network) UpdateExtFlags() {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.UpdateExtFlags()
	}
}

// InitGinc initializes the Ge excitatory and Gi inhibitory
// conductance accumulation states including ActSent and G*Raw values.
// called at start of trial always (at layer level), and can be
// called optionally when delta-based Ge computation needs
// to be updated (e.g., weights might have changed strength).
func (nt *Network) InitGInc() {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitGInc()
	}
}

// GScaleFromAvgAct computes the scaling factor for synaptic input conductances G,
// based on sending layer average activation.
// This attempts to automatically adjust for overall differences in raw activity
// coming into the units to achieve a general target of around .5 to 1
// for the integrated Ge value.
// This is automatically done during AlphaCycInit, but if scaling parameters are
// changed at any point thereafter during AlphaCyc, this must be called.
func (nt *Network) GScaleFromAvgAct() {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.GScaleFromAvgAct()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion methods

// LayersSetOff sets the Off flag for all layers to given setting
func (nt *Network) LayersSetOff(off bool) {
	for _, ly := range nt.Layers {
		ly.Off = off
	}
}

// UnLesionNeurons unlesions neurons in all layers in the network.
// Provides a clean starting point for subsequent lesion experiments.
func (nt *Network) UnLesionNeurons() {
	for _, ly := range nt.Layers {
		// if ly.Off { // keep all sync'd
		// 	continue
		// }
		ly.UnLesionNeurons()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Methods used in MPI computation, which don't depend on MPI specifically

// CollectDWts writes all of the synaptic DWt values to given dwts slice
// which is pre-allocated to given nwts size if dwts is nil,
// in which case the method returns true so that the actual length of
// dwts can be passed next time around.
// Used for MPI sharing of weight changes across processors.
func (nt *Network) CollectDWts(dwts *[]float32, nwts int) bool {
	idx := 0
	made := false
	if *dwts == nil {
		// todo: if nil, compute right size right away
		*dwts = make([]float32, 0, nwts)
		made = true
	}
	for _, ly := range nt.Layers {
		for _, pt := range ly.SendPaths {
			ns := len(pt.Syns)
			nsz := idx + ns
			if len(*dwts) < nsz {
				*dwts = append(*dwts, make([]float32, nsz-len(*dwts))...)
			}
			for j := range pt.Syns {
				sy := &(pt.Syns[j])
				(*dwts)[idx+j] = sy.DWt
			}
			idx += ns
		}
	}
	return made
}

// SetDWts sets the DWt weight changes from given array of floats,
// which must be correct size.
func (nt *Network) SetDWts(dwts []float32) {
	idx := 0
	for _, ly := range nt.Layers {
		for _, pt := range ly.SendPaths {
			ns := len(pt.Syns)
			for j := range pt.Syns {
				sy := &(pt.Syns[j])
				sy.DWt = dwts[idx+j]
			}
			idx += ns
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Misc Reports

// SizeReport returns a string reporting the size of
// each layer and pathway in the network, and total memory footprint.
func (nt *Network) SizeReport() string {
	var b strings.Builder
	neur := 0
	neurMem := 0
	syn := 0
	synMem := 0
	for _, ly := range nt.Layers {
		nn := len(ly.Neurons)
		nmem := nn * int(unsafe.Sizeof(Neuron{}))
		neur += nn
		neurMem += nmem
		fmt.Fprintf(&b, "%14s:\t Neurons: %d\t NeurMem: %v \t Sends To:\n", ly.Name, nn, (datasize.Size)(nmem).String())
		for _, pt := range ly.SendPaths {
			ns := len(pt.Syns)
			syn += ns
			pmem := ns*int(unsafe.Sizeof(Synapse{})) + len(pt.GInc)*4 + len(pt.WbRecv)*int(unsafe.Sizeof(WtBalRecvPath{}))
			synMem += pmem
			fmt.Fprintf(&b, "\t%14s:\t Syns: %d\t SynnMem: %v\n", pt.Recv.Name, ns, (datasize.Size)(pmem).String())
		}
	}
	fmt.Fprintf(&b, "\n\n%14s:\t Neurons: %d\t NeurMem: %v \t Syns: %d \t SynMem: %v\n", nt.Name, neur, (datasize.Size)(neurMem).String(), syn, (datasize.Size)(synMem).String())
	return b.String()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network props for gui

// TODO(v2): props

// var NetworkProps = tree.Props{
// 	"ToolBar": tree.PropSlice{
// 		{"SaveWeightsJSON", tree.Props{
// 			"label": "Save Wts...",
// 			"icon":  "file-save",
// 			"desc":  "Save json-formatted weights",
// 			"Args": tree.PropSlice{
// 				{"Weights File Name", tree.Props{
// 					"default-field": "WtsFile",
// 					"ext":           ".wts,.wts.gz",
// 				}},
// 			},
// 		}},
// 		{"OpenWeightsJSON", tree.Props{
// 			"label": "Open Wts...",
// 			"icon":  "file-open",
// 			"desc":  "Open json-formatted weights",
// 			"Args": tree.PropSlice{
// 				{"Weights File Name", tree.Props{
// 					"default-field": "WtsFile",
// 					"ext":           ".wts,.wts.gz",
// 				}},
// 			},
// 		}},
// 		{"sep-file", tree.BlankProp{}},
// 		{"Build", tree.Props{
// 			"icon": "update",
// 			"desc": "build the network's neurons and synapses according to current params",
// 		}},
// 		{"InitWeights", tree.Props{
// 			"icon": "update",
// 			"desc": "initialize the network weight values according to path parameters",
// 		}},
// 		{"InitActs", tree.Props{
// 			"icon": "update",
// 			"desc": "initialize the network activation values",
// 		}},
// 		{"sep-act", tree.BlankProp{}},
// 		{"AddLayer", tree.Props{
// 			"label": "Add Layer...",
// 			"icon":  "new",
// 			"desc":  "add a new layer to network",
// 			"Args": tree.PropSlice{
// 				{"Layer Name", tree.Props{}},
// 				{"Layer Shape", tree.Props{
// 					"desc": "shape of layer, typically 2D (Y, X) or 4D (Pools Y, Pools X, Units Y, Units X)",
// 				}},
// 				{"Layer Type", tree.Props{
// 					"desc": "type of layer -- used for determining how inputs are applied",
// 				}},
// 			},
// 		}},
// 		{"ConnectLayerNames", tree.Props{
// 			"label": "Connect Layers...",
// 			"icon":  "new",
// 			"desc":  "add a new connection between layers in the network",
// 			"Args": tree.PropSlice{
// 				{"Send Layer Name", tree.Props{}},
// 				{"Recv Layer Name", tree.Props{}},
// 				{"Pattern", tree.Props{
// 					"desc": "pattern to connect with",
// 				}},
// 				{"Path Type", tree.Props{
// 					"desc": "type of pathway -- direction, or other more specialized factors",
// 				}},
// 			},
// 		}},
// 		{"AllWtScales", tree.Props{
// 			"icon":        "file-sheet",
// 			"desc":        "AllWtScales returns a listing of all WtScale parameters in the Network in all Layers, Recv pathways.  These are among the most important and numerous of parameters (in larger networks) -- this helps keep track of what they all are set to.",
// 			"show-return": true,
// 		}},
// 	},
// }
