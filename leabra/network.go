// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"unsafe"

	"github.com/c2h5oh/datasize"
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// leabra.Network has parameters for running a basic rate-coded Leabra network
type Network struct {
	NetworkStru
	WtBalInterval int `def:"10" desc:"how frequently to update the weight balance average weight factor -- relatively expensive"`
	WtBalCtr      int `inactive:"+" desc:"counter for how long it has been since last WtBal"`
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

func (nt *Network) AsLeabra() *Network {
	return nt
}

// NewLayer returns new layer of proper type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPrjn returns new prjn of proper type
func (nt *Network) NewPrjn() emer.Prjn {
	return &Prjn{}
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.WtBalInterval = 10
	nt.WtBalCtr = 0
	for li, ly := range nt.Layers {
		ly.Defaults()
		ly.SetIndex(li)
	}
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	for _, ly := range nt.Layers {
		ly.UpdateParams()
	}
}

// UnitVarNames returns a list of variable names available on the units in this network.
// Not all layers need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) UnitVarNames() []string {
	return NeuronVars
}

// UnitVarProps returns properties for variables
func (nt *Network) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// SynVarNames returns the names of all the variables on the synapses in this network.
// Not all projections need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) SynVarNames() []string {
	return SynapseVars
}

// SynVarProps returns properties for variables
func (nt *Network) SynVarProps() map[string]string {
	return SynapseVarProps
}

//////////////////////////////////////////////////////////////////////////////////////
//  Primary Algorithmic interface.
//
//  The following methods constitute the primary user-called API during AlphaCyc method
//  to compute one complete algorithmic alpha cycle update.
//
//  They just call the corresponding Impl method using the LeabraNetwork interface
//  so that other network types can specialize any of these entry points.

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
func (nt *Network) AlphaCycInit() {
	nt.EmerNet.(LeabraNetwork).AlphaCycInitImpl()
}

// Cycle runs one cycle of activation updating:
// * Sends Ge increments from sending to receiving layers
// * Average and Max Ge stats
// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
// * Activation from Ge, Gi, and Gl
// * Average and Max Act stats
// This basic version doesn't use the time info, but more specialized types do, and we
// want to keep a consistent API for end-user code.
func (nt *Network) Cycle(ltime *Time) {
	nt.EmerNet.(LeabraNetwork).CycleImpl(ltime)
	nt.EmerNet.(LeabraNetwork).CyclePostImpl(ltime) // always call this after std cycle..
}

// CyclePost is called after the standard Cycle update, and calls CyclePost
// on Layers -- this is reserved for any kind of special ad-hoc types that
// need to do something special after Act is finally computed.
// For example, sending a neuromodulatory signal such as dopamine.
func (nt *Network) CyclePost(ltime *Time) {
	nt.EmerNet.(LeabraNetwork).CyclePostImpl(ltime)
}

// QuarterFinal does updating after end of a quarter
func (nt *Network) QuarterFinal(ltime *Time) {
	nt.EmerNet.(LeabraNetwork).QuarterFinalImpl(ltime)
}

// DWt computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWt() {
	nt.EmerNet.(LeabraNetwork).DWtImpl()
}

// WtFmDWt updates the weights from delta-weight changes.
// Also calls WtBalFmWt every WtBalInterval times
func (nt *Network) WtFmDWt() {
	nt.EmerNet.(LeabraNetwork).WtFmDWtImpl()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes synaptic weights and all other associated long-term state variables
// including running-average state values (e.g., layer running average activations etc)
func (nt *Network) InitWts() {
	nt.WtBalCtr = 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitWts()
	}
	// separate pass to enforce symmetry
	// st := time.Now()
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitWtSym()
	}
	// dur := time.Now().Sub(st)
	// fmt.Printf("sym: %v\n", dur)
}

// InitTopoScales initializes synapse-specific scale parameters from
// prjn types that support them, with flags set to support it,
// includes: prjn.PoolTile prjn.Circle.
// call before InitWts if using Topo wts
func (nt *Network) InitTopoScales() {
	scales := &etensor.Float32{}
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		rpjn := ly.RecvPrjns()
		for _, p := range *rpjn {
			if p.IsOff() {
				continue
			}
			pat := p.Pattern()
			switch pt := pat.(type) {
			case *prjn.PoolTile:
				if !pt.HasTopoWts() {
					continue
				}
				pj := p.(LeabraPrjn).AsLeabra()
				slay := p.SendLay()
				pt.TopoWts(slay.Shape(), ly.Shape(), scales)
				pj.SetScalesRPool(scales)
			case *prjn.Circle:
				if !pt.TopoWts {
					continue
				}
				pj := p.(LeabraPrjn).AsLeabra()
				pj.SetScalesFunc(pt.GaussWts)
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
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).DecayState(decay)
	}
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitActs()
	}
}

// InitExt initializes external input state -- call prior to applying external inputs to layers
func (nt *Network) InitExt() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitExt()
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (nt *Network) UpdateExtFlags() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).UpdateExtFlags()
	}
}

// InitGinc initializes the Ge excitatory and Gi inhibitory conductance accumulation states
// including ActSent and G*Raw values.
// called at start of trial always (at layer level), and can be called optionally
// when delta-based Ge computation needs to be updated (e.g., weights
// might have changed strength)
func (nt *Network) InitGInc() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).InitGInc()
	}
}

// AlphaCycInitImpl handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
func (nt *Network) AlphaCycInitImpl() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).AlphaCycInit()
	}
}

// GScaleFmAvgAct computes the scaling factor for synaptic input conductances G,
// based on sending layer average activation.
// This attempts to automatically adjust for overall differences in raw activity
// coming into the units to achieve a general target of around .5 to 1
// for the integrated Ge value.
// This is automatically done during AlphaCycInit, but if scaling parameters are
// changed at any point thereafter during AlphaCyc, this must be called.
func (nt *Network) GScaleFmAvgAct() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(LeabraLayer).GScaleFmAvgAct()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// CycleImpl runs one cycle of activation updating:
// * Sends Ge increments from sending to receiving layers
// * Average and Max Ge stats
// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
// * Activation from Ge, Gi, and Gl
// * Average and Max Act stats
// This basic version doesn't use the time info, but more specialized types do, and we
// want to keep a consistent API for end-user code.
func (nt *Network) CycleImpl(ltime *Time) {
	nt.SendGDelta(ltime) // also does integ
	nt.AvgMaxGe(ltime)
	nt.InhibFmGeAct(ltime)
	nt.ActFmG(ltime)
	nt.AvgMaxAct(ltime)
}

// SendGeDelta sends change in activation since last sent, if above thresholds
// and integrates sent deltas into GeRaw and time-integrated Ge values
func (nt *Network) SendGDelta(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.SendGDelta(ltime) }, "SendGDelta")
	nt.ThrLayFun(func(ly LeabraLayer) { ly.GFmInc(ltime) }, "GFmInc   ")
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (nt *Network) AvgMaxGe(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.AvgMaxGe(ltime) }, "AvgMaxGe")
}

// InhibiFmGeAct computes inhibition Gi from Ge and Act stats within relevant Pools
func (nt *Network) InhibFmGeAct(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.InhibFmGeAct(ltime) }, "InhibFmGeAct")
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
func (nt *Network) ActFmG(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.ActFmG(ltime) }, "ActFmG   ")
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (nt *Network) AvgMaxAct(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.AvgMaxAct(ltime) }, "AvgMaxAct")
}

// CyclePostImpl is called after the standard Cycle update, and calls CyclePost
// on Layers -- this is reserved for any kind of special ad-hoc types that
// need to do something special after Act is finally computed.
// For example, sending a neuromodulatory signal such as dopamine.
func (nt *Network) CyclePostImpl(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.CyclePost(ltime) }, "CyclePost")
}

// QuarterFinalImpl does updating after end of a quarter
func (nt *Network) QuarterFinalImpl(ltime *Time) {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.QuarterFinal(ltime) }, "QuarterFinal")
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWtImpl computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWtImpl() {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.DWt() }, "DWt     ")
}

// WtFmDWtImpl updates the weights from delta-weight changes.
// Also calls WtBalFmWt every WtBalInterval times
func (nt *Network) WtFmDWtImpl() {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.WtFmDWt() }, "WtFmDWt")
	nt.WtBalCtr++
	if nt.WtBalCtr >= nt.WtBalInterval {
		nt.WtBalCtr = 0
		nt.WtBalFmWt()
	}
}

// WtBalFmWt updates the weight balance factors based on average recv weights
func (nt *Network) WtBalFmWt() {
	nt.ThrLayFun(func(ly LeabraLayer) { ly.WtBalFmWt() }, "WtBalFmWt")
}

// LrateMult sets the new Lrate parameter for Prjns to LrateInit * mult.
// Useful for implementing learning rate schedules.
func (nt *Network) LrateMult(mult float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.(LeabraLayer).LrateMult(mult)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion methods

// LayersSetOff sets the Off flag for all layers to given setting
func (nt *Network) LayersSetOff(off bool) {
	for _, ly := range nt.Layers {
		ly.SetOff(off)
	}
}

// UnLesionNeurons unlesions neurons in all layers in the network.
// Provides a clean starting point for subsequent lesion experiments.
func (nt *Network) UnLesionNeurons() {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.(LeabraLayer).AsLeabra().UnLesionNeurons()
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
	for _, lyi := range nt.Layers {
		ly := lyi.(LeabraLayer).AsLeabra()
		for _, pji := range ly.SndPrjns {
			pj := pji.(LeabraPrjn).AsLeabra()
			ns := len(pj.Syns)
			nsz := idx + ns
			if len(*dwts) < nsz {
				*dwts = append(*dwts, make([]float32, nsz-len(*dwts))...)
			}
			for j := range pj.Syns {
				sy := &(pj.Syns[j])
				(*dwts)[idx+j] = sy.DWt
			}
			idx += ns
		}
	}
	return made
}

// SetDWts sets the DWt weight changes from given array of floats, which must be correct size
func (nt *Network) SetDWts(dwts []float32) {
	idx := 0
	for _, lyi := range nt.Layers {
		ly := lyi.(LeabraLayer).AsLeabra()
		for _, pji := range ly.SndPrjns {
			pj := pji.(LeabraPrjn).AsLeabra()
			ns := len(pj.Syns)
			for j := range pj.Syns {
				sy := &(pj.Syns[j])
				sy.DWt = dwts[idx+j]
			}
			idx += ns
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Misc Reports / Threading Allocation

// SizeReport returns a string reporting the size of each layer and projection
// in the network, and total memory footprint.
func (nt *Network) SizeReport() string {
	var b strings.Builder
	neur := 0
	neurMem := 0
	syn := 0
	synMem := 0
	for _, lyi := range nt.Layers {
		ly := lyi.(LeabraLayer).AsLeabra()
		nn := len(ly.Neurons)
		nmem := nn * int(unsafe.Sizeof(Neuron{}))
		neur += nn
		neurMem += nmem
		fmt.Fprintf(&b, "%14s:\t Neurons: %d\t NeurMem: %v \t Sends To:\n", ly.Nm, nn, (datasize.ByteSize)(nmem).HumanReadable())
		for _, pji := range ly.SndPrjns {
			pj := pji.(LeabraPrjn).AsLeabra()
			ns := len(pj.Syns)
			syn += ns
			pmem := ns*int(unsafe.Sizeof(Synapse{})) + len(pj.GInc)*4 + len(pj.WbRecv)*int(unsafe.Sizeof(WtBalRecvPrjn{}))
			synMem += pmem
			fmt.Fprintf(&b, "\t%14s:\t Syns: %d\t SynnMem: %v\n", pj.Recv.Name(), ns, (datasize.ByteSize)(pmem).HumanReadable())
		}
	}
	fmt.Fprintf(&b, "\n\n%14s:\t Neurons: %d\t NeurMem: %v \t Syns: %d \t SynMem: %v\n", nt.Nm, neur, (datasize.ByteSize)(neurMem).HumanReadable(), syn, (datasize.ByteSize)(synMem).HumanReadable())
	return b.String()
}

// ThreadAlloc allocates layers to given number of threads,
// attempting to evenly divide computation.  Returns report
// of thread allocations and estimated computational cost per thread.
func (nt *Network) ThreadAlloc(nThread int) string {
	nl := len(nt.Layers)
	if nl < nThread {
		return fmt.Sprintf("Number of threads: %d > number of layers: %d -- must be less\n", nThread, nl)
	}
	if nl == nThread {
		for li, lyi := range nt.Layers {
			ly := lyi.(LeabraLayer).AsLeabra()
			ly.SetThread(li)
		}
		return fmt.Sprintf("Number of threads: %d == number of layers: %d\n", nThread, nl)
	}

	type td struct {
		Lays []int
		Neur int // neur cost
		Syn  int // send syn cost
		Tot  int // total cost
	}

	avgFunc := func(thds []td) float32 {
		avg := 0
		for i := range thds {
			avg += thds[i].Tot
		}
		return float32(avg) / float32(len(thds))
	}

	devFunc := func(thds []td) float32 {
		avg := avgFunc(thds)
		dev := float32(0)
		for i := range thds {
			dev += math32.Abs(float32(thds[i].Tot) - avg)
		}
		return float32(dev) / float32(len(thds))
	}

	// cache per-layer data first
	ld := make([]td, nl)
	for li, lyi := range nt.Layers {
		ly := lyi.(LeabraLayer).AsLeabra()
		ld[li].Neur, ld[li].Syn, ld[li].Tot = ly.CostEst()
	}

	// number of initial random permutations to create
	initN := 100
	pth := float64(nl) / float64(nThread)
	if pth < 2 {
		initN = 10
	} else if pth > 3 {
		initN = 500
	}
	thrs := make([][]td, initN)
	devs := make([]float32, initN)
	ord := rand.Perm(nl)
	minDev := float32(1.0e20)
	minDevIdx := -1
	for ti := 0; ti < initN; ti++ {
		thds := &thrs[ti]
		*thds = make([]td, nThread)
		for t := 0; t < nThread; t++ {
			thd := &(*thds)[t]
			ist := int(math.Round(float64(t) * pth))
			ied := int(math.Round(float64(t+1) * pth))
			thd.Neur = 0
			thd.Syn = 0
			for i := ist; i < ied; i++ {
				li := ord[i]
				thd.Neur += ld[li].Neur
				thd.Syn += ld[li].Syn
				thd.Tot += ld[li].Tot
				thd.Lays = append(thd.Lays, ord[i])
			}
		}
		dev := devFunc(*thds)
		if dev < minDev {
			minDev = dev
			minDevIdx = ti
		}
		devs[ti] = dev
		erand.PermuteInts(ord)
	}

	// todo: could optimize best case further by trying to switch one layer at random with each other
	// thread, and seeing if that is faster..  but probably not worth it given inaccuracy of estimate.

	var b strings.Builder
	b.WriteString(nt.ThreadReport())

	fmt.Fprintf(&b, "Deviation: %s \t Idx: %d\n", (datasize.ByteSize)(minDev).HumanReadable(), minDevIdx)

	nt.StopThreads()
	nt.BuildThreads()
	nt.StartThreads()

	return b.String()
}

// ThreadReport returns report of thread allocations and
// estimated computational cost per thread.
func (nt *Network) ThreadReport() string {
	var b strings.Builder
	// p := message.NewPrinter(language.English)
	fmt.Fprintf(&b, "Network: %s Auto Thread Allocation for %d threads:\n", nt.Nm, nt.NThreads)
	for th := 0; th < nt.NThreads; th++ {
		tneur := 0
		tsyn := 0
		ttot := 0
		for _, lyi := range nt.ThrLay[th] {
			ly := lyi.(LeabraLayer).AsLeabra()
			neur, syn, tot := ly.CostEst()
			tneur += neur
			tsyn += syn
			ttot += tot
			fmt.Fprintf(&b, "\t%14s: cost: %d K \t neur: %d K \t syn: %d K\n", ly.Nm, tot/1000, neur/1000, syn/1000)
		}
		fmt.Fprintf(&b, "Thread: %d \t cost: %d K \t neur: %d K \t syn: %d K\n", th, ttot/1000, tneur/1000, tsyn/1000)
	}
	return b.String()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network props for gui

var NetworkProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"SaveWtsJSON", ki.Props{
			"label": "Save Wts...",
			"icon":  "file-save",
			"desc":  "Save json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts,.wts.gz",
				}},
			},
		}},
		{"OpenWtsJSON", ki.Props{
			"label": "Open Wts...",
			"icon":  "file-open",
			"desc":  "Open json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts,.wts.gz",
				}},
			},
		}},
		{"sep-file", ki.BlankProp{}},
		{"Build", ki.Props{
			"icon": "update",
			"desc": "build the network's neurons and synapses according to current params",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the network weight values according to prjn parameters",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the network activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"AddLayer", ki.Props{
			"label": "Add Layer...",
			"icon":  "new",
			"desc":  "add a new layer to network",
			"Args": ki.PropSlice{
				{"Layer Name", ki.Props{}},
				{"Layer Shape", ki.Props{
					"desc": "shape of layer, typically 2D (Y, X) or 4D (Pools Y, Pools X, Units Y, Units X)",
				}},
				{"Layer Type", ki.Props{
					"desc": "type of layer -- used for determining how inputs are applied",
				}},
			},
		}},
		{"ConnectLayerNames", ki.Props{
			"label": "Connect Layers...",
			"icon":  "new",
			"desc":  "add a new connection between layers in the network",
			"Args": ki.PropSlice{
				{"Send Layer Name", ki.Props{}},
				{"Recv Layer Name", ki.Props{}},
				{"Pattern", ki.Props{
					"desc": "pattern to connect with",
				}},
				{"Prjn Type", ki.Props{
					"desc": "type of projection -- direction, or other more specialized factors",
				}},
			},
		}},
		{"AllWtScales", ki.Props{
			"icon":        "file-sheet",
			"desc":        "AllWtScales returns a listing of all WtScale parameters in the Network in all Layers, Recv projections.  These are among the most important and numerous of parameters (in larger networks) -- this helps keep track of what they all are set to.",
			"show-return": true,
		}},
	},
}
