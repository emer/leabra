// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"

	"cogentcore.org/core/base/indent"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/weights"
)

// note: paths.go contains algorithm methods; pathbase.go has infrastructure.

// Path implements the Leabra algorithm at the synaptic level,
// in terms of a pathway connecting two layers.
type Path struct {
	emer.PathBase

	// sending layer for this pathway.
	Send *Layer

	// receiving layer for this pathway.
	Recv *Layer

	// type of pathway.
	Type PathTypes

	// initial random weight distribution
	WtInit WtInitParams `display:"inline"`

	// weight scaling parameters: modulates overall strength of pathway,
	// using both absolute and relative factors.
	WtScale WtScaleParams `display:"inline"`

	// synaptic-level learning parameters
	Learn LearnSynParams `display:"add-fields"`

	// CHL are the parameters for CHL learning. if CHL is On then
	// WtSig.SoftBound is automatically turned off, as it is incompatible.
	CHL CHLParams `display:"inline"`

	// special parameters for matrix trace learning
	Trace TraceParams `display:"inline"`

	// synaptic state values, ordered by the sending layer
	// units which owns them -- one-to-one with SConIndex array.
	Syns []Synapse

	// scaling factor for integrating synaptic input conductances (G's).
	// computed in AlphaCycInit, incorporates running-average activity levels.
	GScale float32

	// local per-recv unit increment accumulator for synaptic
	// conductance from sending units. goes to either GeRaw or GiRaw
	// on neuron depending on pathway type.
	GInc []float32

	// per-recv, per-path raw excitatory input, for GPiThalPath
	GeRaw []float32

	// weight balance state variables for this pathway, one per recv neuron.
	WbRecv []WtBalRecvPath

	// number of recv connections for each neuron in the receiving layer,
	// as a flat list.
	RConN []int32 `display:"-"`

	// average and maximum number of recv connections in the receiving layer.
	RConNAvgMax minmax.AvgMax32 `edit:"-" display:"inline"`

	// starting index into ConIndex list for each neuron in
	// receiving layer; list incremented by ConN.
	RConIndexSt []int32 `display:"-"`

	// index of other neuron on sending side of pathway,
	// ordered by the receiving layer's order of units as the
	// outer loop (each start is in ConIndexSt),
	// and then by the sending layer's units within that.
	RConIndex []int32 `display:"-"`

	// index of synaptic state values for each recv unit x connection,
	// for the receiver pathway which does not own the synapses,
	// and instead indexes into sender-ordered list.
	RSynIndex []int32 `display:"-"`

	// number of sending connections for each neuron in the
	// sending layer, as a flat list.
	SConN []int32 `display:"-"`

	// average and maximum number of sending connections
	// in the sending layer.
	SConNAvgMax minmax.AvgMax32 `edit:"-" display:"inline"`

	// starting index into ConIndex list for each neuron in
	// sending layer; list incremented by ConN.
	SConIndexSt []int32 `display:"-"`

	// index of other neuron on receiving side of pathway,
	// ordered by the sending layer's order of units as the
	// outer loop (each start is in ConIndexSt), and then
	// by the sending layer's units within that.
	SConIndex []int32 `display:"-"`
}

// emer.Path interface

func (pt *Path) StyleObject() any      { return pt }
func (pt *Path) RecvLayer() emer.Layer { return pt.Recv }
func (pt *Path) SendLayer() emer.Layer { return pt.Send }
func (pt *Path) TypeName() string      { return pt.Type.String() }
func (pt *Path) TypeNumber() int       { return int(pt.Type) }

func (pt *Path) Defaults() {
	pt.WtInit.Defaults()
	pt.WtScale.Defaults()
	pt.Learn.Defaults()
	pt.CHL.Defaults()
	pt.Trace.Defaults()
	switch pt.Type {
	case CHLPath:
		pt.CHLDefaults()
	case EcCa1Path:
		pt.EcCa1Defaults()
	default:
	}
	pt.GScale = 1
}

// UpdateParams updates all params given any changes that might have been made to individual values
func (pt *Path) UpdateParams() {
	pt.WtScale.Update()
	pt.Learn.Update()
	pt.Learn.LrateInit = pt.Learn.Lrate
	if pt.CHL.On {
		pt.Learn.WtSig.SoftBound = false
	}
	pt.CHL.Update()
	pt.Trace.Update()
}

func (pt *Path) ShouldDisplay(field string) bool {
	switch field {
	case "CHL":
		return pt.Type == CHLPath
	case "Trace":
		return pt.Type == MatrixPath
	default:
		return true
	}
	return true
}

// AllParams returns a listing of all parameters in the Layer
func (pt *Path) AllParams() string {
	str := "///////////////////////////////////////////////////\nPath: " + pt.Name + "\n"
	b, _ := json.MarshalIndent(&pt.WtInit, "", " ")
	str += "WtInit: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pt.WtScale, "", " ")
	str += "WtScale: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pt.Learn, "", " ")
	str += "Learn: {\n " + strings.Replace(JsonToParams(b), " XCal: {", "\n  XCal: {", -1)
	return str
}

func (pt *Path) SynVarNames() []string {
	return SynapseVars
}

// SynVarProps returns properties for variables
func (pt *Path) SynVarProps() map[string]string {
	return SynapseVarProps
}

// SynIndex returns the index of the synapse between given send, recv unit indexes
// (1D, flat indexes). Returns -1 if synapse not found between these two neurons.
// Requires searching within connections for receiving unit.
func (pt *Path) SynIndex(sidx, ridx int) int {
	nc := int(pt.SConN[sidx])
	st := int(pt.SConIndexSt[sidx])
	for ci := 0; ci < nc; ci++ {
		ri := int(pt.SConIndex[st+ci])
		if ri != ridx {
			continue
		}
		return int(st + ci)
	}
	return -1
}

// SynVarIndex returns the index of given variable within the synapse,
// according to *this path's* SynVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (pt *Path) SynVarIndex(varNm string) (int, error) {
	return SynapseVarByName(varNm)
}

// SynVarNum returns the number of synapse-level variables
// for this path.  This is needed for extending indexes in derived types.
func (pt *Path) SynVarNum() int {
	return len(SynapseVars)
}

// Numsyns returns the number of synapses for this path.
// This is the max idx for SynValue1D
// and the number of vals set by SynValues.
func (pt *Path) NumSyns() int {
	return len(pt.Syns)
}

// SynVal1D returns value of given variable index (from SynVarIndex)
// on given SynIndex.  Returns NaN on invalid index.
// This is the core synapse var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (pt *Path) SynValue1D(varIndex int, synIndex int) float32 {
	if synIndex < 0 || synIndex >= len(pt.Syns) {
		return math32.NaN()
	}
	if varIndex < 0 || varIndex >= pt.SynVarNum() {
		return math32.NaN()
	}
	sy := &pt.Syns[synIndex]
	return sy.VarByIndex(varIndex)
}

// SynValues sets values of given variable name for each synapse,
// using the natural ordering of the synapses (sender based for Leabra),
// into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (pt *Path) SynValues(vals *[]float32, varNm string) error {
	vidx, err := pt.SynVarIndex(varNm)
	if err != nil {
		return err
	}
	ns := len(pt.Syns)
	if *vals == nil || cap(*vals) < ns {
		*vals = make([]float32, ns)
	} else if len(*vals) < ns {
		*vals = (*vals)[0:ns]
	}
	for i := range pt.Syns {
		(*vals)[i] = pt.SynValue1D(vidx, i)
	}
	return nil
}

// SynVal returns value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes).
// Returns math32.NaN() for access errors (see SynValTry for error message)
func (pt *Path) SynValue(varNm string, sidx, ridx int) float32 {
	vidx, err := pt.SynVarIndex(varNm)
	if err != nil {
		return math32.NaN()
	}
	synIndex := pt.SynIndex(sidx, ridx)
	return pt.SynValue1D(vidx, synIndex)
}

// SetSynVal sets value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes)
// returns error for access errors.
func (pt *Path) SetSynValue(varNm string, sidx, ridx int, val float32) error {
	vidx, err := pt.SynVarIndex(varNm)
	if err != nil {
		return err
	}
	synIndex := pt.SynIndex(sidx, ridx)
	if synIndex < 0 || synIndex >= len(pt.Syns) {
		return err
	}
	sy := &pt.Syns[synIndex]
	sy.SetVarByIndex(vidx, val)
	if varNm == "Wt" {
		pt.Learn.LWtFromWt(sy)
	}
	return nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWeightsJSON writes the weights from this pathway from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (pt *Path) WriteWeightsJSON(w io.Writer, depth int) {
	slay := pt.Send
	rlay := pt.Recv
	nr := len(rlay.Neurons)
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"From\": %q,\n", slay.Name)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"MetaData\": {\n")))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"GScale\": \"%g\"\n", pt.GScale)))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("},\n"))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Rs\": [\n")))
	depth++
	for ri := 0; ri < nr; ri++ {
		nc := int(pt.RConN[ri])
		st := int(pt.RConIndexSt[ri])
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("{\n"))
		depth++
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"Ri\": %v,\n", ri)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"N\": %v,\n", nc)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Si\": [ "))
		for ci := 0; ci < nc; ci++ {
			si := pt.RConIndex[st+ci]
			w.Write([]byte(fmt.Sprintf("%v", si)))
			if ci == nc-1 {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("],\n"))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Wt\": [ "))
		for ci := 0; ci < nc; ci++ {
			rsi := pt.RSynIndex[st+ci]
			sy := &pt.Syns[rsi]
			w.Write([]byte(strconv.FormatFloat(float64(sy.Wt), 'g', weights.Prec, 32)))
			if ci == nc-1 {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("]\n"))
		depth--
		w.Write(indent.TabBytes(depth))
		if ri == nr-1 {
			w.Write([]byte("}\n"))
		} else {
			w.Write([]byte("},\n"))
		}
	}
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("]\n"))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("}")) // note: leave unterminated as outer loop needs to add , or just \n depending
}

// SetWeights sets the weights for this pathway from weights.Path decoded values
func (pt *Path) SetWeights(pw *weights.Path) error {
	if pw.MetaData != nil {
		if gs, ok := pw.MetaData["GScale"]; ok {
			pv, _ := strconv.ParseFloat(gs, 32)
			pt.GScale = float32(pv)
		}
	}
	var err error
	for i := range pw.Rs {
		pr := &pw.Rs[i]
		for si := range pr.Si {
			er := pt.SetSynValue("Wt", pr.Si[si], pr.Ri, pr.Wt[si]) // updates lin wt
			if er != nil {
				err = er
			}
		}
	}
	return err
}

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (pt *Path) Connect(slay, rlay *Layer, pat paths.Pattern, typ PathTypes) {
	pt.Send = slay
	pt.Recv = rlay
	pt.Pattern = pat
	pt.Type = typ
	pt.Name = pt.Send.Name + "To" + pt.Recv.Name
}

// Validate tests for non-nil settings for the pathway -- returns error
// message or nil if no problems (and logs them if logmsg = true)
func (pt *Path) Validate(logmsg bool) error {
	emsg := ""
	if pt.Pattern == nil {
		emsg += "Pat is nil; "
	}
	if pt.Recv == nil {
		emsg += "Recv is nil; "
	}
	if pt.Send == nil {
		emsg += "Send is nil; "
	}
	if emsg != "" {
		err := errors.New(emsg)
		if logmsg {
			log.Println(emsg)
		}
		return err
	}
	return nil
}

// Build constructs the full connectivity among the layers
// as specified in this pathway.
// Calls Validate and returns false if invalid.
// Pattern.Connect is called to get the pattern of the connection.
// Then the connection indexes are configured according to that pattern.
func (pt *Path) Build() error {
	if pt.Off {
		return nil
	}
	err := pt.Validate(true)
	if err != nil {
		return err
	}
	ssh := &pt.Send.Shape
	rsh := &pt.Recv.Shape
	sendn, recvn, cons := pt.Pattern.Connect(ssh, rsh, pt.Recv == pt.Send)
	slen := ssh.Len()
	rlen := rsh.Len()
	tcons := pt.SetNIndexSt(&pt.SConN, &pt.SConNAvgMax, &pt.SConIndexSt, sendn)
	tconr := pt.SetNIndexSt(&pt.RConN, &pt.RConNAvgMax, &pt.RConIndexSt, recvn)
	if tconr != tcons {
		log.Printf("%v programmer error: total recv cons %v != total send cons %v\n", pt.String(), tconr, tcons)
	}
	pt.RConIndex = make([]int32, tconr)
	pt.RSynIndex = make([]int32, tconr)
	pt.SConIndex = make([]int32, tcons)

	sconN := make([]int32, slen) // temporary mem needed to tracks cur n of sending cons

	cbits := cons.Values
	for ri := 0; ri < rlen; ri++ {
		rbi := ri * slen     // recv bit index
		rtcn := pt.RConN[ri] // number of cons
		rst := pt.RConIndexSt[ri]
		rci := int32(0)
		for si := 0; si < slen; si++ {
			if !cbits.Index(rbi + si) { // no connection
				continue
			}
			sst := pt.SConIndexSt[si]
			if rci >= rtcn {
				log.Printf("%v programmer error: recv target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pt.String(), rtcn, ri, si)
				break
			}
			pt.RConIndex[rst+rci] = int32(si)

			sci := sconN[si]
			stcn := pt.SConN[si]
			if sci >= stcn {
				log.Printf("%v programmer error: send target total con number: %v exceeded at recv idx: %v, send idx: %v\n", pt.String(), stcn, ri, si)
				break
			}
			pt.SConIndex[sst+sci] = int32(ri)
			pt.RSynIndex[rst+rci] = sst + sci
			(sconN[si])++
			rci++
		}
	}
	pt.Syns = make([]Synapse, len(pt.SConIndex))
	pt.GInc = make([]float32, rlen)
	pt.GeRaw = make([]float32, rlen)
	pt.WbRecv = make([]WtBalRecvPath, rlen)
	return nil
}

// SetNIndexSt sets the *ConN and *ConIndexSt values given n tensor from Pat.
// Returns total number of connections for this direction.
func (pt *Path) SetNIndexSt(n *[]int32, avgmax *minmax.AvgMax32, idxst *[]int32, tn *tensor.Int32) int32 {
	ln := tn.Len()
	tnv := tn.Values
	*n = make([]int32, ln)
	*idxst = make([]int32, ln)
	idx := int32(0)
	avgmax.Init()
	for i := 0; i < ln; i++ {
		nv := tnv[i]
		(*n)[i] = nv
		(*idxst)[i] = idx
		idx += nv
		avgmax.UpdateValue(float32(nv), int32(i))
	}
	avgmax.CalcAvg()
	return idx
}

// String satisfies fmt.Stringer for path
func (pt *Path) String() string {
	str := ""
	if pt.Recv == nil {
		str += "recv=nil; "
	} else {
		str += pt.Recv.Name + " <- "
	}
	if pt.Send == nil {
		str += "send=nil"
	} else {
		str += pt.Send.Name
	}
	if pt.Pattern == nil {
		str += " Pat=nil"
	} else {
		str += " Pat=" + pt.Pattern.Name()
	}
	return str
}
