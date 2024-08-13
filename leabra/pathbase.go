// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"errors"
	"log"

	"cogentcore.org/core/views"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/path"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/minmax"
)

// Path contains the basic structural information for a pathway of synaptic
// connections between two layers, including all the synaptic connection-level data.
type Path struct {
	emer.PathBase

	// sending layer for this pathway.
	Send *Layer

	// receiving layer for this pathway.
	Recv *Layer

	// type of pathway.
	Type PathTypes

	// number of recv connections for each neuron in the receiving layer, as a flat list
	RConN []int32 `display:"-"`

	// average and maximum number of recv connections in the receiving layer
	RConNAvgMax minmax.AvgMax32 `inactive:"+" view:"inline"`

	// starting index into ConIndex list for each neuron in receiving layer -- just a list incremented by ConN
	RConIndexSt []int32 `display:"-"`

	// index of other neuron on sending side of pathway, ordered by the receiving layer's order of units as the outer loop (each start is in ConIndexSt), and then by the sending layer's units within that
	RConIndex []int32 `display:"-"`

	// index of synaptic state values for each recv unit x connection, for the receiver pathway which does not own the synapses, and instead indexes into sender-ordered list
	RSynIndex []int32 `display:"-"`

	// number of sending connections for each neuron in the sending layer, as a flat list
	SConN []int32 `display:"-"`

	// average and maximum number of sending connections in the sending layer
	SConNAvgMax minmax.AvgMax32 `inactive:"+" view:"inline"`

	// starting index into ConIndex list for each neuron in sending layer -- just a list incremented by ConN
	SConIndexSt []int32 `display:"-"`

	// index of other neuron on receiving side of pathway, ordered by the sending layer's order of units as the outer loop (each start is in ConIndexSt), and then by the sending layer's units within that
	SConIndex []int32 `display:"-"`
}

// emer.Path interface

func (pt *Path) StyleObject() any      { return pt }
func (pt *Path) RecvLayer() emer.Layer           { return pt.Recv }
func (pt *Path) SendLayer() emer.Layer           { return pt.Send }
func (pt *Path) TypeName() string { return pt.type.String() }

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (pt *Path) Connect(slay, rlay *Layer, pat path.Pattern, typ PathType) {
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
	if pt.Pat == nil {
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

// BuildStru constructs the full connectivity among the layers as specified in this pathway.
// Calls Validate and returns false if invalid.
// Pat.Connect is called to get the pattern of the connection.
// Then the connection indexes are configured according to that pattern.
func (pt *Path) BuildStru() error {
	if pt.Off {
		return nil
	}
	err := pt.Validate(true)
	if err != nil {
		return err
	}
	ssh := pt.Send.Shape()
	rsh := pt.Recv.Shape()
	sendn, recvn, cons := pt.Pat.Connect(ssh, rsh, pt.Recv == pt.Send)
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
	return nil
}

// SetNIndexSt sets the *ConN and *ConIndexSt values given n tensor from Pat.
// Returns total number of connections for this direction.
func (pt *Path) SetNIndexSt(n *[]int32, avgmax *minmax.AvgMax32, idxst *[]int32, tn *etensor.Int32) int32 {
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
		str += pt.Recv.Name() + " <- "
	}
	if pt.Send == nil {
		str += "send=nil"
	} else {
		str += pt.Send.Name()
	}
	if pt.Pat == nil {
		str += " Pat=nil"
	} else {
		str += " Pat=" + pt.Pat.Name()
	}
	return str
}


