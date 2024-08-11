// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"errors"
	"log"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/path"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/goki/gi/giv"
)

//////////////////////////////////////////////////////////////////////////////////////
//  PathType

// PathType is the type of the pathway (extensible for more specialized algorithms).
// Class parameter styles automatically key off of these types.
type PathType int32 //enums:enum

// The pathway types
const (
	// Forward is a feedforward, bottom-up pathway from sensory inputs to higher layers
	Forward PathType = iota

	// Back is a feedback, top-down pathway from higher layers back to lower layers
	Back

	// Lateral is a lateral pathway within the same layer / area
	Lateral

	// Inhib is an inhibitory pathway that drives inhibitory synaptic inputs instead of excitatory
	Inhib
)

// PathStru contains the basic structural information for specifying a pathway of synaptic
// connections between two layers, and maintaining all the synaptic connection-level data.
// The exact same struct object is added to the Recv and Send layers, and it manages everything
// about the connectivity, and methods on the Path handle all the relevant computation.
type PathStru struct {

	// [view: -] we need a pointer to ourselves as an LeabraPath, which can always be used to extract the true underlying type of object when path is embedded in other structs -- function receivers do not have this ability so this is necessary.
	LeabraPrj LeabraPath `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an LeabraPath, which can always be used to extract the true underlying type of object when path is embedded in other structs -- function receivers do not have this ability so this is necessary."`

	// inactivate this pathway -- allows for easy experimentation
	Off bool `desc:"inactivate this pathway -- allows for easy experimentation"`

	// Class is for applying parameter styles, can be space separated multple tags
	Cls string `desc:"Class is for applying parameter styles, can be space separated multple tags"`

	// can record notes about this pathway here
	Notes string `desc:"can record notes about this pathway here"`

	// sending layer for this pathway
	Send emer.Layer `desc:"sending layer for this pathway"`

	// receiving layer for this pathway -- the emer.Layer interface can be converted to the specific Layer type you are using, e.g., rlay := path.Recv.(*leabra.Layer)
	Recv emer.Layer `desc:"receiving layer for this pathway -- the emer.Layer interface can be converted to the specific Layer type you are using, e.g., rlay := path.Recv.(*leabra.Layer)"`

	// pattern of connectivity
	Pat path.Pattern `desc:"pattern of connectivity"`

	// type of pathway -- Forward, Back, Lateral, or extended type in specialized algorithms -- matches against .Cls parameter styles (e.g., .Back etc)
	Typ emer.PathType `desc:"type of pathway -- Forward, Back, Lateral, or extended type in specialized algorithms -- matches against .Cls parameter styles (e.g., .Back etc)"`

	// [view: -] number of recv connections for each neuron in the receiving layer, as a flat list
	RConN []int32 `view:"-" desc:"number of recv connections for each neuron in the receiving layer, as a flat list"`

	// [view: inline] average and maximum number of recv connections in the receiving layer
	RConNAvgMax minmax.AvgMax32 `inactive:"+" view:"inline" desc:"average and maximum number of recv connections in the receiving layer"`

	// [view: -] starting index into ConIdx list for each neuron in receiving layer -- just a list incremented by ConN
	RConIdxSt []int32 `view:"-" desc:"starting index into ConIdx list for each neuron in receiving layer -- just a list incremented by ConN"`

	// [view: -] index of other neuron on sending side of pathway, ordered by the receiving layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that
	RConIdx []int32 `view:"-" desc:"index of other neuron on sending side of pathway, ordered by the receiving layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that"`

	// [view: -] index of synaptic state values for each recv unit x connection, for the receiver pathway which does not own the synapses, and instead indexes into sender-ordered list
	RSynIdx []int32 `view:"-" desc:"index of synaptic state values for each recv unit x connection, for the receiver pathway which does not own the synapses, and instead indexes into sender-ordered list"`

	// [view: -] number of sending connections for each neuron in the sending layer, as a flat list
	SConN []int32 `view:"-" desc:"number of sending connections for each neuron in the sending layer, as a flat list"`

	// [view: inline] average and maximum number of sending connections in the sending layer
	SConNAvgMax minmax.AvgMax32 `inactive:"+" view:"inline" desc:"average and maximum number of sending connections in the sending layer"`

	// [view: -] starting index into ConIdx list for each neuron in sending layer -- just a list incremented by ConN
	SConIdxSt []int32 `view:"-" desc:"starting index into ConIdx list for each neuron in sending layer -- just a list incremented by ConN"`

	// [view: -] index of other neuron on receiving side of pathway, ordered by the sending layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that
	SConIdx []int32 `view:"-" desc:"index of other neuron on receiving side of pathway, ordered by the sending layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that"`
}

// emer.Path interface

// Init MUST be called to initialize the path's pointer to itself as an emer.Path
// which enables the proper interface methods to be called.
func (ps *PathStru) Init(path emer.Path) {
	ps.LeabraPrj = path.(LeabraPath)
}

func (ps *PathStru) TypeName() string { return "Path" } // always, for params..
func (ps *PathStru) Class() string    { return ps.LeabraPrj.PathTypeName() + " " + ps.Cls }
func (ps *PathStru) Name() string {
	return ps.Send.Name + "To" + ps.Recv.Name
}
func (ps *PathStru) Label() string         { return ps.Name }
func (ps *PathStru) RecvLayer() emer.Layer { return ps.Recv }
func (ps *PathStru) SendLayer() emer.Layer { return ps.Send }
func (ps *PathStru) Pattern() path.Pattern { return ps.Pat }
func (ps *PathStru) Type() emer.PathType   { return ps.Typ }
func (ps *PathStru) PathTypeName() string  { return ps.Typ.String() }

func (ps *PathStru) IsOff() bool {
	return ps.Off || ps.Recv.Off || ps.Send.Off
}
func (ps *PathStru) SetOff(off bool) { ps.Off = off }

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (ps *PathStru) Connect(slay, rlay emer.Layer, pat path.Pattern, typ emer.PathType) {
	ps.Send = slay
	ps.Recv = rlay
	ps.Pat = pat
	ps.Typ = typ
}

// Validate tests for non-nil settings for the pathway -- returns error
// message or nil if no problems (and logs them if logmsg = true)
func (ps *PathStru) Validate(logmsg bool) error {
	emsg := ""
	if ps.Pat == nil {
		emsg += "Pat is nil; "
	}
	if ps.Recv == nil {
		emsg += "Recv is nil; "
	}
	if ps.Send == nil {
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
func (ps *PathStru) BuildStru() error {
	if ps.Off {
		return nil
	}
	err := ps.Validate(true)
	if err != nil {
		return err
	}
	ssh := ps.Send.Shape
	rsh := ps.Recv.Shape
	sendn, recvn, cons := ps.Pat.Connect(ssh, rsh, ps.Recv == ps.Send)
	slen := ssh.Len()
	rlen := rsh.Len()
	tcons := ps.SetNIdxSt(&ps.SConN, &ps.SConNAvgMax, &ps.SConIdxSt, sendn)
	tconr := ps.SetNIdxSt(&ps.RConN, &ps.RConNAvgMax, &ps.RConIdxSt, recvn)
	if tconr != tcons {
		log.Printf("%v programmer error: total recv cons %v != total send cons %v\n", ps.String(), tconr, tcons)
	}
	ps.RConIdx = make([]int32, tconr)
	ps.RSynIdx = make([]int32, tconr)
	ps.SConIdx = make([]int32, tcons)

	sconN := make([]int32, slen) // temporary mem needed to tracks cur n of sending cons

	cbits := cons.Values
	for ri := 0; ri < rlen; ri++ {
		rbi := ri * slen     // recv bit index
		rtcn := ps.RConN[ri] // number of cons
		rst := ps.RConIdxSt[ri]
		rci := int32(0)
		for si := 0; si < slen; si++ {
			if !cbits.Index(rbi + si) { // no connection
				continue
			}
			sst := ps.SConIdxSt[si]
			if rci >= rtcn {
				log.Printf("%v programmer error: recv target total con number: %v exceeded at recv idx: %v, send idx: %v\n", ps.String(), rtcn, ri, si)
				break
			}
			ps.RConIdx[rst+rci] = int32(si)

			sci := sconN[si]
			stcn := ps.SConN[si]
			if sci >= stcn {
				log.Printf("%v programmer error: send target total con number: %v exceeded at recv idx: %v, send idx: %v\n", ps.String(), stcn, ri, si)
				break
			}
			ps.SConIdx[sst+sci] = int32(ri)
			ps.RSynIdx[rst+rci] = sst + sci
			(sconN[si])++
			rci++
		}
	}
	return nil
}

// SetNIdxSt sets the *ConN and *ConIdxSt values given n tensor from Pat.
// Returns total number of connections for this direction.
func (ps *PathStru) SetNIdxSt(n *[]int32, avgmax *minmax.AvgMax32, idxst *[]int32, tn *etensor.Int32) int32 {
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
		avgmax.UpdateVal(float32(nv), int32(i))
	}
	avgmax.CalcAvg()
	return idx
}

// String satisfies fmt.Stringer for path
func (ps *PathStru) String() string {
	str := ""
	if ps.Recv == nil {
		str += "recv=nil; "
	} else {
		str += ps.Recv.Name + " <- "
	}
	if ps.Send == nil {
		str += "send=nil"
	} else {
		str += ps.Send.Name
	}
	if ps.Pat == nil {
		str += " Pat=nil"
	} else {
		str += " Pat=" + ps.Pat.Name
	}
	return str
}

// ApplyParams applies given parameter style Sheet to this pathway.
// Calls UpdateParams if anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ps *PathStru) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	app, err := pars.Apply(ps.LeabraPrj, setMsg) // essential to go through LeabraPrj
	if app {
		ps.LeabraPrj.UpdateParams()
	}
	return app, err
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (ps *PathStru) NonDefaultParams() string {
	pth := ps.Recv.Name + "." + ps.Name // redundant but clearer..
	nds := giv.StructNonDefFieldsStr(ps.LeabraPrj, pth)
	return nds
}
