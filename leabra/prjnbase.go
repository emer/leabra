// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"errors"
	"log"

	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/minmax"
	"github.com/goki/gi/giv"
)

// PrjnBase contains the basic structural information for specifying a projection of synaptic
// connections between two layers, and maintaining all the synaptic connection-level data.
// The exact same struct object is added to the Recv and Send layers, and it manages everything
// about the connectivity, and methods on the Prjn handle all the relevant computation.
type PrjnBase struct {

	// we need a pointer to ourselves as an LeabraPrjn, which can always be used to extract the true underlying type of object when prjn is embedded in other structs -- function receivers do not have this ability so this is necessary.
	LeabraPrj LeabraPrjn `copy:"-" json:"-" xml:"-" view:"-"`

	// inactivate this projection -- allows for easy experimentation
	Off bool

	// Class is for applying parameter styles, can be space separated multple tags
	Cls string

	// can record notes about this projection here
	Notes string

	// sending layer for this projection
	Send emer.Layer

	// receiving layer for this projection -- the emer.Layer interface can be converted to the specific Layer type you are using, e.g., rlay := prjn.Recv.(*leabra.Layer)
	Recv emer.Layer

	// pattern of connectivity
	Pat prjn.Pattern

	// type of projection -- Forward, Back, Lateral, or extended type in specialized algorithms -- matches against .Cls parameter styles (e.g., .Back etc)
	Typ emer.PrjnType

	// number of recv connections for each neuron in the receiving layer, as a flat list
	RConN []int32 `view:"-"`

	// average and maximum number of recv connections in the receiving layer
	RConNAvgMax minmax.AvgMax32 `inactive:"+" view:"inline"`

	// starting index into ConIdx list for each neuron in receiving layer -- just a list incremented by ConN
	RConIdxSt []int32 `view:"-"`

	// index of other neuron on sending side of projection, ordered by the receiving layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that
	RConIdx []int32 `view:"-"`

	// index of synaptic state values for each recv unit x connection, for the receiver projection which does not own the synapses, and instead indexes into sender-ordered list
	RSynIdx []int32 `view:"-"`

	// number of sending connections for each neuron in the sending layer, as a flat list
	SConN []int32 `view:"-"`

	// average and maximum number of sending connections in the sending layer
	SConNAvgMax minmax.AvgMax32 `inactive:"+" view:"inline"`

	// starting index into ConIdx list for each neuron in sending layer -- just a list incremented by ConN
	SConIdxSt []int32 `view:"-"`

	// index of other neuron on receiving side of projection, ordered by the sending layer's order of units as the outer loop (each start is in ConIdxSt), and then by the sending layer's units within that
	SConIdx []int32 `view:"-"`
}

// emer.Prjn interface

// Init MUST be called to initialize the prjn's pointer to itself as an emer.Prjn
// which enables the proper interface methods to be called.
func (ps *PrjnBase) Init(prjn emer.Prjn) {
	ps.LeabraPrj = prjn.(LeabraPrjn)
}

func (ps *PrjnBase) TypeName() string { return "Prjn" } // always, for params..
func (ps *PrjnBase) Class() string    { return ps.LeabraPrj.PrjnTypeName() + " " + ps.Cls }
func (ps *PrjnBase) Name() string {
	return ps.Send.Name() + "To" + ps.Recv.Name()
}
func (pj *PrjnBase) SetClass(cls string) emer.Prjn { pj.Cls = cls; return pj.LeabraPrj }
func (pj *PrjnBase) AddClass(cls string)           { pj.Cls = params.AddClass(pj.Cls, cls) }
func (ps *PrjnBase) Label() string                 { return ps.Name() }
func (ps *PrjnBase) RecvLay() emer.Layer           { return ps.Recv }
func (ps *PrjnBase) SendLay() emer.Layer           { return ps.Send }
func (ps *PrjnBase) Pattern() prjn.Pattern         { return ps.Pat }
func (ps *PrjnBase) Type() emer.PrjnType           { return ps.Typ }
func (ps *PrjnBase) PrjnTypeName() string          { return ps.Typ.String() }

func (ps *PrjnBase) IsOff() bool {
	return ps.Off || ps.Recv.IsOff() || ps.Send.IsOff()
}
func (ps *PrjnBase) SetOff(off bool) { ps.Off = off }

// Connect sets the connectivity between two layers and the pattern to use in interconnecting them
func (ps *PrjnBase) Connect(slay, rlay emer.Layer, pat prjn.Pattern, typ emer.PrjnType) {
	ps.Send = slay
	ps.Recv = rlay
	ps.Pat = pat
	ps.Typ = typ
}

// Validate tests for non-nil settings for the projection -- returns error
// message or nil if no problems (and logs them if logmsg = true)
func (ps *PrjnBase) Validate(logmsg bool) error {
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

// BuildStru constructs the full connectivity among the layers as specified in this projection.
// Calls Validate and returns false if invalid.
// Pat.Connect is called to get the pattern of the connection.
// Then the connection indexes are configured according to that pattern.
func (ps *PrjnBase) BuildStru() error {
	if ps.Off {
		return nil
	}
	err := ps.Validate(true)
	if err != nil {
		return err
	}
	ssh := ps.Send.Shape()
	rsh := ps.Recv.Shape()
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
func (ps *PrjnBase) SetNIdxSt(n *[]int32, avgmax *minmax.AvgMax32, idxst *[]int32, tn *etensor.Int32) int32 {
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

// String satisfies fmt.Stringer for prjn
func (ps *PrjnBase) String() string {
	str := ""
	if ps.Recv == nil {
		str += "recv=nil; "
	} else {
		str += ps.Recv.Name() + " <- "
	}
	if ps.Send == nil {
		str += "send=nil"
	} else {
		str += ps.Send.Name()
	}
	if ps.Pat == nil {
		str += " Pat=nil"
	} else {
		str += " Pat=" + ps.Pat.Name()
	}
	return str
}

// ApplyParams applies given parameter style Sheet to this projection.
// Calls UpdateParams if anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ps *PrjnBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	app, err := pars.Apply(ps.LeabraPrj, setMsg) // essential to go through LeabraPrj
	if app {
		ps.LeabraPrj.UpdateParams()
	}
	return app, err
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (ps *PrjnBase) NonDefaultParams() string {
	pth := ps.Recv.Name() + "." + ps.Name() // redundant but clearer..
	nds := giv.StructNonDefFieldsStr(ps.LeabraPrj, pth)
	return nds
}
