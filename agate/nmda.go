// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// NMDAMaintPrjn is a projection with strong NMDA maintenance channels.
// It essentially marks a projection for special treatment in a MaintLayer
// which actually does the NMDA computations.  Excitatory conductance is aggregated
// separately for this projection.
type NMDAMaintPrjn struct {
	leabra.Prjn // access as .Prjn
}

var KiT_NMDAMaintPrjn = kit.Types.AddType(&NMDAMaintPrjn{}, PrjnProps)

func (pj *NMDAMaintPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.WtInit.Mean = 0.5
	pj.WtInit.Var = 0
	// todo: learning off by default?
}

func (pj *NMDAMaintPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

func (pj *NMDAMaintPrjn) Type() emer.PrjnType {
	return NMDAMaint
}

func (pj *NMDAMaintPrjn) PrjnTypeName() string {
	if pj.Typ < emer.PrjnTypeN {
		return pj.Typ.String()
	}
	ptyp := PrjnType(pj.Typ)
	ts := ptyp.String()
	sz := len(ts)
	if sz > 0 {
		return ts[:sz-1] // cut off trailing _
	}
	return ""
}

//////////////////////////////////////////////////////////////////////////////////////
//  PrjnType

// PrjnType has the AGate extensions to the emer.PrjnType types, for gui
type PrjnType deep.PrjnType

//go:generate stringer -type=PrjnType

var KiT_PrjnType = kit.Enums.AddEnumExt(deep.KiT_PrjnType, PrjnTypeN, kit.NotBitFlag, nil)

// The AGate prjn types
const (
	// NMDAMaint are projections that have strong NMDA channels supporting maintenance
	NMDAMaint emer.PrjnType = emer.PrjnType(deep.PrjnTypeN) + iota
)

// gui versions
const (
	NMDAMaint_ PrjnType = PrjnType(deep.PrjnTypeN) + iota
	PrjnTypeN
)

var PrjnProps = ki.Props{
	"EnumType:Typ": KiT_PrjnType,
}
