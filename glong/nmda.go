// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// NMDAParams control the NMDA dynamics in PFC Maint neurons, based on Brunel & Wang (2001)
// parameters.  We have to do some things to make it work for rate code neurons..
type NMDAParams struct {

	// [def: 0.4] extra contribution to Vm associated with action potentials, on average -- produces key nonlinearity associated with spiking, from backpropagating action potentials.  0.4 seems good..
	ActVm float32 `def:"0.4" desc:"extra contribution to Vm associated with action potentials, on average -- produces key nonlinearity associated with spiking, from backpropagating action potentials.  0.4 seems good.."`

	// cycle upon which to start updating AlphaMax value
	AlphaMaxCyc int `desc:"cycle upon which to start updating AlphaMax value"`

	// [def: 100] decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential
	Tau float32 `def:"100" desc:"decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential"`

	// strength of NMDA current -- 0.02 is just over level sufficient to maintain in face of completely blank input
	Gbar float32 `desc:"strength of NMDA current -- 0.02 is just over level sufficient to maintain in face of completely blank input"`
}

func (np *NMDAParams) Defaults() {
	np.ActVm = 0.4
	np.AlphaMaxCyc = 30
	np.Tau = 100
	np.Gbar = 0.02
}

// VmEff returns the effective Vm value including backpropagating action potentials from ActVm
func (np *NMDAParams) VmEff(vm, act float32) float32 {
	return vm + np.ActVm*act
}

// GFmV returns the NMDA conductance as a function of normalized membrane potential
func (np *NMDAParams) GFmV(v float32) float32 {
	vbio := mat32.Min(v*100-100, 0) // critical to not go past 0
	return 1 / (1 + 0.28*mat32.FastExp(-0.062*vbio))
}

// NMDA returns the updated NMDA activation from current NMDA and NMDASyn input
func (np *NMDAParams) NMDA(nmda, nmdaSyn float32) float32 {
	return nmda + nmdaSyn - (nmda / np.Tau)
}

// Gnmda returns the NMDA net conductance from nmda activation and vm
func (np *NMDAParams) Gnmda(nmda, vm float32) float32 {
	return np.Gbar * np.GFmV(vm) * nmda
}

///////////////////////////////////////////////////////////////////////////
// NMDAPrjn

// NMDAPrjn is a projection with NMDA maintenance channels.
// It marks a projection for special treatment in a MaintLayer
// which actually does the NMDA computations.  Excitatory conductance is aggregated
// separately for this projection.
type NMDAPrjn struct {
	leabra.Prjn // access as .Prjn
}

var KiT_NMDAPrjn = kit.Types.AddType(&NMDAPrjn{}, PrjnProps)

func (pj *NMDAPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

func (pj *NMDAPrjn) Type() emer.PrjnType {
	return NMDA
}

func (pj *NMDAPrjn) PrjnTypeName() string {
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

// PrjnType has the GLong extensions to the emer.PrjnType types, for gui
type PrjnType emer.PrjnType

//go:generate stringer -type=PrjnType

var KiT_PrjnType = kit.Enums.AddEnumExt(emer.KiT_PrjnType, PrjnTypeN, kit.NotBitFlag, nil)

// The GLong prjn types
const (
	// NMDAPrjn are projections that have strong NMDA channels supporting maintenance
	NMDA emer.PrjnType = emer.PrjnType(emer.PrjnTypeN) + iota
)

// gui versions
const (
	NMDA_ PrjnType = PrjnType(emer.PrjnTypeN) + iota
	PrjnTypeN
)

var PrjnProps = ki.Props{
	"EnumType:Typ": KiT_PrjnType,
}
