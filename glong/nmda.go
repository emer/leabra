// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package glong

import (
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

// NMDAParams control the NMDA dynamics in PFC Maint neurons, based on Brunel & Wang (2001)
// parameters.  We have to do some things to make it work for rate code neurons..
type NMDAParams struct {

	// extra contribution to Vm associated with action potentials, on average -- produces key nonlinearity associated with spiking, from backpropagating action potentials.  0.4 seems good..
	ActVm float32 `def:"0.4"`

	// cycle upon which to start updating AlphaMax value
	AlphaMaxCyc int

	// decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential
	Tau float32 `def:"100"`

	// strength of NMDA current -- 0.02 is just over level sufficient to maintain in face of completely blank input
	Gbar float32
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
	vbio := math32.Min(v*100-100, 0) // critical to not go past 0
	return 1 / (1 + 0.28*math32.FastExp(-0.062*vbio))
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
// NMDAPath

// NMDAPath is a pathway with NMDA maintenance channels.
// It marks a pathway for special treatment in a MaintLayer
// which actually does the NMDA computations.  Excitatory conductance is aggregated
// separately for this pathway.
type NMDAPath struct {
	leabra.Path // access as .Path
}

func (pj *NMDAPath) UpdateParams() {
	pj.Path.UpdateParams()
}

func (pj *NMDAPath) Type() emer.PathType {
	return NMDA
}

func (pj *NMDAPath) PathTypeName() string {
	if pj.Typ < emer.PathTypeN {
		return pj.Typ.String()
	}
	ptyp := PathType(pj.Typ)
	ts := ptyp.String()
	sz := len(ts)
	if sz > 0 {
		return ts[:sz-1] // cut off trailing _
	}
	return ""
}

//////////////////////////////////////////////////////////////////////////////////////
//  PathType

// PathType has the GLong extensions to the emer.PathType types, for gui
type PathType emer.PathType //enums:enum

// The GLong path types
const (
	// NMDAPath are pathways that have strong NMDA channels supporting maintenance
	NMDA emer.PathType = emer.PathType(emer.PathTypeN) + iota
)

// gui versions
const (
	NMDA_ PathType = PathType(emer.PathTypeN) + iota
)

// var PathProps = tree.Props{
// 	"EnumType:Typ": KiT_PathType,
// }
