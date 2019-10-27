// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/goki/ki/kit"
)

// MSNLayer represents medium spiny neurons, principal cells of the striatum.
// They are subject to dopamine-based learning, and drive modulation of cortical
// and subcortical function via inhibitory outputs.
type MSNLayer struct {
	ModLayer
	DaR      DaReceptors   `desc: "dominant type of dopamine receptor"`
	MatPat   MatrixPatch   `desc:"matrix (matrisome) or patch (striosome) MSN type. Matrix projects to GPe / GPi,SNr in dorsal striatum and is primarily responsible for gating events, and in ventral are responsive to CS onsets and drive gating in vmPFC. Patch in dorsal shunt dopamine signals, and in ventral are responsible for blocking transient dopamine bursts via shunting and dipping"`
	DorsVent DorsalVentral `desc:"dorsal (dlPFC gating) vs. ventral (vmPFC gating, DA modulation)"`
}

//////////////////////////////////////////////////////////////////////
// Enums

// MatrixPatch for matrix (matrisome) vs. patch (striosome) structure
type MatrixPatch int

//go:generate stringer -type=MatrixPatch

var KiT_MatrixPatch = kit.Enums.AddEnum(MatrixPatchN, false, nil)

func (ev MatrixPatch) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *MatrixPatch) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// Matrix are matrisome type units that project to GPe / GPi,SNr in dorsal striatum and are primarily responsible for gating events, and in ventral are responsive to CS onsets and drive gating in vmPFC
	Matrix MatrixPatch = iota

	// Patch are striosome type units, which in dorsal may shunt dopamine signals, and in ventral are responsible for blocking transient dopamine bursts via shunting and dipping
	Patch

	MatrixPatchN
)

// DorsalVentral for region of striatum
type DorsalVentral int

//go:generate stringer -type=DorsalVentral

var KiT_DorsalVentral = kit.Enums.AddEnum(DorsalVentralN, false, nil)

func (ev DorsalVentral) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *DorsalVentral) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// Dorsal striatum -- projects to GPe / GPi,SNr and drives gating of PFC (Matrix) and modulation of dopamine (Patch)
	Dorsal DorsalVentral = iota

	// Ventral striatum -- projects to VTA, LHB, ventral pallidum -- drives updating of OFC, ACC and modulation of VTA dopamine
	Ventral

	DorsalVentralN
)

// GateTypes for region of striatum
type GateTypes int

//go:generate stringer -type=GateTypes

var KiT_GateTypes = kit.Enums.AddEnum(GateTypesN, false, nil)

func (ev GateTypes) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *GateTypes) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	// Maint is maintenance gating -- toggles active maintenance in PFC
	Maint GateTypes = iota

	// Out is output gating -- drives deep layer activation
	Out

	GateTypesN
)
