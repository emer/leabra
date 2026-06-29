// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"github.com/emer/emergent/v2/params"
)

// LayerParams contains all of the layer parameters, which
// implement the Leabra algorithm at the layer level.
type LayerParams struct {
	// type of layer.
	Type LayerTypes

	// Activation parameters and methods for computing activations.
	Act ActParams `display:"add-fields"`

	// Inhibition parameters and methods for computing layer-level inhibition.
	Inhib InhibParams `display:"add-fields"`

	// Learning parameters and methods that operate at the neuron level.
	Learn LearnNeurParams `display:"add-fields"`

	// Burst has parameters for computing Burst from act, in Superficial layers
	// (but also needed in Deep layers for deep self connections).
	Burst BurstParams `display:"inline"`

	// Pulvinar has parameters for computing Pulvinar plus-phase (outcome)
	// activations based on Burst activation from corresponding driver neuron.
	Pulvinar PulvinarParams `display:"inline"`

	// Drivers are names of SuperLayer(s) that sends 5IB Burst driver
	// inputs to this layer.
	Drivers Drivers

	// RW are Rescorla-Wagner RL learning parameters.
	RW RWParams `display:"inline"`

	// TD are Temporal Differences RL learning parameters.
	TD TDParams `display:"inline"`

	// Matrix BG gating parameters
	Matrix MatrixParams `display:"inline"`

	// PBWM has general PBWM parameters, including the shape
	// of overall Maint + Out gating system that this layer is part of.
	PBWM PBWMParams `display:"inline"`

	// GPiGate are gating parameters determining threshold for gating etc.
	GPiGate GPiGateParams `display:"inline"`

	// CIN cholinergic interneuron parameters.
	CIN CINParams `display:"inline"`

	// PFC Gating parameters
	PFCGate PFCGateParams `display:"inline"`

	// PFC Maintenance parameters
	PFCMaint PFCMaintParams `display:"inline"`

	// PFCDyns dynamic behavior parameters, which provide deterministic
	// control over PFC maintenance dynamics. The rows of PFC units
	// (along Y axis) behave according to corresponding index of Dyns
	// (inner loop is Super Y axis, outer is Dyn types).
	// Ensure Y dim has even multiple of len(Dyns).
	PFCDyns PFCDyns

	// pointer back to our layer
	Layer *Layer
}

func (ly *LayerParams) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Burst.Defaults()
	ly.Pulvinar.Defaults()
	ly.RW.Defaults()
	ly.TD.Defaults()
	ly.Matrix.Defaults()
	ly.PBWM.Defaults()
	ly.GPiGate.Defaults()
	ly.CIN.Defaults()
	ly.PFCGate.Defaults()
	ly.PFCMaint.Defaults()
	ly.Inhib.Layer.On = true
	ly.DefaultsForType()
}

// DefaultsForType sets the default parameter values for a given layer type.
func (ly *LayerParams) DefaultsForType() {
	switch ly.Type {
	case ClampDaLayer:
		ly.ClampDaDefaults()
	case MatrixLayer:
		ly.MatrixDefaults()
	case GPiThalLayer:
		ly.GPiThalDefaults()
	case CINLayer:
	case PFCLayer:
	case PFCDeepLayer:
		ly.PFCDeepDefaults()
	}
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving pathways of this layer
func (ly *LayerParams) UpdateParams() {
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()
	ly.Burst.Update()
	ly.Pulvinar.Update()
	ly.RW.Update()
	ly.TD.Update()
	ly.Matrix.Update()
	ly.PBWM.Update()
	ly.GPiGate.Update()
	ly.CIN.Update()
	ly.PFCGate.Update()
	ly.PFCMaint.Update()
}

func (ly *LayerParams) ShouldDisplay(field string) bool {
	isPBWM := ly.Type == MatrixLayer || ly.Type == GPiThalLayer || ly.Type == CINLayer || ly.Type == PFCLayer || ly.Type == PFCDeepLayer
	switch field {
	case "Burst":
		return ly.Type == SuperLayer || ly.Type == CTLayer
	case "Pulvinar", "Drivers":
		return ly.Type == PulvinarLayer
	case "RW":
		return ly.Type == RWPredLayer || ly.Type == RWDaLayer
	case "TD":
		return ly.Type == TDPredLayer || ly.Type == TDIntegLayer || ly.Type == TDDaLayer
	case "PBWM":
		return isPBWM
	case "SendTo":
		return ly.Type == GPiThalLayer || ly.Type == ClampDaLayer || ly.Type == RWDaLayer || ly.Type == TDDaLayer || ly.Type == CINLayer
	case "Matrix":
		return ly.Type == MatrixLayer
	case "GPiGate":
		return ly.Type == GPiThalLayer
	case "CIN":
		return ly.Type == CINLayer
	case "PFCGate", "PFCMaint":
		return ly.Type == PFCLayer || ly.Type == PFCDeepLayer
	case "PFCDyns":
		return ly.Type == PFCDeepLayer
	default:
		return true
	}
	return true
}

// ParamsString returns a listing of all parameters in the Layer and
// pathways within the layer. If nonDefault is true, only report those
// not at their default values.
func (ly *LayerParams) ParamsString(nonDefault bool) string {
	return params.PrintStruct(ly, 1, func(path string, ft reflect.StructField, fv any) bool {
		if ft.Tag.Get("display") == "-" {
			return false
		}
		if nonDefault {
			if def := ft.Tag.Get("default"); def != "" {
				if reflectx.ValueIsDefault(reflect.ValueOf(fv), def) {
					return false
				}
			} else {
				if reflectx.NonPointerType(ft.Type).Kind() != reflect.Struct {
					return false
				}
			}
		}
		isPBWM := ly.Type == MatrixLayer || ly.Type == GPiThalLayer || ly.Type == CINLayer || ly.Type == PFCLayer || ly.Type == PFCDeepLayer
		switch path {
		case "Act", "Inhib", "Learn":
			return true
		case "Burst":
			return ly.Type == SuperLayer || ly.Type == CTLayer
		case "Pulvinar", "Drivers":
			return ly.Type == PulvinarLayer
		case "RW":
			return ly.Type == RWPredLayer || ly.Type == RWDaLayer
		case "TD":
			return ly.Type == TDPredLayer || ly.Type == TDIntegLayer || ly.Type == TDDaLayer
		case "PBWM":
			return isPBWM
		case "SendTo":
			return ly.Type == GPiThalLayer || ly.Type == ClampDaLayer || ly.Type == RWDaLayer || ly.Type == TDDaLayer || ly.Type == CINLayer
		case "Matrix":
			return ly.Type == MatrixLayer
		case "GPiGate":
			return ly.Type == GPiThalLayer
		case "CIN":
			return ly.Type == CINLayer
		case "PFCGate", "PFCMaint":
			return ly.Type == PFCLayer || ly.Type == PFCDeepLayer
		case "PFCDyns":
			return ly.Type == PFCDeepLayer
		}
		return false
	},
		func(path string, ft reflect.StructField, fv any) string {
			if nonDefault {
				if def := ft.Tag.Get("default"); def != "" {
					return reflectx.ToString(fv) + " [" + def + "]"
				}
			}
			return ""
		})
}

// StyleClass implements the [params.Styler] interface for parameter setting,
// and must only be called after the network has been built, and is current,
// because it uses the global CurrentNetwork variable.
func (ly *LayerParams) StyleClass() string {
	return ly.Type.String() + " " + ly.Layer.Class
}

// StyleName implements the [params.Styler] interface for parameter setting,
// and must only be called after the network has been built, and is current,
// because it uses the global CurrentNetwork variable.
func (ly *LayerParams) StyleName() string {
	return ly.Layer.Name
}
