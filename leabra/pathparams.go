// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"github.com/emer/emergent/v2/params"
)

// PathParams contains all of the path parameters, which
// implement the Leabra algorithm at the path level.
type PathParams struct {
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

	// Path points back to our path.
	Path *Path
}

func (pt *PathParams) Defaults() {
	pt.WtInit.Defaults()
	pt.WtScale.Defaults()
	pt.Learn.Defaults()
	pt.CHL.Defaults()
	pt.Trace.Defaults()
	pt.DefaultsForType()
}

func (pt *PathParams) DefaultsForType() {
	switch pt.Type {
	case CHLPath:
		pt.CHLDefaults()
	case EcCa1Path:
		pt.EcCa1Defaults()
	case TDPredPath:
		pt.TDPredDefaults()
	case RWPath:
		pt.RWDefaults()
	case MatrixPath:
		pt.MatrixDefaults()
	case DaHebbPath:
		pt.DaHebbDefaults()
	}
}

// UpdateParams updates all params given any changes that might have been made to individual values
func (pt *PathParams) UpdateParams() {
	pt.WtScale.Update()
	pt.Learn.Update()
	pt.Learn.LrateInit = pt.Learn.Lrate
	if pt.Type == CHLPath && pt.CHL.On {
		pt.Learn.WtSig.SoftBound = false
	}
	pt.CHL.Update()
	pt.Trace.Update()
}

func (pt *PathParams) ShouldDisplay(field string) bool {
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

// ParamsString returns a listing of all parameters in the Layer and
// pathways within the layer. If nonDefault is true, only report those
// not at their default values.
func (pt *PathParams) ParamsString(nonDefault bool) string {
	return params.PrintStruct(pt, 1, func(path string, ft reflect.StructField, fv any) bool {
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
		switch path {
		case "WtInit", "WtScale", "Learn":
			return true
		case "CHL":
			return pt.Type == CHLPath
		case "Trace":
			return pt.Type == MatrixPath
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
func (pt *PathParams) StyleClass() string {
	return pt.Type.String() + " " + pt.Path.Class
}

// StyleName implements the [params.Styler] interface for parameter setting,
// and must only be called after the network has been built, and is current,
// because it uses the global CurrentNetwork variable.
func (pt *PathParams) StyleName() string {
	return pt.Path.Name
}
