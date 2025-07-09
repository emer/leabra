// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"reflect"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/lab/base/mpi"
	"github.com/cogentcore/yaegi/interp"
	"github.com/emer/emergent/v2/params"
)

// type aliases for params generic types that we use:
type (
	// LayerSheets contains Layer parameter Sheets.
	LayerSheets = params.Sheets[*LayerParams]

	// LayerSheet is one Layer parameter Sheet.
	LayerSheet = params.Sheet[*LayerParams]

	// LayerSel is one Layer parameter Selector.
	LayerSel = params.Sel[*LayerParams]

	// PathSheets contains Path parameter Sheets.
	PathSheets = params.Sheets[*PathParams]

	// PathSheet is one Path parameter Sheet.
	PathSheet = params.Sheet[*PathParams]

	// PathSel is one Path parameter Selector.
	PathSel = params.Sel[*PathParams]
)

// Params contains the [LayerParams] and [PathParams] parameter setting functions
// provided by the [emergent] [params] package.
type Params struct {

	// Layer has the parameters to apply to the [LayerParams] for layers.
	Layer LayerSheets `display:"-"`

	// Path has the parameters to apply to the [PathParams] for paths.
	Path PathSheets `display:"-"`

	// ExtraSheets has optional additional sheets of parameters to apply
	// after the default Base sheet. Use "Script" for default Script sheet.
	// Multiple names separated by spaces can be used (don't put spaces in Sheet names!)
	ExtraSheets string

	// Tag is an optional additional tag to add to log file names to identify
	// a specific run of the model (typically set by a config file or args).
	Tag string

	// Script is a parameter setting script, which adds to the Layer and Path sheets
	// typically using the "Script" set name.
	Script string `display:"-"`

	// Interp is the yaegi interpreter for running the script.
	Interp *interp.Interpreter `display:"-"`
}

// ScriptParams is a template for yaegi interpreted parameters
var ScriptParams = `sim.Sim.Params.Layer["Script"] = &axon.LayerSheet{
	&axon.LayerSel{Sel:"Layer", Set: func(ly *axon.LayerParams) {
		// set params
	}},
}
sim.Sim.Params.Path["Script"] = &axon.PathSheet{
	&axon.PathSel{Sel:"Path", Set: func(pt *axon.PathParams) {
		// set params
	}},
}
`

// Config configures the ExtraSheets, Tag, and Network fields, and
// initializes the yaegi interpreter for dynamic parameter scripts.
// Pass a reflect.ValueOf(*Sim) to initialize the yaegi interpreter.
// Sim must have Params in a field called Params.
func (pr *Params) Config(layer LayerSheets, path PathSheets, extraSheets, tag string, sim reflect.Value) {
	pr.Layer = layer
	pr.Path = path
	report := ""
	if extraSheets != "" {
		pr.ExtraSheets = extraSheets
		report += " ExtraSheets: " + extraSheets
	}
	if tag != "" {
		pr.Tag = tag
		report += " Tag: " + tag
	}
	if report != "" {
		mpi.Printf("Params Set: %s\n", report)
	}
	pr.Interp = interp.New(interp.Options{})
	pr.Interp.Use(interp.Exports{
		"github.com/emer/axon/axon": map[string]reflect.Value{
			"LayerParams": reflect.ValueOf((*LayerParams)(nil)),
			"PathParams":  reflect.ValueOf((*PathParams)(nil)),
			"LayerSel":    reflect.ValueOf((*LayerSel)(nil)),
			"LayerSheet":  reflect.ValueOf((*LayerSheet)(nil)),
			"LayerSheets": reflect.ValueOf((*LayerSheets)(nil)),
			"PathSel":     reflect.ValueOf((*PathSel)(nil)),
			"PathSheet":   reflect.ValueOf((*PathSheet)(nil)),
			"PathSheets":  reflect.ValueOf((*PathSheets)(nil)),
		},
		"github.com/emer/axon/sim/sim": map[string]reflect.Value{
			"Sim": sim,
		},
	})
	pr.Interp.ImportUsed()
}

// Name returns name of current set of parameters, including Tag.
// if ExtraSheets is empty then it returns "Base", otherwise returns ExtraSheets
func (pr *Params) Name() string {
	rn := ""
	if pr.Tag != "" {
		rn += pr.Tag + "_"
	}
	if pr.ExtraSheets == "" {
		rn += "Base"
	} else {
		rn += pr.ExtraSheets
	}
	return rn
}

// RunName returns the name of a simulation run based on params Name()
// and starting run number.
func (pr *Params) RunName(startRun int) string {
	return fmt.Sprintf("%s_%03d", pr.Name(), startRun)
}

// ApplyAll applies all parameters to given network,
// using "Base" Sheet then any ExtraSheets,
// for Layer and Path params (each must have the named sheets,
// for proper error checking in case of typos).
func (pr *Params) ApplyAll(net *Network) {
	pr.ApplySheet(net, "Base")
	if pr.ExtraSheets == "" {
		return
	}
	if pr.Script != "" {
		_, err := pr.Interp.Eval(pr.Script)
		if err != nil {
			fmt.Println(pr.Script)
			errors.Log(err)
		}
	}
	sps := strings.Fields(pr.ExtraSheets)
	for _, ps := range sps {
		if ps == "Base" {
			continue
		}
		pr.ApplySheet(net, ps)
	}
}

// ApplySheet applies parameters for given [params.Sheet] name
// for Layer and Path params (each must have the named sheets,
// for proper error checking in case of typos).
func (pr *Params) ApplySheet(net *Network, sheetName string) error {
	lsheet, err := pr.Layer.SheetByName(sheetName)
	if err != nil {
		return err
	}
	psheet, err := pr.Path.SheetByName(sheetName)
	if err != nil {
		return err
	}
	lsheet.SelMatchReset()
	psheet.SelMatchReset()

	ApplyParamSheets(net, lsheet, psheet)
	return nil
}

// ApplyParamSheets applies Layer and Path parameters from given sheets,
// returning true if any applied.
func ApplyParamSheets(net *Network, layer *params.Sheet[*LayerParams], path *params.Sheet[*PathParams]) bool {
	appl := ApplyLayerSheet(net, layer)
	appp := ApplyPathSheet(net, path)
	return appl || appp
}

// ApplyLayerSheet applies Layer parameters from given sheet, returning true if any applied.
func ApplyLayerSheet(net *Network, sheet *params.Sheet[*LayerParams]) bool {
	applied := false
	for _, ly := range net.Layers {
		app := sheet.Apply(&ly.Params)
		ly.UpdateParams()
		if app {
			applied = true
		}
	}
	return applied
}

// ApplyPathSheet applies Path parameters from given sheet, returning true if any applied.
func ApplyPathSheet(net *Network, sheet *params.Sheet[*PathParams]) bool {
	applied := false
	for _, ly := range net.Layers {
		for _, pt := range ly.RecvPaths {
			app := sheet.Apply(&pt.Params)
			pt.UpdateParams()
			if app {
				applied = true
			}
		}
	}
	return applied
}
