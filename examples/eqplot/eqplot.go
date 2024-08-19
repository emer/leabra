// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// eqplot plots an equation updating over time in a table.Table and Plot2D.
// This is a good starting point for any plotting to explore specific equations.
// This example plots a double exponential (biexponential) model of synaptic currents.
package main

import (
	"math"
	"strconv"

	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/plot"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		sim.RunGUI()
	} else {
		sim.RunNoGUI()
	}
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// rise time constant
	RiseTau float64

	// decay time constant -- must NOT be same as RiseTau
	DecayTau float64

	// initial value of GsX driving variable at point of synaptic input onset -- decays expoentially from this start
	GsXInit float64

	// time when peak conductance occurs, in TimeInc units
	MaxTime float64 `edit:"-"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float64 `edit:"-"`

	// total number of time steps to take
	TimeSteps int

	// time increment per step
	TimeInc float64

	// table for plot
	Table *table.Table `display:"no-inline"`

	// the plot
	Plot *plot.Plot2D `display:"-"`

	// main GUI window
	Win *core.Window `display:"-"`

	// the master toolbar
	ToolBar *core.ToolBar `display:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.RiseTau = 10
	ss.DecayTau = 100
	ss.GsXInit = 1
	ss.TimeSteps = 1000
	ss.TimeInc = .001
	ss.Update()
	ss.Table = &table.Table{}
	ss.ConfigTable(ss.Table)
}

// Equation for biexponential synapse from here:
// https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html

// Update updates computed values
func (ss *Sim) Update() {
	ss.TauFact = math.Pow(ss.DecayTau/ss.RiseTau, ss.RiseTau/(ss.DecayTau-ss.RiseTau))
	ss.MaxTime = ((ss.RiseTau * ss.DecayTau) / (ss.DecayTau - ss.RiseTau)) * math.Log(ss.DecayTau/ss.RiseTau)
}

// Run runs the equation.
func (ss *Sim) Run() {
	ss.Update()
	dt := ss.Table
	dt.SetNumRows(ss.TimeSteps)
	time := 0.0
	gs := 0.0
	x := ss.GsXInit
	for t := 0; t < ss.TimeSteps; t++ {
		// record starting state first, then update
		dt.SetCellFloat("Time", t, time)
		dt.SetCellFloat("Gs", t, gs)
		dt.SetCellFloat("GsX", t, x)

		dGs := (ss.TauFact*x - gs) / ss.RiseTau
		dX := -x / ss.DecayTau
		gs += dGs
		x += dX
		time += ss.TimeInc
	}
	ss.Plot.Update()
}

func (ss *Sim) ConfigTable(dt *table.Table) {
	dt.SetMetaData("name", "EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"Gs", tensor.FLOAT64, nil, nil},
		{"GsX", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Gs", plot.On, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("GsX", plot.On, plot.FixMin, 0, plot.FloatMax, 0)
	return plt
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Window {
	width := 1600
	height := 1200

	// core.WinEventTrace = true

	core.SetAppName("eqplot")
	core.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := core.NewMainWindow("eqplot", "Plotting Equations", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := core.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := core.AddNewSplitView(mfr, "split")
	split.Dim = math32.X
	split.SetStretchMax()

	sv := core.NewForm(split, "sv")
	sv.SetStruct(ss)

	tv := core.AddNewTabView(split, "tv")

	plt := tv.AddNewTab(plot.KiT_Plot2D, "Plot").(*plot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Table)

	split.SetSplits(.3, .7)

	tbar.AddAction(core.ActOpts{Label: "Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Run()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "README", Icon: icons.FileMarkdown, Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			core.OpenURL("https://github.com/emer/leabra/blob/main/examples/eqplot/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := core.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*core.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*core.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
}
