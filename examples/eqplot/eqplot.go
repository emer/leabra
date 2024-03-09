// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// eqplot plots an equation updating over time in a etable.Table and Plot2D.
// This is a good starting point for any plotting to explore specific equations.
// This example plots a double exponential (biexponential) model of synaptic currents.
package main

import (
	"math"
	"strconv"

	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	_ "github.com/emer/etable/v2/etview" // include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/mat32"
)

func main() {
	TheSim.Config()
	gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
		guirun()
	})
}

func guirun() {
	win := TheSim.ConfigGui()
	win.StartEventLoop()
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
	MaxTime float64 `inactive:"+"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float64 `inactive:"+"`

	// total number of time steps to take
	TimeSteps int

	// time increment per step
	TimeInc float64

	// table for plot
	Table *etable.Table `view:"no-inline"`

	// the plot
	Plot *eplot.Plot2D `view:"-"`

	// main GUI window
	Win *gi.Window `view:"-"`

	// the master toolbar
	ToolBar *gi.ToolBar `view:"-"`
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
	ss.Table = &etable.Table{}
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

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Gs", etensor.FLOAT64, nil, nil},
		{"GsX", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Gs", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("GsX", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	// gi.WinEventTrace = true

	gi.SetAppName("eqplot")
	gi.SetAppAbout(`This plots an equation. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("eqplot", "Plotting Equations", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "Plot").(*eplot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Table)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Run()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/eqplot/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	win.MainMenuUpdated()
	return win
}
