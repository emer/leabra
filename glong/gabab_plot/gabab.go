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
	sim.Config()
	sim.VmRun()
	b := sim.ConfigGUI()
	b.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// multiplier on GABAb as function of voltage
	GABAbv float64 `def:"0.1"`

	// offset of GABAb function
	GABAbo float64 `def:"10"`

	// GABAb reversal / driving potential
	GABAberev float64 `def:"-90"`

	// starting voltage
	Vstart float64 `def:"-90"`

	// ending voltage
	Vend float64 `def:"0"`

	// voltage increment
	Vstep float64 `def:"1"`

	// max number of spikes
	Smax int `def:"15"`

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
	VGTable *table.Table `display:"no-inline"`

	// table for plot
	SGTable *table.Table `display:"no-inline"`

	// table for plot
	TimeTable *table.Table `display:"no-inline"`

	// the plot
	VGPlot *plot.Plot2D `display:"-"`

	// the plot
	SGPlot *plot.Plot2D `display:"-"`

	// the plot
	TimePlot *plot.Plot2D `display:"-"`

	// main GUI window
	Win *core.Window `display:"-"`

	// the master toolbar
	ToolBar *core.ToolBar `display:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.GABAbv = 0.1
	ss.GABAbo = 10
	ss.GABAberev = -90
	ss.Vstart = -90
	ss.Vend = 0
	ss.Vstep = 1
	ss.Smax = 15
	ss.RiseTau = 45
	ss.DecayTau = 50
	ss.GsXInit = 1
	ss.TimeSteps = 200
	ss.TimeInc = .001
	ss.Update()

	ss.VGTable = &table.Table{}
	ss.ConfigVGTable(ss.VGTable)

	ss.SGTable = &table.Table{}
	ss.ConfigSGTable(ss.SGTable)

	ss.TimeTable = &table.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
	ss.TauFact = math.Pow(ss.DecayTau/ss.RiseTau, ss.RiseTau/(ss.DecayTau-ss.RiseTau))
	ss.MaxTime = ((ss.RiseTau * ss.DecayTau) / (ss.DecayTau - ss.RiseTau)) * math.Log(ss.DecayTau/ss.RiseTau)
}

// VGRun runs the V-G equation.
func (ss *Sim) VGRun() {
	ss.Update()
	dt := ss.VGTable

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	v := 0.0
	g := 0.0
	for vi := 0; vi < nv; vi++ {
		v = ss.Vstart + float64(vi)*ss.Vstep
		g = (v - ss.GABAberev) / (1 + math.Exp(ss.GABAbv*((v-ss.GABAberev)+ss.GABAbo)))

		dt.SetCellFloat("V", vi, v)
		dt.SetCellFloat("g_GABAb", vi, g)
	}
	ss.VGPlot.Update()
}

func (ss *Sim) ConfigVGTable(dt *table.Table) {
	dt.SetMetaData("name", "EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"V", tensor.FLOAT64, nil, nil},
		{"g_GABAb", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigVGPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "V-G Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", plot.Off, plot.FloatMin, 0, plot.FloatMax, 0)
	plt.SetColParams("g_GABAb", plot.On, plot.FixMin, 0, plot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////////

// SGRun runs the spike-g equation.
func (ss *Sim) SGRun() {
	ss.Update()
	dt := ss.SGTable

	dt.SetNumRows(ss.Smax)
	s := 0.0
	g := 0.0
	for si := 0; si < ss.Smax; si++ {
		s = float64(si)
		g = 1 / (1 + math.Exp(-(s-7.1)/1.4))

		dt.SetCellFloat("S", si, s)
		dt.SetCellFloat("gmax_GABAb", si, g)
	}
	ss.SGPlot.Update()
}

func (ss *Sim) ConfigSGTable(dt *table.Table) {
	dt.SetMetaData("name", "SG_EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"S", tensor.FLOAT64, nil, nil},
		{"gmax_GABAb", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigSGPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "S-G Function Plot"
	plt.Params.XAxisCol = "S"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("S", plot.Off, plot.FloatMin, 0, plot.FloatMax, 0)
	plt.SetColParams("gmax_GABAb", plot.On, plot.FixMin, 0, plot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////////

// TimeRun runs the equation.
func (ss *Sim) TimeRun() {
	ss.Update()
	dt := ss.TimeTable

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
	ss.TimePlot.Update()
}

func (ss *Sim) ConfigTimeTable(dt *table.Table) {
	dt.SetMetaData("name", "TimeEqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Time", tensor.FLOAT64, nil, nil},
		{"Gs", tensor.FLOAT64, nil, nil},
		{"GsX", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "G Time Function Plot"
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

	plt := tv.AddNewTab(plot.KiT_Plot2D, "VGPlot").(*plot.Plot2D)
	ss.VGPlot = ss.ConfigVGPlot(plt, ss.VGTable)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "SGPlot").(*plot.Plot2D)
	ss.SGPlot = ss.ConfigSGPlot(plt, ss.SGTable)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TimePlot").(*plot.Plot2D)
	ss.TimePlot = ss.ConfigTimePlot(plt, ss.TimeTable)

	split.SetSplits(.3, .7)

	tbar.AddAction(core.ActOpts{Label: "Run VG", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.VGRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Run SG", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.SGRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Run Time", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.TimeRun()
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
