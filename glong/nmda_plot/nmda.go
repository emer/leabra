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

	"cogentcore.org/core/gimain"
	"cogentcore.org/core/math32"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
)

func main() {
	TheSim.Config()
	gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
		guirun()
	})
}

func guirun() {
	win := TheSim.ConfigGUI()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// Sim holds the params, table, etc
type Sim struct {

	// multiplier on NMDA as function of voltage
	NMDAv float64 `def:"0.062"`

	// denominator of NMDA function
	NMDAd float64 `def:"3.57"`

	// NMDA reversal / driving potential
	NMDAerev float64 `def:"0"`

	// starting voltage
	Vstart float64 `def:"-90"`

	// ending voltage
	Vend float64 `def:"0"`

	// voltage increment
	Vstep float64 `def:"1"`

	// decay time constant for NMDA current -- rise time is 2 msec and not worth extra effort for biexponential
	Tau float64 `def:"100"`

	// number of time steps
	TimeSteps int

	// NMDA g current input at every time step
	Gin float64

	// table for plot
	Table *etable.Table `view:"no-inline"`

	// the plot
	Plot *eplot.Plot2D `view:"-"`

	// table for plot
	TimeTable *etable.Table `view:"no-inline"`

	// the plot
	TimePlot *eplot.Plot2D `view:"-"`

	// main GUI window
	Win *core.Window `view:"-"`

	// the master toolbar
	ToolBar *core.ToolBar `view:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.NMDAv = 0.062
	ss.NMDAd = 3.57
	ss.NMDAerev = 0
	ss.Vstart = -90
	ss.Vend = 0
	ss.Vstep = 1
	ss.Tau = 100
	ss.TimeSteps = 1000
	ss.Gin = .5
	ss.Update()
	ss.Table = &etable.Table{}
	ss.ConfigTable(ss.Table)
	ss.TimeTable = &etable.Table{}
	ss.ConfigTimeTable(ss.TimeTable)
}

// Update updates computed values
func (ss *Sim) Update() {
}

// Equation here:
// https://brian2.readthedocs.io/en/stable/examples/frompapers.Brunel_Wang_2001.html

// Run runs the equation.
func (ss *Sim) Run() {
	ss.Update()
	dt := ss.Table

	nv := int((ss.Vend - ss.Vstart) / ss.Vstep)
	dt.SetNumRows(nv)
	v := 0.0
	g := 0.0
	for vi := 0; vi < nv; vi++ {
		v = ss.Vstart + float64(vi)*ss.Vstep
		g = (ss.NMDAerev - v) / (1 + 1*math.Exp(-ss.NMDAv*v)/ss.NMDAd)

		dt.SetCellFloat("V", vi, v)
		dt.SetCellFloat("g_NMDA", vi, g)
	}
	ss.Plot.Update()
}

func (ss *Sim) ConfigTable(dt *etable.Table) {
	dt.SetMetaData("name", "EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"V", etensor.FLOAT64, nil, nil},
		{"g_NMDA", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "NMDA V-G Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("g_NMDA", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

/////////////////////////////////////////////////////////////////

// TimeRun runs the equation over time.
func (ss *Sim) TimeRun() {
	ss.Update()
	dt := ss.TimeTable

	dt.SetNumRows(ss.TimeSteps)
	g := 0.0
	for ti := 0; ti < ss.TimeSteps; ti++ {
		t := float64(ti) * .001
		if ti < ss.TimeSteps/2 {
			g = g + ss.Gin - g/ss.Tau
		} else {
			g = g - g/ss.Tau
		}

		dt.SetCellFloat("Time", ti, t)
		dt.SetCellFloat("g_NMDA", ti, g)
	}
	ss.TimePlot.Update()
}

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"g_NMDA", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Time Function Plot"
	plt.Params.XAxisCol = "Time"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Time", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("g_NMDA", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
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

	sv := views.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := core.AddNewTabView(split, "tv")

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "Plot").(*eplot.Plot2D)
	ss.Plot = ss.ConfigPlot(plt, ss.Table)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TimePlot").(*eplot.Plot2D)
	ss.TimePlot = ss.ConfigTimePlot(plt, ss.TimeTable)

	split.SetSplits(.3, .7)

	tbar.AddAction(core.ActOpts{Label: "V-G Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		ss.Run()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Time Run", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send tree.Ki, sig int64, data interface{}) {
		ss.TimeRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send tree.Ki, sig int64, data interface{}) {
			core.OpenURL("https://github.com/emer/leabra/blob/master/examples/eqplot/README.md")
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
