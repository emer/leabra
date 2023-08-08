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

	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
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

	// [def: 0.1] multiplier on GABAb as function of voltage
	GABAbv float64 `def:"0.1" desc:"multiplier on GABAb as function of voltage"`

	// [def: 10] offset of GABAb function
	GABAbo float64 `def:"10" desc:"offset of GABAb function"`

	// [def: -90] GABAb reversal / driving potential
	GABAberev float64 `def:"-90" desc:"GABAb reversal / driving potential"`

	// [def: -90] starting voltage
	Vstart float64 `def:"-90" desc:"starting voltage"`

	// [def: 0] ending voltage
	Vend float64 `def:"0" desc:"ending voltage"`

	// [def: 1] voltage increment
	Vstep float64 `def:"1" desc:"voltage increment"`

	// [def: 15] max number of spikes
	Smax int `def:"15" desc:"max number of spikes"`

	// rise time constant
	RiseTau float64 `desc:"rise time constant"`

	// decay time constant -- must NOT be same as RiseTau
	DecayTau float64 `desc:"decay time constant -- must NOT be same as RiseTau"`

	// initial value of GsX driving variable at point of synaptic input onset -- decays expoentially from this start
	GsXInit float64 `desc:"initial value of GsX driving variable at point of synaptic input onset -- decays expoentially from this start"`

	// time when peak conductance occurs, in TimeInc units
	MaxTime float64 `inactive:"+" desc:"time when peak conductance occurs, in TimeInc units"`

	// time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))
	TauFact float64 `inactive:"+" desc:"time constant factor used in integration: (Decay / Rise) ^ (Rise / (Decay - Rise))"`

	// total number of time steps to take
	TimeSteps int `desc:"total number of time steps to take"`

	// time increment per step
	TimeInc float64 `desc:"time increment per step"`

	// [view: no-inline] table for plot
	VGTable *etable.Table `view:"no-inline" desc:"table for plot"`

	// [view: no-inline] table for plot
	SGTable *etable.Table `view:"no-inline" desc:"table for plot"`

	// [view: no-inline] table for plot
	TimeTable *etable.Table `view:"no-inline" desc:"table for plot"`

	// [view: -] the plot
	VGPlot *eplot.Plot2D `view:"-" desc:"the plot"`

	// [view: -] the plot
	SGPlot *eplot.Plot2D `view:"-" desc:"the plot"`

	// [view: -] the plot
	TimePlot *eplot.Plot2D `view:"-" desc:"the plot"`

	// [view: -] main GUI window
	Win *gi.Window `view:"-" desc:"main GUI window"`

	// [view: -] the master toolbar
	ToolBar *gi.ToolBar `view:"-" desc:"the master toolbar"`
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

	ss.VGTable = &etable.Table{}
	ss.ConfigVGTable(ss.VGTable)

	ss.SGTable = &etable.Table{}
	ss.ConfigSGTable(ss.SGTable)

	ss.TimeTable = &etable.Table{}
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

func (ss *Sim) ConfigVGTable(dt *etable.Table) {
	dt.SetMetaData("name", "EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"V", etensor.FLOAT64, nil, nil},
		{"g_GABAb", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigVGPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "V-G Function Plot"
	plt.Params.XAxisCol = "V"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("V", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("g_GABAb", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
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

func (ss *Sim) ConfigSGTable(dt *etable.Table) {
	dt.SetMetaData("name", "SG_EqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"S", etensor.FLOAT64, nil, nil},
		{"gmax_GABAb", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigSGPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "S-G Function Plot"
	plt.Params.XAxisCol = "S"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("S", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("gmax_GABAb", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
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

func (ss *Sim) ConfigTimeTable(dt *etable.Table) {
	dt.SetMetaData("name", "TimeEqPlotTable")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Time", etensor.FLOAT64, nil, nil},
		{"Gs", etensor.FLOAT64, nil, nil},
		{"GsX", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTimePlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "G Time Function Plot"
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

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "VGPlot").(*eplot.Plot2D)
	ss.VGPlot = ss.ConfigVGPlot(plt, ss.VGTable)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "SGPlot").(*eplot.Plot2D)
	ss.SGPlot = ss.ConfigSGPlot(plt, ss.SGTable)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TimePlot").(*eplot.Plot2D)
	ss.TimePlot = ss.ConfigTimePlot(plt, ss.TimeTable)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Run VG", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.VGRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run SG", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.SGRun()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run Time", Icon: "update", Tooltip: "Run the equations and plot results."}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.TimeRun()
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
