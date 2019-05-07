// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ra25 runs a simple random-associator four-layer leabra network
// that uses the standard supervised learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units)
package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/eplot"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/timer"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/gi/svg"
	"github.com/goki/gi/units"
	"github.com/goki/ki/ki"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// todo:
//
// * make etable/eplot2d.Plot which encapsulates the SVGEditor and
//   shows its columns -- basically replicating behavior of C++ GraphView for eaiser
//   dynamic selection of what to plot, how to plot it, etc.
//   Trick is how to make it also customizable via code..
//
// * LogTrainTrial, LogTestTrial, LogTestCycle and associated plots
//
// * etable/eview.TableView (gridview -- is there a gonum version?) and TableEdit (spreadsheet-like editor)
//   and show these in another tab for the input patterns
//

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// DefaultParams are the initial default parameters for this simulation
var DefaultParams = emer.ParamStyle{
	{"Prjn", emer.Params{
		"Prjn.Learn.Norm.On":     1,
		"Prjn.Learn.Momentum.On": 1,
		"Prjn.Learn.WtBal.On":    0,
	}},
	// "Layer": {
	// 	"Layer.Inhib.Layer.Gi": 1.8, // this is the default
	// },
	{"#Output", emer.Params{
		"Layer.Inhib.Layer.Gi": 1.4, // this turns out to be critical for small output layer
	}},
	{".Back", emer.Params{
		"Prjn.WtScale.Rel": 0.2, // this is generally quite important
	}},
}

// these are the plot color names to use in order for successive lines -- feel free to choose your own!
var PlotColorNames = []string{"black", "red", "blue", "ForestGreen", "purple", "orange", "brown", "chartreuse", "navy", "cyan", "magenta", "tan", "salmon", "yellow4", "SkyBlue", "pink"}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net        *leabra.Network `view:"no-inline"`
	Pats       *etable.Table   `view:"no-inline"`
	EpcLog     *etable.Table   `view:"no-inline"`
	Params     emer.ParamStyle `view:"no-inline"`
	MaxEpcs    int             `desc:"maximum number of epochs to run"`
	Epoch      int
	Trial      int
	Time       leabra.Time
	ViewOn     bool              `desc:"whether to update the network view while running"`
	TrainUpdt  leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt   leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	Plot       bool              `desc:"update the epoch plot while running?"`
	PlotVals   []string          `desc:"values to plot in epoch plot"`
	Sequential bool              `desc:"set to true to present items in sequential order"`
	Test       bool              `desc:"set to true to not call learning methods"`

	// statistics
	EpcSSE     float32 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE  float32 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr  float32 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcPctCor  float32 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcCosDiff float32 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`

	// internal state - view:"-"
	SumSSE     float32          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE  float32          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff float32          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	CntErr     int              `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	Porder     []int            `view:"-" inactive:"+" desc:"permuted pattern order"`
	EpcPlotSvg *svg.Editor      `view:"-" desc:"the epoch plot svg editor"`
	NetView    *netview.NetView `view:"-" desc:"the network viewer"`
	StopNow    bool             `view:"-" desc:"flag to stop running"`
	RndSeed    int64            `view:"-" desc:"the current random seed"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &etable.Table{}
	ss.EpcLog = &etable.Table{}
	ss.Params = DefaultParams
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.Cycle
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigNet()
	ss.OpenPats()
	ss.ConfigEpcLog()
}

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
	}
	ss.Epoch = 0
	ss.Trial = 0
	ss.StopNow = false
	ss.Time.Reset()
	np := ss.Pats.NumRows()
	ss.Porder = rand.Perm(np)            // always start with new one so random order is identical
	ss.Net.StyleParams(ss.Params, false) // true) // set msg
	ss.Net.InitWts()
	ss.EpcLog.SetNumRows(0)
	ss.UpdateView()
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

func (ss *Sim) UpdateView() {
	if ss.NetView != nil {
		ss.NetView.Update()
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					ss.UpdateView()
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView()
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Quarter:
				ss.UpdateView()
			case leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView()
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
		ss.Net.WtFmDWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView()
	}
}

// ApplyInputs applies input patterns from given row of given Table.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(pats *etable.Table, row int) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	inPats := pats.ColByName(inLay.Nm).(*etensor.Float32)
	outPats := pats.ColByName(outLay.Nm).(*etensor.Float32)

	// SubSpace gets the 2D cell at given row in tensor column
	inp, _ := inPats.SubSpace(2, []int{row})
	outp, _ := outPats.SubSpace(2, []int{row})
	inLay.ApplyExt(inp)
	outLay.ApplyExt(outp)
}

// TrainTrial runs one trial of training (Trial is an environmentally-defined
// term -- see leabra.TimeScales for different standard terminology).
// In other models, this function might be more naturally expressed in other terms.
func (ss *Sim) TrainTrial() {
	row := ss.Trial
	if !ss.Sequential {
		row = ss.Porder[ss.Trial]
	}
	ss.ApplyInputs(ss.Pats, row)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate

	// To allow for interactive single-step running, all of the higher temporal
	// scales must be incorporated into the trial level run method.
	// This is a good general principle for even more complex environments:
	// there should be a single method call that gets the next "step" of the
	// environment, and all the higher levels of temporal structure should all
	// be properly updated through this one lowest-level method call.
	ss.Trial++
	np := ss.Pats.NumRows()
	if ss.Trial >= np {
		ss.LogEpoch()
		if ss.Plot {
			ss.PlotEpcLog()
		}
		ss.Trial = 0
		ss.Epoch++
		erand.PermuteInts(ss.Porder)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView()
		}
	}
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float32) {
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	cosdiff = outLay.CosDiff.Cos
	sse, avgsse = outLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		ss.SumSSE += sse
		ss.SumAvgSSE += avgsse
		ss.SumCosDiff += cosdiff
		if sse != 0 {
			ss.CntErr++
		}
	}
	return
}

// LogEpoch adds data from current epoch to the EpochLog table -- computes epoch
// averages prior to logging.
// Epoch counter is assumed to not have yet been incremented.
func (ss *Sim) LogEpoch() {
	ss.EpcLog.SetNumRows(ss.Epoch + 1)
	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	hid2Lay := ss.Net.LayerByName("Hidden2").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	np := float32(ss.Pats.NumRows())
	ss.EpcSSE = ss.SumSSE / np
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / np
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float32(ss.CntErr) / np
	ss.CntErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / np
	ss.SumCosDiff = 0

	epc := ss.Epoch

	ss.EpcLog.ColByName("Epoch").SetFloat1D(epc, float64(epc))
	ss.EpcLog.ColByName("SSE").SetFloat1D(epc, float64(ss.EpcSSE))
	ss.EpcLog.ColByName("Avg SSE").SetFloat1D(epc, float64(ss.EpcAvgSSE))
	ss.EpcLog.ColByName("Pct Err").SetFloat1D(epc, float64(ss.EpcPctErr))
	ss.EpcLog.ColByName("Pct Cor").SetFloat1D(epc, float64(ss.EpcPctCor))
	ss.EpcLog.ColByName("CosDiff").SetFloat1D(epc, float64(ss.EpcCosDiff))
	ss.EpcLog.ColByName("Hid1 ActAvg").SetFloat1D(epc, float64(hid1Lay.Pools[0].ActAvg.ActPAvgEff))
	ss.EpcLog.ColByName("Hid2 ActAvg").SetFloat1D(epc, float64(hid2Lay.Pools[0].ActAvg.ActPAvgEff))
	ss.EpcLog.ColByName("Out ActAvg").SetFloat1D(epc, float64(outLay.Pools[0].ActAvg.ActPAvgEff))

}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	curEpc := ss.Epoch
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.Epoch > curEpc {
			break
		}
	}
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	stEpc := ss.Epoch
	tmr := timer.Time{}
	tmr.Start()
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.Epoch >= ss.MaxEpcs {
			break
		}
	}
	tmr.Stop()
	epcs := ss.Epoch - stEpc
	fmt.Printf("Took %6g secs for %v epochs, avg per epc: %6g\n", tmr.TotalSecs(), epcs, tmr.TotalSecs()/float64(epcs))
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	row := ss.Trial
	ss.ApplyInputs(ss.Pats, row)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.Trial++
	// todo: trial log, cycle log
	np := ss.Pats.NumRows()
	if ss.Trial >= np {
		ss.Trial = 0
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView()
		}
		// todo: test epoch log
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	np := ss.Pats.NumRows()
	ss.Trial = 0
	for trl := 0; trl < np; trl++ {
		ss.TestTrial()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Config methods

func (ss *Sim) ConfigNet() {
	net := ss.Net
	net.InitName(net, "RA25")
	inLay := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1Lay := net.AddLayer2D("Hidden1", 7, 7, emer.Hidden)
	hid2Lay := net.AddLayer4D("Hidden2", 2, 4, 3, 2, emer.Hidden)
	outLay := net.AddLayer2D("Output", 5, 5, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	hid2Lay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	net.ConnectLayers(inLay, hid1Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid1Lay, hid2Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid2Lay, outLay, prjn.NewFull(), emer.Forward)

	net.ConnectLayers(outLay, hid2Lay, prjn.NewFull(), emer.Back)
	net.ConnectLayers(hid2Lay, hid1Lay, prjn.NewFull(), emer.Back)

	// if Thread {
	// 	hid2Lay.SetThread(1)
	// 	outLay.SetThread(1)
	// }

	net.Defaults()
	net.StyleParams(ss.Params, true) // set msg
	net.Build()
	net.InitWts()
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}, 25)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("random_5x5_25_gen.dat", ',', true)
}

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	err := dt.OpenCSV("random_5x5_25.dat", '\t')
	if err != nil {
		log.Println(err)
	}
}

func (ss *Sim) ConfigEpcLog() {
	dt := ss.EpcLog
	dt.SetFromSchema(etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT32, nil, nil},
		{"Avg SSE", etensor.FLOAT32, nil, nil},
		{"Pct Err", etensor.FLOAT32, nil, nil},
		{"Pct Cor", etensor.FLOAT32, nil, nil},
		{"CosDiff", etensor.FLOAT32, nil, nil},
		{"Hid1 ActAvg", etensor.FLOAT32, nil, nil},
		{"Hid2 ActAvg", etensor.FLOAT32, nil, nil},
		{"Out ActAvg", etensor.FLOAT32, nil, nil},
	}, 0)
	ss.PlotVals = []string{"SSE", "Pct Err"}
	ss.Plot = true
}

// PlotEpcLog plots given epoch log using PlotVals Y axis columns into EpcPlotSvg
func (ss *Sim) PlotEpcLog() *plot.Plot {
	if !ss.EpcPlotSvg.IsVisible() {
		return nil
	}
	dt := ss.EpcLog
	plt, _ := plot.New() // todo: keep around?
	plt.Title.Text = "Random Associator Epoch Log"
	plt.X.Label.Text = "Epoch"
	plt.Y.Label.Text = "Y"

	const lineWidth = 1

	for i, cl := range ss.PlotVals {
		xy, _ := eplot.NewTableXYNames(dt, "Epoch", cl)
		l, _ := plotter.NewLine(xy)
		l.LineStyle.Width = vg.Points(lineWidth)
		clr, _ := gi.ColorFromString(PlotColorNames[i%len(PlotColorNames)], nil)
		l.LineStyle.Color = clr
		plt.Add(l)
		plt.Legend.Add(cl, l)
	}
	plt.Legend.Top = true
	eplot.PlotViewSVG(plt, ss.EpcPlotSvg, 5, 5, 2)
	return plt
}

// SaveEpcPlot plots given epoch log using PlotVals Y axis columns and saves to .svg file
func (ss *Sim) SaveEpcPlot(fname string) {
	plt := ss.PlotEpcLog()
	plt.Save(5, 5, fname)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("ra25")
	gi.SetAppAbout(`This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	plot.DefaultFont = "Helvetica"

	win := gi.NewWindow2D("ra25", "Leabra Random Associator", width, height, true)

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	// split.SetProp("horizontal-align", "center")
	// split.SetProp("margin", 2.0) // raw numbers = px = 96 dpi pixels
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss, nil)
	// sv.SetStretchMaxWidth()
	// sv.SetStretchMaxHeight()

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.SetStretchMaxWidth()
	nv.SetStretchMaxHeight()
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv

	svge := tv.AddNewTab(svg.KiT_Editor, "Epc Plot").(*svg.Editor)
	svge.InitScale()
	svge.Fill = true
	svge.SetProp("background-color", "white")
	svge.SetProp("width", units.NewValue(float32(width/2), units.Px))
	svge.SetProp("height", units.NewValue(float32(height-100), units.Px))
	svge.SetStretchMaxWidth()
	svge.SetStretchMaxHeight()
	ss.EpcPlotSvg = svge

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Init()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			go ss.Train()
		})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Stop()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TrainTrial()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TrainEpoch()
			vp.FullRender2DTree()
		})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TestTrial()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TestAll()
			vp.FullRender2DTree()
		})

	tbar.AddSeparator("file")

	tbar.AddAction(gi.ActOpts{Label: "Epoch Plot", Icon: "update"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.PlotEpcLog()
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Wts", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Net.SaveWtsJSON("ra25_net_trained.wts") // todo: call method to prompt
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Log", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.EpcLog.SaveCSV("ra25_epc.dat", ',', true)
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Plot", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.SaveEpcPlot("ra25_cur_epc_plot.svg")
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Params", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			// todo: need save / load methods for these
			// ss.EpcLog.SaveCSV("ra25_epc.dat", ',', true)
		})

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
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

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

func mainrun() {
	// gi3d.Update3DTrace = true
	// gi.Update2DTrace = true
	// gi.Render2DTrace = true
	//
	// todo: args
	TheSim.New()
	TheSim.Config()
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
