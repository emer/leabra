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
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/timer"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"gonum.org/v1/plot"
)

// todo:
//
// * colorscale in giv, including gui
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

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net        *leabra.Network   `view:"no-inline"`
	Pats       *etable.Table     `view:"no-inline" desc:"the training patterns"`
	EpcLog     *etable.Table     `view:"no-inline" desc:"epoch-level log data"`
	TstTrlLog  *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstCycLog  *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	Params     emer.ParamStyle   `view:"no-inline"`
	MaxEpcs    int               `desc:"maximum number of epochs to run"`
	Epoch      int               `desc:"current epoch"`
	Trial      int               `desc:"current trial"`
	TrialName  string            `inactive:"+" desc:"current trial name"`
	Time       leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn     bool              `desc:"whether to update the network view while running"`
	TrainUpdt  leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt   leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	Sequential bool              `desc:"set to true to present items in sequential order"`
	Test       bool              `desc:"set to true to not call learning methods"`

	// statistics
	TrlSSE     float32 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE  float32 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff float32 `inactive:"+" desc:"current trial's cosine difference"`
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
	NetView    *netview.NetView `view:"-" desc:"the network viewer"`
	EpcPlot    *eplot.Plot2D    `view:"-" desc:"the epoch plot"`
	TstTrlPlot *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	StopNow    bool             `view:"-" desc:"flag to stop running"`
	RndSeed    int64            `view:"-" desc:"the current random seed"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &etable.Table{}
	ss.EpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
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
	ss.ConfigTstTrlLog()
	ss.ConfigTstCycLog()
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

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Epoch:\t%d\tTrial:\t%d\tName:\t%v\t\t\t", ss.Epoch, ss.Trial, ss.TrialName)
}

func (ss *Sim) UpdateView() {
	if ss.NetView != nil {
		ss.NetView.Update(ss.Counters()) // this is a lot slower but anyway we need the counters
		// ss.NetView.Update("")
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
			if !train {
				ss.LogTstCyc(ss.Time.Cycle)
			}
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
	names := pats.ColByName("Name").(*etensor.String)
	ss.TrialName = names.Values[row]

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
		ss.LogEpc()
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
	ss.TrlCosDiff = outLay.CosDiff.Cos
	ss.TrlSSE, ss.TrlAvgSSE = outLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if accum {
		ss.SumSSE += ss.TrlSSE
		ss.SumAvgSSE += ss.TrlAvgSSE
		ss.SumCosDiff += ss.TrlCosDiff
		if ss.TrlSSE != 0 {
			ss.CntErr++
		}
	}
	return
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

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// SaveParams saves the current params -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveParams(filename gi.FileName) {
	// ss.Net.SaveWeights(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	row := ss.Trial
	ss.ApplyInputs(ss.Pats, row)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl()

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
	ss.StopNow = false
	np := ss.Pats.NumRows()
	ss.Trial = 0
	for trl := 0; trl < np; trl++ {
		ss.TestTrial()
		if ss.StopNow {
			break
		}
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

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// LogEpc adds data from current epoch to the EpcLog table.
// computes epoch averages prior to logging.
// Epoch counter is assumed to not have yet been incremented.
func (ss *Sim) LogEpc() {
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

	ss.EpcPlot.Update()
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
}

func (ss *Sim) ConfigEpcPlot() {
	ss.EpcPlot.Params.Title = "Leabra Random Associator 25 Epoch Plot"
	ss.EpcPlot.Params.XAxisCol = "Epoch"
	// order of params: on, fixMin, min, fixMax, max
	ss.EpcPlot.SetColParams("Epoch", false, true, 0, false, 0)
	ss.EpcPlot.SetColParams("SSE", false, true, 0, false, 0)
	ss.EpcPlot.SetColParams("Avg SSE", false, true, 0, false, 0)
	ss.EpcPlot.SetColParams("Pct Err", true, true, 0, true, 1) // default plot
	ss.EpcPlot.SetColParams("Pct Cor", true, true, 0, true, 1) // default plot
	ss.EpcPlot.SetColParams("CosDiff", false, true, 0, true, 1)
	ss.EpcPlot.SetColParams("Hid1 ActAvg", false, true, 0, true, .5)
	ss.EpcPlot.SetColParams("Hid2 ActAvg", false, true, 0, true, .5)
	ss.EpcPlot.SetColParams("Out ActAvg", false, true, 0, true, .5)
}

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log only contains Pats.NumRows() entries
func (ss *Sim) LogTstTrl() {
	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	hid2Lay := ss.Net.LayerByName("Hidden2").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	trl := ss.Trial

	ss.TstTrlLog.ColByName("Trial").SetFloat1D(trl, float64(trl))
	ss.TstTrlLog.ColByName("TrialName").SetString1D(trl, ss.TrialName)
	ss.TstTrlLog.ColByName("SSE").SetFloat1D(trl, float64(ss.TrlSSE))
	ss.TstTrlLog.ColByName("Avg SSE").SetFloat1D(trl, float64(ss.TrlAvgSSE))
	ss.TstTrlLog.ColByName("CosDiff").SetFloat1D(trl, float64(ss.TrlCosDiff))
	ss.TstTrlLog.ColByName("Hid1 ActM.Avg").SetFloat1D(trl, float64(hid1Lay.Pools[0].ActM.Avg))
	ss.TstTrlLog.ColByName("Hid2 ActM.Avg").SetFloat1D(trl, float64(hid2Lay.Pools[0].ActM.Avg))
	ss.TstTrlLog.ColByName("Out ActM.Avg").SetFloat1D(trl, float64(outLay.Pools[0].ActM.Avg))

	ss.TstTrlPlot.Update()
}

func (ss *Sim) ConfigTstTrlLog() {
	dt := ss.TstTrlLog
	np := ss.Pats.NumRows()
	dt.SetFromSchema(etable.Schema{
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT32, nil, nil},
		{"Avg SSE", etensor.FLOAT32, nil, nil},
		{"CosDiff", etensor.FLOAT32, nil, nil},
		{"Hid1 ActM.Avg", etensor.FLOAT32, nil, nil},
		{"Hid2 ActM.Avg", etensor.FLOAT32, nil, nil},
		{"Out ActM.Avg", etensor.FLOAT32, nil, nil},
	}, np)
}

func (ss *Sim) ConfigTstTrlPlot() {
	ss.TstTrlPlot.Params.Title = "Leabra Random Associator 25 Test Trial Plot"
	ss.TstTrlPlot.Params.XAxisCol = "Trial"
	// order of params: on, fixMin, min, fixMax, max
	ss.TstTrlPlot.SetColParams("Trial", false, true, 0, false, 0)
	ss.TstTrlPlot.SetColParams("Trial Name", false, true, 0, false, 0)
	ss.TstTrlPlot.SetColParams("SSE", false, true, 0, false, 0)
	ss.TstTrlPlot.SetColParams("Avg SSE", true, true, 0, false, 0)
	ss.TstTrlPlot.SetColParams("CosDiff", true, true, 0, true, 1)
	ss.TstTrlPlot.SetColParams("Hid1 ActM.Avg", true, true, 0, true, .5)
	ss.TstTrlPlot.SetColParams("Hid2 ActM.Avg", true, true, 0, true, .5)
	ss.TstTrlPlot.SetColParams("Out ActM.Avg", true, true, 0, true, .5)
}

// LogTstCyc adds data from current trial to the TstCycLog table.
// log only contains Pats.NumRows() entries
func (ss *Sim) LogTstCyc(cyc int) {
	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	hid2Lay := ss.Net.LayerByName("Hidden2").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	ss.TstCycLog.ColByName("Cycle").SetFloat1D(cyc, float64(cyc))
	ss.TstCycLog.ColByName("Hid1 Ge.Avg").SetFloat1D(cyc, float64(hid1Lay.Pools[0].Ge.Avg))
	ss.TstCycLog.ColByName("Hid2 Ge.Avg").SetFloat1D(cyc, float64(hid2Lay.Pools[0].Ge.Avg))
	ss.TstCycLog.ColByName("Out Ge.Avg").SetFloat1D(cyc, float64(outLay.Pools[0].Ge.Avg))
	ss.TstCycLog.ColByName("Hid1 Act.Avg").SetFloat1D(cyc, float64(hid1Lay.Pools[0].Act.Avg))
	ss.TstCycLog.ColByName("Hid2 Act.Avg").SetFloat1D(cyc, float64(hid2Lay.Pools[0].Act.Avg))
	ss.TstCycLog.ColByName("Out Act.Avg").SetFloat1D(cyc, float64(outLay.Pools[0].Act.Avg))

	ss.TstCycPlot.Update()
}

func (ss *Sim) ConfigTstCycLog() {
	dt := ss.TstCycLog
	np := 100 // max cycles
	dt.SetFromSchema(etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"Hid1 Ge.Avg", etensor.FLOAT32, nil, nil},
		{"Hid2 Ge.Avg", etensor.FLOAT32, nil, nil},
		{"Out Ge.Avg", etensor.FLOAT32, nil, nil},
		{"Hid1 Act.Avg", etensor.FLOAT32, nil, nil},
		{"Hid2 Act.Avg", etensor.FLOAT32, nil, nil},
		{"Out Act.Avg", etensor.FLOAT32, nil, nil},
	}, np)
}

func (ss *Sim) ConfigTstCycPlot() {
	ss.TstCycPlot.Params.Title = "Leabra Random Associator 25 Test Cycle Plot"
	ss.TstCycPlot.Params.XAxisCol = "Cycle"
	// order of params: on, fixMin, min, fixMax, max
	ss.TstCycPlot.SetColParams("Cycle", false, true, 0, false, 0)
	ss.TstCycPlot.SetColParams("Hid1 Ge.Avg", true, true, 0, true, .5)
	ss.TstCycPlot.SetColParams("Hid2 Ge.Avg", true, true, 0, true, .5)
	ss.TstCycPlot.SetColParams("Out Ge.Avg", true, true, 0, true, .5)
	ss.TstCycPlot.SetColParams("Hid1 Act.Avg", true, true, 0, true, .5)
	ss.TstCycPlot.SetColParams("Hid2 Act.Avg", true, true, 0, true, .5)
	ss.TstCycPlot.SetColParams("Out Act.Avg", true, true, 0, true, .5)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

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
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss, nil)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "EpcPlot").(*eplot.Plot2D)
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(ss.EpcLog)
	ss.EpcPlot = plt
	ss.ConfigEpcPlot()

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(ss.TstTrlLog)
	ss.TstTrlPlot = plt
	ss.ConfigTstTrlPlot()

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(ss.TstCycLog)
	ss.TstCycPlot = plt
	ss.ConfigTstCycPlot()

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
			go ss.TrainEpoch()
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
			go ss.TestAll()
			vp.FullRender2DTree()
		})

	tbar.AddSeparator("file")

	tbar.AddAction(gi.ActOpts{Label: "Save Wts", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "SaveWeights", vp) // this auto prompts for filename using file chooser
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Params", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "SaveParams", vp) // this auto prompts for filename using file chooser
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

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts",
				}},
			},
		}},
		{"SaveParams", ki.Props{
			"desc": "save parameters to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".params",
				}},
			},
		}},
	},
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
