// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ra25 runs a simple random-associator four-layer leabra network
// that uses the standard supervised learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units)
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/timer"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/gi/oswin"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"gonum.org/v1/plot"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "false",
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.4",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "50",
				}},
		},
	}},
	{Name: "DefaultInhib", Desc: "output uses default inhib instead of lower", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Output", Desc: "go back to default",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "takes longer -- generally doesn't finish..",
				Params: params.Params{
					"Sim.MaxEpcs": "100",
				}},
		},
	}},
	{Name: "NoMomentum", Desc: "no momentum or normalization", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no norm or momentum",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
				}},
		},
	}},
	{Name: "WtBalOn", Desc: "try with weight bal on", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "weight bal on",
				Params: params.Params{
					"Prjn.Learn.WtBal.On": "true",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *leabra.Network   `view:"no-inline"`
	Pats         *etable.Table     `view:"no-inline" desc:"the training patterns to use"`
	TrnEpcLog    *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog    *etable.Table     `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats  *etable.Table     `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog    *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog       *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Params       params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag          string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files)"`
	MaxRuns      int               `desc:"maximum number of model runs to perform"`
	MaxEpcs      int               `desc:"maximum number of epochs to run per model run"`
	TrainEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	Time         leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn       bool              `desc:"whether to update the network view while running"`
	TrainUpdt    leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int               `desc:"how often to run through all the test patterns, in terms of training epochs"`

	// statistics: note use float64 as that is best for etable.Table
	TrlSSE     float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE  float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcSSE     float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE  float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcPctCor  float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcCosDiff float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	FirstZero  int     `inactive:"+" desc:"epoch at when SSE first went to zero"`

	// internal state - view:"-"
	SumSSE       float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	CntErr       int              `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	NetView      *netview.NetView `view:"-" desc:"the network viewer"`
	TrnEpcPlot   *eplot.Plot2D    `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D    `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D    `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File         `view:"-" desc:"log file"`
	RunFile      *os.File         `view:"-" desc:"log file"`
	SaveWts      bool             `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool             `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool             `view:"-" desc:"if true, print message for all params that are set"`
	StopNow      bool             `view:"-" desc:"flag to stop running"`
	RndSeed      int64            `view:"-" desc:"the current random seed"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.Cycle
	ss.TestInterval = 5
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	//ss.ConfigPats()
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet()
	ss.ConfigTrnEpcLog()
	ss.ConfigTstEpcLog()
	ss.ConfigTstTrlLog()
	ss.ConfigTstCycLog()
	ss.ConfigRunLog()
}

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.NewRun()
	ss.UpdateView(true)
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.TrainEnv.TrialName)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.TestEnv.TrialName)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil {
		ss.NetView.Update(ss.Counters(train)) // this is a lot slower but anyway we need the counters
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
					ss.UpdateView(train)
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Quarter:
				ss.UpdateView(train)
			case leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
		ss.Net.WtFmDWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.Update() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	inPats := en.State(inLay.Nm)
	if inPats != nil {
		inLay.ApplyExt(inPats)
	}
	outPats := en.State(outLay.Nm)
	if inPats != nil {
		outLay.ApplyExt(outPats)
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc()
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
			ss.StopNow = false // keep going
		}
		if epc >= ss.MaxEpcs { // done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NewRun()
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun()
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumSSE = 0
	ss.SumAvgSSE = 0
	ss.SumCosDiff = 0
	ss.CntErr = 0
	ss.FirstZero = -1
	// clear rest just to make Sim look initialized
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float64) {
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)
	ss.TrlCosDiff = float64(outLay.CosDiff.Cos)
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
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	stEpc := ss.TrainEnv.Epoch.Cur
	tmr := timer.Time{}
	tmr.Start()
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	tmr.Stop()
	epcs := ss.TrainEnv.Epoch.Prv - stEpc
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

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial() {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		ss.StopNow = true
		ss.LogTstEpc()
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl()
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.StopNow = false
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial()
		if ss.StopNow {
			break
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Config methods

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

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
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
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
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.dat", '\t')
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc() {
	dt := ss.TrnEpcLog
	row := dt.Rows
	ss.TrnEpcLog.SetNumRows(row + 1)

	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	hid2Lay := ss.Net.LayerByName("Hidden2").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	epc := ss.TrainEnv.Epoch.Prv           // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Table.Len()) // number of trials in view

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.CntErr) / nt
	ss.CntErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	dt.SetCellFloat("Hid1 ActAvg", row, float64(hid1Lay.Pools[0].ActAvg.ActPAvgEff))
	dt.SetCellFloat("Hid2 ActAvg", row, float64(hid2Lay.Pools[0].ActAvg.ActPAvgEff))
	dt.SetCellFloat("Out ActAvg", row, float64(outLay.Pools[0].ActAvg.ActPAvgEff))

	ss.TrnEpcPlot.Update()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, '\t')
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, '\t', true)
	}
}

func (ss *Sim) ConfigTrnEpcLog() {
	dt := ss.TrnEpcLog
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Hid1 ActAvg", etensor.FLOAT64, nil, nil},
		{"Hid2 ActAvg", etensor.FLOAT64, nil, nil},
		{"Out ActAvg", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTrnEpcPlot() {
	plt := ss.TrnEpcPlot
	plt.Params.Title = "Leabra Random Associator 25 Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", true, true, 0, true, 1) // default plot
	plt.SetColParams("PctCor", true, true, 0, true, 1) // default plot
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
	plt.SetColParams("Hid1 ActAvg", false, true, 0, true, .5)
	plt.SetColParams("Hid2 ActAvg", false, true, 0, true, .5)
	plt.SetColParams("Out ActAvg", false, true, 0, true, .5)
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log only contains Pats.NumRows() entries
func (ss *Sim) LogTstTrl() {
	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	hid2Lay := ss.Net.LayerByName("Hidden2").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TestEnv.Trial.Cur
	dt := ss.TstTrlLog

	dt.SetCellFloat("Epoch", trl, float64(epc))
	dt.SetCellFloat("Trial", trl, float64(trl))
	dt.SetCellString("TrialName", trl, ss.TestEnv.TrialName)
	dt.SetCellFloat("SSE", trl, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", trl, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", trl, ss.TrlCosDiff)
	dt.SetCellFloat("Hid1 ActM.Avg", trl, float64(hid1Lay.Pools[0].ActM.Avg))
	dt.SetCellFloat("Hid2 ActM.Avg", trl, float64(hid2Lay.Pools[0].ActM.Avg))
	dt.SetCellFloat("Out ActM.Avg", trl, float64(outLay.Pools[0].ActM.Avg))

	dt.SetCellTensor("InAct", trl, inLay.UnitValsTensor("Act"))
	dt.SetCellTensor("OutActM", trl, outLay.UnitValsTensor("ActM"))
	dt.SetCellTensor("OutActP", trl, outLay.UnitValsTensor("ActP"))

	ss.TstTrlPlot.Update()
}

func (ss *Sim) ConfigTstTrlLog() {
	inLay := ss.Net.LayerByName("Input").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	dt := ss.TstTrlLog
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	nt := ss.TestEnv.Table.Len() // number in view
	dt.SetFromSchema(etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Hid1 ActM.Avg", etensor.FLOAT64, nil, nil},
		{"Hid2 ActM.Avg", etensor.FLOAT64, nil, nil},
		{"Out ActM.Avg", etensor.FLOAT64, nil, nil},
		{"InAct", etensor.FLOAT64, inLay.Shp.Shp, nil},
		{"OutActM", etensor.FLOAT64, outLay.Shp.Shp, nil},
		{"OutActP", etensor.FLOAT64, outLay.Shp.Shp, nil},
	}, nt)
}

func (ss *Sim) ConfigTstTrlPlot() {
	plt := ss.TstTrlPlot
	plt.Params.Title = "Leabra Random Associator 25 Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("Trial", false, true, 0, false, 0)
	plt.SetColParams("TrialName", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", true, true, 0, false, 0)
	plt.SetColParams("CosDiff", true, true, 0, true, 1)
	plt.SetColParams("Hid1 ActM.Avg", true, true, 0, true, .5)
	plt.SetColParams("Hid2 ActM.Avg", true, true, 0, true, .5)
	plt.SetColParams("Out ActM.Avg", true, true, 0, true, .5)

	plt.SetColParams("InAct", false, true, 0, true, 1)
	plt.SetColParams("OutActM", false, true, 0, true, 1)
	plt.SetColParams("OutActP", false, true, 0, true, 1)
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc() {
	dt := ss.TstEpcLog
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val > 0
	})[0])
	dt.SetCellFloat("PctCor", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val == 0
	})[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	// todo: how to grab acts from layer as tensor and record to table.

	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("SSE", row) > 0 // include error trials
	})
	ss.TstErrLog = trlix.NewTable()

	allsp := split.All(trlix)
	split.Agg(allsp, "SSE", agg.AggSum)
	split.Agg(allsp, "AvgSSE", agg.AggMean)
	split.Agg(allsp, "InAct", agg.AggMean)
	split.Agg(allsp, "OutActM", agg.AggMean)
	split.Agg(allsp, "OutActP", agg.AggMean)

	ss.TstErrStats = allsp.AggsToTable(false)

	ss.TstEpcPlot.Update()
}

func (ss *Sim) ConfigTstEpcLog() {
	dt := ss.TstEpcLog
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTstEpcPlot() {
	plt := ss.TstEpcPlot
	plt.Params.Title = "Leabra Random Associator 25 Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("Epoch", false, true, 0, false, 0)
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", true, true, 0, true, 1) // default plot
	plt.SetColParams("PctCor", true, true, 0, true, 1) // default plot
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log only contains Pats.NumRows() entries
func (ss *Sim) LogTstCyc(cyc int) {
	dt := ss.TstCycLog
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	hid1Lay := ss.Net.LayerByName("Hidden1").(*leabra.Layer)
	hid2Lay := ss.Net.LayerByName("Hidden2").(*leabra.Layer)
	outLay := ss.Net.LayerByName("Output").(*leabra.Layer)

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	dt.SetCellFloat("Hid1 Ge.Avg", cyc, float64(hid1Lay.Pools[0].Ge.Avg))
	dt.SetCellFloat("Hid2 Ge.Avg", cyc, float64(hid2Lay.Pools[0].Ge.Avg))
	dt.SetCellFloat("Out Ge.Avg", cyc, float64(outLay.Pools[0].Ge.Avg))
	dt.SetCellFloat("Hid1 Act.Avg", cyc, float64(hid1Lay.Pools[0].Act.Avg))
	dt.SetCellFloat("Hid2 Act.Avg", cyc, float64(hid2Lay.Pools[0].Act.Avg))
	dt.SetCellFloat("Out Act.Avg", cyc, float64(outLay.Pools[0].Act.Avg))

	if cyc%10 == 0 { // too slow to do every cyc
		ss.TstCycPlot.Update()
	}
}

func (ss *Sim) ConfigTstCycLog() {
	dt := ss.TstCycLog
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	np := 100 // max cycles
	dt.SetFromSchema(etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
		{"Hid1 Ge.Avg", etensor.FLOAT64, nil, nil},
		{"Hid2 Ge.Avg", etensor.FLOAT64, nil, nil},
		{"Out Ge.Avg", etensor.FLOAT64, nil, nil},
		{"Hid1 Act.Avg", etensor.FLOAT64, nil, nil},
		{"Hid2 Act.Avg", etensor.FLOAT64, nil, nil},
		{"Out Act.Avg", etensor.FLOAT64, nil, nil},
	}, np)
}

func (ss *Sim) ConfigTstCycPlot() {
	plt := ss.TstCycPlot
	plt.Params.Title = "Leabra Random Associator 25 Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", false, true, 0, false, 0)
	plt.SetColParams("Hid1 Ge.Avg", true, true, 0, true, .5)
	plt.SetColParams("Hid2 Ge.Avg", true, true, 0, true, .5)
	plt.SetColParams("Out Ge.Avg", true, true, 0, true, .5)
	plt.SetColParams("Hid1 Act.Avg", true, true, 0, true, .5)
	plt.SetColParams("Hid2 Act.Avg", true, true, 0, true, .5)
	plt.SetColParams("Out Act.Avg", true, true, 0, true, .5)
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun() {
	dt := ss.RunLog
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	ss.RunLog.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	// compute mean over last N epochs for run level
	nlast := 10
	epcix := etable.NewIdxView(epclog)
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast-1:]

	params := ss.ParamsName()

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(false)

	ss.RunPlot.Update()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, '\t')
		}
		dt.WriteCSVRow(ss.RunFile, row, '\t', true)
	}
}

func (ss *Sim) ConfigRunLog() {
	dt := ss.RunLog
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigRunPlot() {
	plt := ss.RunPlot
	plt.Params.Title = "Leabra Random Associator 25 Run Plot"
	plt.Params.XAxisCol = "Run"
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", false, true, 0, false, 0)
	plt.SetColParams("FirstZero", true, true, 0, false, 0) // default plot
	plt.SetColParams("SSE", false, true, 0, false, 0)
	plt.SetColParams("AvgSSE", false, true, 0, false, 0)
	plt.SetColParams("PctErr", false, true, 0, true, 1)
	plt.SetColParams("PctCor", false, true, 0, true, 1)
	plt.SetColParams("CosDiff", false, true, 0, true, 1)
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
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(ss.TrnEpcLog)
	ss.TrnEpcPlot = plt
	ss.ConfigTrnEpcPlot()

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

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(ss.TstEpcLog)
	ss.TstEpcPlot = plt
	ss.ConfigTstEpcPlot()

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	plt.Params.XAxisCol = "Run"
	plt.SetTable(ss.RunLog)
	ss.RunPlot = plt
	ss.ConfigRunPlot()

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Init()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			go ss.Train()
		})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Stop()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TrainTrial()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			go ss.TrainEpoch()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			go ss.TrainRun()
			vp.FullRender2DTree()
		})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TestTrial()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.StringPromptDialog(vp, "", "Test Item",
				gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					dlg := send.(*gi.Dialog)
					if sig == int64(gi.DialogAccepted) {
						val := gi.StringPromptDialogValue(dlg)
						idxs := ss.TestEnv.Table.RowsByString("Name", val, true, true) // contains, ignoreCase
						if len(idxs) == 0 {
							gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, true, false, nil, nil)
						} else {
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
						}
					}
				})
		})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			go ss.TestAll()
			vp.FullRender2DTree()
		})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			oswin.TheApp.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
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

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, true, true,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

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

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		TheSim.Init()
		win := TheSim.ConfigGui()
		win.StartEventLoop()
	}
}
