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
	"strconv"
	"time"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/plot"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/split"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tensor/tensormpi"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/leabra/v2/leabra"
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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Path", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Path.Learn.Norm.On":     "true",
					"Path.Learn.Momentum.On": "true",
					"Path.Learn.WtBal.On":    "false",
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
					"Layer.Act.Gbar.L":     "0.1", // set explictly, new default, a bit better vs 0.2
				}},
			{Sel: ".Back", Desc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Path.WtScale.Rel": "0.2",
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.4",
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
			{Sel: "Path", Desc: "no norm or momentum",
				Params: params.Params{
					"Path.Learn.Norm.On":     "false",
					"Path.Learn.Momentum.On": "false",
				}},
		},
	}},
	{Name: "WtBalOn", Desc: "try with weight bal on", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Path", Desc: "weight bal on",
				Params: params.Params{
					"Path.Learn.WtBal.On": "true",
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

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// the training patterns to use
	Pats *table.Table `display:"no-inline"`

	// training epoch-level log data
	TrnEpcLog *table.Table `display:"no-inline"`

	// training trial-level log data
	TrnTrlLog *table.Table `display:"no-inline"`

	// all training trial-level log data (aggregated from MPI)
	TrnTrlLogAll *table.Table `display:"no-inline"`

	// testing epoch-level log data
	TstEpcLog *table.Table `display:"no-inline"`

	// testing trial-level log data
	TstTrlLog *table.Table `display:"no-inline"`

	// all testing trial-level log data (aggregated from MPI)
	TstTrlLogAll *table.Table `display:"no-inline"`

	// log of all test trials where errors were made
	TstErrLog *table.Table `display:"no-inline"`

	// stats on test trials where errors were made
	TstErrStats *table.Table `display:"no-inline"`

	// testing cycle-level log data
	TstCycLog *table.Table `display:"no-inline"`

	// summary log of each run
	RunLog *table.Table `display:"no-inline"`

	// aggregate stats on all runs
	RunStats *table.Table `display:"no-inline"`

	// full collection of param sets
	Params params.Sets `display:"no-inline"`

	// which set of *additional* parameters to use -- always applies Base and optionaly this next if set
	ParamSet string

	// extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)
	Tag string

	// maximum number of model runs to perform
	MaxRuns int

	// maximum number of epochs to run per model run
	MaxEpcs int

	// if a positive number, training will stop after this many epochs with zero SSE
	NZeroStop int

	// Training environment -- contains everything about iterating over input / output patterns over training -- NOTE: using empi version
	TrainEnv env.MPIFixedTable

	// Testing environment -- manages iterating over testing -- NOTE: using empi version
	TestEnv env.MPIFixedTable

	// leabra timing parameters and state
	Time leabra.Context

	// whether to update the network view while running
	ViewOn bool

	// at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model
	TrainUpdate etime.Times

	// at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model
	TestUpdate etime.Times

	// how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing
	TestInterval int

	// names of layers to collect more detailed stats on (avg act, etc)
	LayStatNms []string

	// 1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)
	TrlErr float64 `edit:"-"`

	// current trial's sum squared error
	TrlSSE float64 `edit:"-"`

	// current trial's average sum squared error
	TrlAvgSSE float64 `edit:"-"`

	// current trial's cosine difference
	TrlCosDiff float64 `edit:"-"`

	// how long did the epoch take per trial in wall-clock milliseconds
	EpcPerTrlMSec float64 `edit:"-"`

	// epoch at when SSE first went to zero
	FirstZero int `edit:"-"`

	// number of epochs in a row with zero SSE
	NZero int `edit:"-"`

	// sum to increment as we go through epoch
	SumErr float64 `display:"-" edit:"-"`

	// sum to increment as we go through epoch
	SumSSE float64 `display:"-" edit:"-"`

	// sum to increment as we go through epoch
	SumAvgSSE float64 `display:"-" edit:"-"`

	// sum to increment as we go through epoch
	SumCosDiff float64 `display:"-" edit:"-"`

	// main GUI window
	Win *core.Window `display:"-"`

	// the network viewer
	NetView *netview.NetView `display:"-"`

	// the master toolbar
	ToolBar *core.ToolBar `display:"-"`

	// the training epoch plot
	TrnEpcPlot *plot.Plot2D `display:"-"`

	// the train-trial plot
	TrnTrlPlot *plot.Plot2D `display:"-"`

	// the testing epoch plot
	TstEpcPlot *plot.Plot2D `display:"-"`

	// the test-trial plot
	TstTrlPlot *plot.Plot2D `display:"-"`

	// the test-cycle plot
	TstCycPlot *plot.Plot2D `display:"-"`

	// the run plot
	RunPlot *plot.Plot2D `display:"-"`

	// log file
	TrnEpcFile *os.File `display:"-"`

	// log file
	TrnTrlFile *os.File `display:"-"`

	// log file
	RunFile *os.File `display:"-"`

	// for holding layer values
	ValuesTsrs map[string]*tensor.Float32 `display:"-"`

	// for command-line run only, auto-save final weights after each run
	SaveWeights bool `display:"-"`

	// if true, runing in no GUI mode
	NoGui bool `display:"-"`

	// if true, print message for all params that are set
	LogSetParams bool `display:"-"`

	// true if sim is running
	IsRunning bool `display:"-"`

	// flag to stop running
	StopNow bool `display:"-"`

	// flag to initialize NewRun if last one finished
	NeedsNewRun bool `display:"-"`

	// the current random seed
	RndSeed int64 `display:"-"`

	// timer for last epoch
	LastEpcTime time.Time `display:"-"`

	// if true, use MPI to distribute computation across nodes
	UseMPI bool `display:"-"`

	// mpi communicator
	Comm *mpi.Comm `display:"-"`

	// buffer of all dwt weight changes -- for mpi sharing
	AllDWts []float32 `display:"-"`

	// buffer of MPI summed dwt weight changes
	SumDWts []float32 `display:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &table.Table{}
	ss.TrnEpcLog = &table.Table{}
	ss.TrnTrlLog = &table.Table{}
	ss.TrnTrlLogAll = &table.Table{}
	ss.TstEpcLog = &table.Table{}
	ss.TstTrlLog = &table.Table{}
	ss.TstTrlLogAll = &table.Table{}
	ss.TstCycLog = &table.Table{}
	ss.RunLog = &table.Table{}
	ss.RunStats = &table.Table{}
	ss.Params = ParamSets
	ss.RndSeed = 1
	ss.ViewOn = true
	ss.TrainUpdate = leabra.AlphaCycle
	ss.TestUpdate = leabra.Cycle
	ss.TestInterval = 5
	ss.LayStatNms = []string{"Hidden1", "Hidden2", "Output"}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	//ss.ConfigPats()
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLogAll)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstTrlLog(ss.TstTrlLogAll)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
		ss.NZeroStop = 5
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Table = table.NewIndexView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Table = table.NewIndexView(ss.Pats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	// note: to create a train / test split of pats, do this:
	// all := table.NewIndexView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "RA25")
	inp := net.AddLayer2D("Input", 5, 5, leabra.InputLayer)
	hid1 := net.AddLayer2D("Hidden1", 7, 7, leabra.SuperLayer)
	hid2 := net.AddLayer4D("Hidden2", 2, 4, 3, 2, leabra.SuperLayer)
	out := net.AddLayer2D("Output", 5, 5, leabra.TargetLayer)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	hid2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	// note: see emergent/path module for all the options on how to connect
	// NewFull returns a new paths.Full connectivity pattern
	full := paths.NewFull()

	net.ConnectLayers(inp, hid1, full, leabra.ForwardPath)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	// if Thread {
	// 	hid2.SetThread(1)
	// 	out.SetThread(1)
	// }

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWeights()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

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
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train), -1)
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFromDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdate := ss.TrainUpdate
	if !train {
		viewUpdate = ss.TestUpdate
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.MPIWtFromDWt() // special MPI version
	}

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdate {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
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
			switch {
			case viewUpdate <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdate == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdate == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	// ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Name)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdate > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train
	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWeights {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWeightsJSON(core.Filename(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWeights()
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlErr = 0
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlSSE, ss.TrlAvgSSE = out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	if ss.TrlSSE > 0 {
		ss.TrlErr = 1
	} else {
		ss.TrlErr = 0
	}
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
	ss.Stopped()
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
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with views.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename core.Filename) {
	ss.Net.SaveWeightsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdate > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
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
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on change -- don't wrap
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

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
	pset, err := ss.Params.SetByName(setNm)
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
	dt.SetFromSchema(table.Schema{
		{"Name", tensor.STRING, nil, nil},
		{"Input", tensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", tensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}, 24)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("random_5x5_24_gen.csv", table.Comma, table.Headers)
}

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_24.tsv", table.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValuesTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValuesTsr(name string) *tensor.Float32 {
	if ss.ValuesTsrs == nil {
		ss.ValuesTsrs = make(map[string]*tensor.Float32)
	}
	tsr, ok := ss.ValuesTsrs[name]
	if !ok {
		tsr = &tensor.Float32{}
		ss.ValuesTsrs[name] = tsr
	}
	return tsr
}

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
	nm := ss.Net.Nm + "_" + ss.RunName() + "_" + lognm
	if mpi.WorldRank() > 0 {
		nm += fmt.Sprintf("_%d", mpi.WorldRank())
	}
	nm += ".tsv"
	return nm
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *table.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv          // this is triggered by increment so use previous value
	nt := float64(len(ss.TrainEnv.Order)) // number of trials in view

	trl := ss.TrnTrlLog
	if ss.UseMPI {
		tensormpi.GatherTableRows(ss.TrnTrlLogAll, ss.TrnTrlLog, ss.Comm)
		trl = ss.TrnTrlLogAll
	}

	tix := table.NewIndexView(trl)

	pcterr := stats.Mean(tix, "Err")[0]

	if ss.FirstZero < 0 && pcterr == 0 {
		ss.FirstZero = epc
	}
	if pcterr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, stats.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, stats.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, pcterr)
	dt.SetCellFloat("PctCor", row, 1-stats.Mean(tix, "Err")[0])
	dt.SetCellFloat("CosDiff", row, stats.Mean(tix, "CosDiff")[0])
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Name+"_ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, table.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, table.Tab)
	}

	if ss.TrnTrlFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			trl.WriteCSVHeaders(ss.TrnTrlFile, table.Tab)
		}
		for ri := 0; ri < trl.Rows; ri++ {
			trl.WriteCSVRow(ss.TrnTrlFile, ri, table.Tab)
		}
	}

	ss.TrnTrlLog.SetNumRows(0) // reset
}

func (ss *Sim) ConfigTrnEpcLog(dt *table.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"PctErr", tensor.FLOAT64, nil, nil},
		{"PctCor", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
		{"PerTrlMSec", tensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, table.Column{lnm + "_ActAvg", tensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("PctErr", plot.On, plot.FixMin, 0, plot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", plot.On, plot.FixMin, 0, plot.FixMax, 1) // default plot
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"_ActAvg", plot.Off, plot.FixMin, 0, plot.FixMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *table.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur
	row := dt.Rows

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *table.Table) {
	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"Trial", tensor.INT64, nil, nil},
		{"TrialName", tensor.STRING, nil, nil},
		{"Err", tensor.FLOAT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Train Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Trial", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("TrialName", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Err", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.On, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("CosDiff", plot.On, plot.FixMin, 0, plot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *table.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	inp := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	trl := ss.TestEnv.Trial.Cur
	row := dt.Rows

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Name+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	ivt := ss.ValuesTsr("Input")
	ovt := ss.ValuesTsr("Output")
	inp.UnitValuesTensor(ivt, "Act")
	dt.SetCellTensor("InAct", row, ivt)
	out.UnitValuesTensor(ovt, "ActM")
	dt.SetCellTensor("OutActM", row, ovt)
	out.UnitValuesTensor(ovt, "ActP")
	dt.SetCellTensor("OutActP", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *table.Table) {
	inp := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"Trial", tensor.INT64, nil, nil},
		{"TrialName", tensor.STRING, nil, nil},
		{"Err", tensor.FLOAT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, table.Column{lnm + " ActM.Avg", tensor.FLOAT64, nil, nil})
	}
	sch = append(sch, table.Schema{
		{"InAct", tensor.FLOAT64, inp.Shp.Shp, nil},
		{"OutActM", tensor.FLOAT64, out.Shp.Shp, nil},
		{"OutActP", tensor.FLOAT64, out.Shp.Shp, nil},
	}...)
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstTrlPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Trial", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("TrialName", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Err", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.On, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("CosDiff", plot.On, plot.FixMin, 0, plot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", plot.Off, plot.FixMin, 0, plot.FixMax, .5)
	}

	plt.SetColParams("InAct", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("OutActM", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("OutActP", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *table.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	if ss.UseMPI {
		tensormpi.GatherTableRows(ss.TstTrlLogAll, ss.TstTrlLog, ss.Comm)
		trl = ss.TstTrlLogAll
	}

	tix := table.NewIndexView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, stats.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, stats.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, stats.Mean(tix, "Err")[0])
	dt.SetCellFloat("PctCor", row, 1-stats.Mean(tix, "Err")[0])
	dt.SetCellFloat("CosDiff", row, stats.Mean(tix, "CosDiff")[0])

	trlix := table.NewIndexView(trl)
	trlix.Filter(func(et *table.Table, row int) bool {
		return et.CellFloat("SSE", row) > 0 // include error trials
	})
	ss.TstErrLog = trlix.NewTable()

	allsp := split.All(trlix)
	split.Agg(allsp, "SSE", stats.AggSum)
	split.Agg(allsp, "AvgSSE", stats.AggMean)
	split.Agg(allsp, "InAct", stats.AggMean)
	split.Agg(allsp, "OutActM", stats.AggMean)
	split.Agg(allsp, "OutActP", stats.AggMean)

	ss.TstErrStats = allsp.AggsToTable(table.AddAggName)

	ss.TstTrlLog.SetNumRows(0)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *table.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(table.Schema{
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"PctErr", tensor.FLOAT64, nil, nil},
		{"PctCor", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("PctErr", plot.On, plot.FixMin, 0, plot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", plot.On, plot.FixMin, 0, plot.FixMax, 1) // default plot
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *table.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Name+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Name+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *table.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := table.Schema{
		{"Cycle", tensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, table.Column{lnm + " Ge.Avg", tensor.FLOAT64, nil, nil})
		sch = append(sch, table.Column{lnm + " Act.Avg", tensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", true, true, 0, true, .5)
		plt.SetColParams(lnm+" Act.Avg", true, true, 0, true, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *table.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	epcix := table.NewIndexView(epclog)
	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Indexes = epcix.Indexes[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("SSE", row, stats.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, stats.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, stats.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, stats.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, stats.Mean(epcix, "CosDiff")[0])

	runix := table.NewIndexView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(table.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, table.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, table.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *table.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(table.Schema{
		{"Run", tensor.INT64, nil, nil},
		{"Params", tensor.STRING, nil, nil},
		{"FirstZero", tensor.FLOAT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"PctErr", tensor.FLOAT64, nil, nil},
		{"PctCor", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigRunPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("FirstZero", plot.On, plot.FixMin, 0, plot.FloatMax, 0) // default plot
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("PctErr", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("PctCor", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Window {
	width := 1600
	height := 1200

	// core.WinEventTrace = true

	core.SetAppName("ra25")
	core.SetAppAbout(`This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := core.NewMainWindow("ra25", "Leabra Random Associator", width, height)
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

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Options.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv

	nv.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(math32.Vector3{0, 0, 0}, math32.Vector3{0, 1, 0})

	plt := tv.AddNewTab(plot.KiT_Plot2D, "TrnEpcPlot").(*plot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TrnTrlPlot").(*plot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TstTrlPlot").(*plot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TstCycPlot").(*plot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TstEpcPlot").(*plot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "RunPlot").(*plot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(core.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(core.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *core.Action) {
			act.SetActiveStateUpdate(!ss.IsRunning)
		}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(core.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(core.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on change -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		core.StringPromptDialog(vp, "", "Test Item",
			core.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
				dlg := send.(*core.Dialog)
				if sig == int64(core.DialogAccepted) {
					val := core.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, table.Contains, table.IgnoreCase)
					if len(idxs) == 0 {
						core.PromptDialog(nil, core.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, core.AddOk, core.NoCancel, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(core.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(core.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(core.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(core.ActOpts{Label: "README", Icon: icons.FileMarkdown, Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			core.OpenURL("https://github.com/emer/leabra/blob/main/examples/ra25/README.md")
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

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*core.Action)
	// fmen.Menu.AddAction(core.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(core.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	core.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		core.PromptDialog(vp, core.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, core.AddOk, core.AddCancel,
			win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
				if sig == int64(core.DialogAccepted) {
					core.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// core.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *core.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		core.PromptDialog(vp, core.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, core.AddOk, core.AddCancel,
			win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
				if sig == int64(core.DialogAccepted) {
					core.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *core.Window) {
		go core.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = tree.Props{
	"CallMethods": tree.PropSlice{
		{"SaveWeights", tree.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": tree.PropSlice{
				{"File Name", tree.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveTrlLog bool
	var saveRunLog bool
	var saveProcLog bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWeights, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveTrlLog, "trllog", false, "if true, save train trial log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&saveProcLog, "proclog", false, "if true, save log files separately for each processor (for debugging)")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.BoolVar(&ss.UseMPI, "mpi", false, "if set, use MPI for distributed computation")
	flag.Parse()

	if ss.UseMPI {
		ss.MPIInit()
	}

	// key for Config and Init to be after MPIInit
	ss.Config()
	ss.Init()

	if note != "" {
		mpi.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		mpi.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog && (saveProcLog || mpi.WorldRank() == 0) {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			mpi.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveTrlLog && (saveProcLog || mpi.WorldRank() == 0) {
		var err error
		fnm := ss.LogFileName("trl")
		ss.TrnTrlFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnTrlFile = nil
		} else {
			mpi.Printf("Saving trial log to: %v\n", fnm)
			defer ss.TrnTrlFile.Close()
		}
	}
	if saveRunLog && (saveProcLog || mpi.WorldRank() == 0) {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			mpi.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}

	if ss.SaveWeights {
		if mpi.WorldRank() != 0 {
			ss.SaveWeights = false
		}
		mpi.Printf("Saving final weights per run\n")
	}
	mpi.Printf("Running %d Runs\n", ss.MaxRuns)
	ss.Train()
	ss.MPIFinalize()
}

////////////////////////////////////////////////////////////////////
//  MPI code

// MPIInit initializes MPI
func (ss *Sim) MPIInit() {
	mpi.Init()
	var err error
	ss.Comm, err = mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
		ss.UseMPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.UseMPI {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
func (ss *Sim) CollectDWts(net *leabra.Network) {
	made := net.CollectDWts(&ss.AllDWts, 8329) // plug in number from printout below, to avoid realloc
	if made {
		mpi.Printf("MPI: AllDWts len: %d\n", len(ss.AllDWts)) // put this number in above make
	}
}

// MPIWtFromDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFromDWt() {
	if ss.UseMPI {
		ss.CollectDWts(ss.Net)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts)
	}
	ss.Net.WtFromDWt()
}
