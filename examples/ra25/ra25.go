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

	"cogentcore.org/core/core"
	"cogentcore.org/core/events"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/agg"
	"cogentcore.org/core/tensor/split"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/views"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/path"
	"github.com/emer/leabra/v2/leabra"
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		guirun()
	}
}

func guirun() {
	TheSim.Init()
	TheSim.ConfigGUI()
	TheSim.GUI.Body.RunMainWindow()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {Desc: "these are the best params", Sheets: params.Sheets{
		"NetSize": &params.Sheet{
			{Sel: ".Hidden", Desc: "all hidden layers",
				Params: params.Params{
					"Layer.X": "7",
					"Layer.Y": "7",
				}},
		},
		"Network": &params.Sheet{
			{Sel: "Path", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Path.Learn.Norm.On":     "true",
					"Path.Learn.Momentum.On": "true",
					"Path.Learn.WtBal.On":    "true", // no diff really
					// "Path.Learn.WtBal.Targs": "true", // no diff here
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
					"Layer.Act.Init.Decay": "0.0",
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
	"DefaultInhib": {Desc: "output uses default inhib instead of lower", Sheets: params.Sheets{
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
	"NoMomentum": {Desc: "no momentum or normalization", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Path", Desc: "no norm or momentum",
				Params: params.Params{
					"Path.Learn.Norm.On":     "false",
					"Path.Learn.Momentum.On": "false",
				}},
		},
	}},
	"WtBalOn": {Desc: "try with weight bal on", Sheets: params.Sheets{
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
	Net *leabra.Network `display:"no-inline"`

	// all parameter management
	Params emer.Params `display:"inline"`

	// extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)
	Tag string

	// the training patterns to use
	Pats *table.Table `display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats

	// Contains all the logs and information about the logs.'
	Logs elog.Logs

	// starting run number -- typically 0 but can be set in command args for parallel runs on a cluster
	StartRun int

	// maximum number of model runs to perform (starting from StartRun)
	MaxRuns int

	// maximum number of epochs to run per model run
	MaxEpcs int

	// if a positive number, training will stop after this many epochs with zero SSE
	NZeroStop int

	// Training environment -- contains everything about iterating over input / output patterns over training
	TrainEnv env.FixedTable

	// Testing environment -- manages iterating over testing
	TestEnv env.FixedTable

	// leabra timing parameters and state
	Time leabra.Time

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"inline"`

	// how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing
	TestInterval int

	// how frequently (in epochs) to compute PCA on hidden representations to measure variance?
	PCAInterval int

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// for command-line run only, auto-save final weights after each run
	SaveWeights bool `display:"-"`

	// if true, runing in no GUI mode
	NoGui bool `display:"-"`

	// if true, print message for all params that are set
	LogSetParams bool `display:"-"`

	// flag to initialize NewRun if last one finished
	NeedsNewRun bool `display:"-"`

	// a list of random seeds to use for each run
	RndSeeds []int64 `display:"-"`

	// net data for recording in nogui mode
	NetData *netview.NetData `display:"-"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &table.Table{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.TestInterval = 5
	ss.PCAInterval = 5
	ss.Time.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	//ss.ConfigPats()
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
		ss.NZeroStop = 5
	}

	ss.TrainEnv.Name = "TrainEnv"
	ss.TrainEnv.Table = table.NewIndexView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Name = "TestEnv"
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
	ss.Params.AddLayers([]string{"Hidden1", "Hidden2"}, "Hidden")
	ss.Params.SetObject("NetSize")

	net.InitName(net, "RA25")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", ss.Params.LayY("Hidden1", 7), ss.Params.LayX("Hidden1", 7), emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", ss.Params.LayY("Hidden2", 7), ss.Params.LayX("Hidden2", 7), emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	// hid2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	// note: see emergent/path module for all the options on how to connect
	// NewFull returns a new path.Full connectivity pattern
	full := path.NewFull()

	net.ConnectLayers(inp, hid1, full, emer.Forward)
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
	ss.Params.SetObject("Network")
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
	ss.InitRndSeed()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetMsg = ss.LogSetParams
	ss.Params.SetAll()
	ss.NewRun()
	ss.ViewUpdate.Update()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.TrainEnv.Run.Cur
	rand.Seed(ss.RndSeeds[run])
}

// NewRndSeed gets a new set of random seeds based on current time -- otherwise uses
// the same random seeds for every run
func (ss *Sim) NewRndSeed() {
	rs := time.Now().UnixNano()
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = rs + int64(i)
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
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			ss.StatCounters(train)
			if !train {
				ss.Log(etime.Test, etime.Cycle)
			}
			ss.Time.CycleInc()
			ss.ViewUpdate.UpdateCycle(cyc)
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		ss.ViewUpdate.UpdateTime(etime.GammaCycle)
	}
	ss.StatCounters(train)

	if train {
		ss.Net.DWt()
		ss.ViewUpdate.RecordSyns() // note: critical to update weights here so DWt is visible
		ss.Net.WtFmDWt()
	}
	ss.ViewUpdate.UpdateTime(etime.AlphaCycle)
	if !train {
		ss.GUI.UpdatePlot(etime.Test, etime.Cycle) // make sure always updated at end
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
		if (ss.PCAInterval > 0) && ((epc-1)%ss.PCAInterval == 0) { // -1 so runs on first epc
			ss.PCAStats()
		}
		ss.Log(etime.Train, etime.Epoch)
		ss.ViewUpdate.UpdateTime(etime.Epoch)
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.GUI.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true) // train
	ss.TrialStats()
	ss.Log(etime.Train, etime.Trial)
	if (ss.PCAInterval > 0) && (epc%ss.PCAInterval == 0) {
		ss.Log(etime.Analyze, etime.Trial)
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(etime.Train, etime.Run)
	if ss.SaveWeights {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %s\n", fnm)
		ss.Net.SaveWeightsJSON(core.Filename(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWeights()
	ss.InitStats()
	ss.StatCounters(true)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
	ss.NeedsNewRun = false
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.GUI.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.GUI.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.GUI.StopNow = false
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.GUI.Stopped()
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
		ss.ViewUpdate.UpdateTime(etime.Epoch)
		ss.Log(etime.Test, etime.Epoch)
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false) // !train
	ss.TrialStats()
	ss.Log(etime.Test, etime.Trial)
	if ss.NetData != nil { // offline record net data from testing, just final state
		ss.NetData.Record(ss.ViewUpdate.Text, -1, 1)
	}
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false) // !train
	ss.TrialStats()
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on change -- don't wrap
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.GUI.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	sch := table.Schema{
		{"Name", tensor.STRING, nil, nil},
		{"Input", tensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", tensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, 25)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("random_5x5_25_gen.tsv", table.Tab, table.Headers)
}

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", table.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	rn := ""
	if ss.Tag != "" {
		rn += ss.Tag + "_"
	}
	rn += ss.Params.Name()
	if ss.StartRun > 0 {
		rn += fmt.Sprintf("_%03d", ss.StartRun)
	}
	return rn
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Name + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Name + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("TrlErr", 0.0)
	ss.Stats.SetFloat("TrlSSE", 0.0)
	ss.Stats.SetFloat("TrlAvgSSE", 0.0)
	ss.Stats.SetFloat("TrlCosDiff", 0.0)
	ss.Stats.SetInt("FirstZero", -1) // critical to reset to -1
	ss.Stats.SetInt("NZero", 0)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them to the GUI, if the GUI is active
func (ss *Sim) StatCounters(train bool) {
	ev := ss.TrainEnv
	if !train {
		ev = ss.TestEnv
	}
	ss.Stats.SetInt("Run", ss.TrainEnv.Run.Cur)
	ss.Stats.SetInt("Epoch", ss.TrainEnv.Epoch.Cur)
	ss.Stats.SetInt("Trial", ev.Trial.Cur)
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "AvgSSE", "TrlErr", "TrlCosDiff"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	sse, avgsse := out.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	ss.Stats.SetFloat("TrlSSE", sse)
	ss.Stats.SetFloat("TrlAvgSSE", avgsse)
	ss.Stats.SetFloat("TrlCosDiff", float64(out.CosDiff.Cos))

	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

//////////////////////////////////////////////
//  Logging

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "Params")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows
	switch {
	case mode == etime.Test && time == etime.Epoch:
		ss.LogTestErrors()
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == etime.Cycle {
		ss.GUI.UpdateCyclePlot(etime.Test, ss.Time.Cycle)
	} else {
		ss.GUI.UpdatePlot(mode, time)
	}

	switch {
	case mode == etime.Train && time == etime.Run:
		ss.LogRunStats()
	}
}

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func (ss *Sim) LogTestErrors() {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIndexView("TestErrors")
	ix.Filter(func(et *table.Table, row int) bool {
		return et.CellFloat("Err", row) > 0 // include error trials
	})
	ss.Logs.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.Agg(allsp, "SSE", agg.AggSum)
	// note: can add other stats to compute
	ss.Logs.MiscTables["TestErrorStats"] = allsp.AggsToTable(table.AddAggName)
}

// LogRunStats records stats across all runs, at Train Run scope
func (ss *Sim) LogRunStats() {
	sk := etime.Scope(etime.Train, etime.Run)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIndexView("RunStats")

	spl := split.GroupBy(ix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.Logs.MiscTables["RunStats"] = spl.AggsToTable(table.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func (ss *Sim) PCAStats() {
	ss.Stats.PCAStats(ss.Logs.IndexView(etime.Analyze, etime.Trial), "ActM", ss.Net.LayersByClass("Hidden"))
	ss.Logs.ResetLog(etime.Analyze, etime.Trial)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Leabra Random Associator"
	ss.GUI.MakeBody(ss, "ra25", title, `This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{0, 0, 0}, math32.Vector3{0, 1, 0})
	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.Body.AddAppBar(func(tb *core.Toolbar) {
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Init", Icon: "update",
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Train",
			Icon:    icons.ModelTraining,
			Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					ss.GUI.IsRunning = true
					tb.ApplyStyleTree()
					tb.NeedsRender()
					go ss.Train()
				}
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Stop",
			Icon:    icons.Stop,
			Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.",
			Active:  egui.ActiveRunning,
			Func: func() {
				ss.Stop()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Step Trial",
			Icon:    icons.Step,
			Tooltip: "Advances one training trial at a time.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					ss.GUI.IsRunning = true
					go func() {
						ss.TrainTrial()
						ss.GUI.IsRunning = false
					}()
				}
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Step Epoch",
			Icon:    icons.FastForward,
			Tooltip: "Advances one epoch (complete set of training patterns) at a time.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					ss.GUI.IsRunning = true
					tb.ApplyStyleTree()
					tb.NeedsRender()
					go ss.TrainEpoch()
				}
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Step Run",
			Icon:    icons.FastForward,
			Tooltip: "Advances one full training Run at a time.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					ss.GUI.IsRunning = true
					tb.ApplyStyleTree()
					tb.NeedsRender()
					go ss.TrainRun()
				}
			},
		})

		////////////////////////////////////////////////
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Test Trial",
			Icon:    icons.Step,
			Tooltip: "Runs the next testing trial.",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					ss.GUI.IsRunning = true
					go func() {
						ss.TestTrial(false) // don't return on change -- wrap
						ss.GUI.IsRunning = false
					}()
				}
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Test Item",
			Icon:    icons.Step,
			Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
			Active:  egui.ActiveStopped,
			Func: func() {
				d := core.NewBody().AddTitle("Test item").AddText("Enter the name of a given input pattern to test (case insensitive, contains given string.")
				name := ""
				views.NewValue(d, &name)
				d.AddBottomBar(func(pw core.Widget) {
					d.AddCancel(pw)
					d.AddOK(pw).OnClick(func(e events.Event) {
						idxs := ss.TestEnv.Table.RowsByString("Name", name, table.Contains, table.IgnoreCase)
						if len(idxs) == 0 {
							core.MessageSnackbar(tb, fmt.Sprintf("Name %q not found", name))
						} else {
							if !ss.GUI.IsRunning {
								go func() {
									ss.GUI.IsRunning = true
									fmt.Printf("testing index: %d\n", idxs[0])
									ss.TestItem(idxs[0])
									ss.GUI.IsRunning = false
								}()
							}
						}
					})
				})
				d.RunDialog(tb)
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Test All",
			Icon:    icons.FastForward,
			Tooltip: "Run through the full set of testing items",
			Active:  egui.ActiveStopped,
			Func: func() {
				if !ss.GUI.IsRunning {
					ss.GUI.IsRunning = true
					tb.ApplyStyleTree()
					tb.NeedsRender()
					go ss.RunTestAll()
				}
			},
		})

		////////////////////////////////////////////////
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Reset run log",
			Icon:    icons.Reset,
			Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.Logs.ResetLog(etime.Train, etime.Run)
				ss.GUI.UpdatePlot(etime.Train, etime.Run)
			},
		})
		////////////////////////////////////////////////
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "New Seed",
			Icon:    icons.Add,
			Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.NewRndSeed()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "README",
			Icon:    icons.FileMarkdown,
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)
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
	var saveRunLog bool
	var saveNetData bool
	var note string
	flag.StringVar(&ss.Params.ExtraSets, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWeights, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&saveNetData, "netdata", false, "if true, save network activation etc data from testing trials, for later viewing in netview")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.Params.ExtraSets != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.Params.ExtraSets)
	}

	if saveEpcLog {
		fnm := ss.LogFileName("epc")
		ss.Logs.SetLogFile(etime.Train, etime.Epoch, fnm)
	}
	if saveRunLog {
		fnm := ss.LogFileName("run")
		ss.Logs.SetLogFile(etime.Train, etime.Run, fnm)
	}
	if saveNetData {
		ss.NetData = &netview.NetData{}
		ss.NetData.Init(ss.Net, 200, true, 200) // 200 = amount to save
	}
	if ss.SaveWeights {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs starting at %d\n", ss.MaxRuns, ss.StartRun)
	ss.TrainEnv.Run.Set(ss.StartRun)
	ss.TrainEnv.Run.Max = ss.StartRun + ss.MaxRuns
	ss.NewRun()
	ss.Train()

	ss.Logs.CloseLogFiles()

	if saveNetData {
		ndfn := ss.Net.Name + "_" + ss.RunName() + ".netdata.gz"
		ss.NetData.SaveJSON(core.Filename(ndfn))
	}
}
