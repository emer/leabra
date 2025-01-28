// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deep_fsa runs a DeepLeabra network on the classic Reber grammar
// finite state automaton problem.
package main

//go:generate core generate -add-types

import (
	"log"
	"os"

	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32/vecint"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/etensor/tensor/table"
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

// ParamSets is the default set of parameters.
// Base is always applied, and others can be optionally
// selected to apply on top of that.
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "norm and momentum on is critical, wt bal not as much but fine",
			Params: params.Params{
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.WtBal.On":    "true",
			}},
		{Sel: "Layer", Desc: "using default 1.8 inhib for hidden layers",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":     "1.8",
				"Layer.Learn.AvgL.Gain":    "1.5",  // key to lower relative to 2.5
				"Layer.Act.Gbar.L":         "0.1",  // lower leak = better
				"Layer.Inhib.ActAvg.Fixed": "true", // simpler to have everything fixed, for replicability
				"Layer.Act.Init.Decay":     "0",    // essential to have all layers no decay
			}},
		{Sel: ".SuperLayer", Desc: "fix avg act",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Fixed": "true",
			}},
		{Sel: ".BackPath", Desc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Path.WtScale.Rel": "0.2",
			}},
		{Sel: ".PulvinarLayer", Desc: "standard weight is .3 here for larger distributed reps. no learn",
			Params: params.Params{
				"Layer.Pulvinar.DriveScale": "0.8", // using .8 for localist layer
			}},
		{Sel: ".CTCtxtPath", Desc: "no weight balance on CT context paths -- makes a diff!",
			Params: params.Params{
				"Path.Learn.WtBal.On": "false", // this should be true for larger DeepLeabra models -- e.g., sg..
			}},
		{Sel: ".CTFromSuper", Desc: "initial weight = 0.5 much better than 0.8",
			Params: params.Params{
				"Path.WtInit.Mean": "0.5",
			}},
		{Sel: ".Input", Desc: "input layers need more inhibition",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":    "2.0",
				"Layer.Inhib.ActAvg.Init": "0.15",
			}},
		{Sel: "#HiddenPToHiddenCT", Desc: "critical to make this small so deep context dominates",
			Params: params.Params{
				"Path.WtScale.Rel": "0.05",
			}},
		{Sel: "#HiddenCTToHiddenCT", Desc: "testing",
			Params: params.Params{
				"Path.Learn.WtBal.On": "false",
			}},
	},
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {

	// network parameters
	Network map[string]any

	// size of hidden layer -- can use emer.LaySize for 4D layers
	Hidden1Size vecint.Vector2i `default:"{'X':7,'Y':7}" nest:"+"`

	// size of hidden layer -- can use emer.LaySize for 4D layers
	Hidden2Size vecint.Vector2i `default:"{'X':7,'Y':7}" nest:"+"`

	// Extra Param Sheet name(s) to use (space separated if multiple).
	// must be valid name as listed in compiled-in params or loaded params
	Sheet string

	// extra tag to add to file names and logs saved from this run
	Tag string

	// user note -- describe the run params etc -- like a git commit message for the run
	Note string

	// Name of the JSON file to input saved parameters from.
	File string `nest:"+"`

	// Save a snapshot of all current param and config settings
	// in a directory named params_<datestamp> (or _good if Good is true), then quit.
	// Useful for comparing to later changes and seeing multiple views of current params.
	SaveAll bool `nest:"+"`

	// For SaveAll, save to params_good for a known good params state.
	// This can be done prior to making a new release after all tests are passing.
	// add results to git to provide a full diff record of all params over time.
	Good bool `nest:"+"`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	// starting run number, which determines the random seed.
	// runs counts from there, can do all runs in parallel by launching
	// separate jobs with each run, runs = 1.
	Run int `default:"0"`

	// total number of runs to do when running Train
	NRuns int `default:"5" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"100"`

	// stop run after this number of perfect, zero-error epochs.
	NZero int `default:"2"`

	// total number of trials per epoch.  Should be an even multiple of NData.
	NTrials int `default:"100"`

	// how often to run through all the test patterns, in terms of training epochs.
	// can use 0 or -1 for no testing.
	TestInterval int `default:"5"`

	// how frequently (in epochs) to compute PCA on hidden representations
	// to measure variance?
	PCAInterval int `default:"5"`

	// if non-empty, is the name of weights file to load at start
	// of first run, for testing.
	StartWts string
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// if true, save final weights after each run
	SaveWeights bool

	// if true, save train epoch log to file, as .epc.tsv typically
	Epoch bool `default:"true" nest:"+"`

	// if true, save run log to file, as .run.tsv typically
	Run bool `default:"true" nest:"+"`

	// if true, save train trial log to file, as .trl.tsv typically. May be large.
	Trial bool `default:"false" nest:"+"`

	// if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there.
	TestEpoch bool `default:"false" nest:"+"`

	// if true, save testing trial log to file, as .tst_trl.tsv typically. May be large.
	TestTrial bool `default:"false" nest:"+"`

	// if true, save network activation etc data from testing trials,
	// for later viewing in netview.
	NetData bool
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// specify include files here, and after configuration,
	// it contains list of include files added.
	Includes []string

	// open the GUI -- does not automatically run -- if false,
	// then runs automatically and quits.
	GUI bool `default:"true"`

	// log debugging information
	Debug bool

	// InputNames are names of input letters
	InputNames []string

	// InputNameMap has indexes of InputNames
	InputNameMap map[string]int

	// parameter related configuration options
	Params ParamConfig `display:"add-fields"`

	// sim running related configuration options
	Run RunConfig `display:"add-fields"`

	// data logging related configuration options
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// network parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// the training patterns to use
	Patterns *table.Table `new-window:"+" display:"no-inline"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// leabra timing parameters and state
	Context leabra.Context `new-window:"+"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	econfig.Config(&ss.Config, "config.toml")
	ss.Config.InputNames = []string{"B", "T", "S", "X", "V", "P", "E"}
	ss.Net = leabra.NewNetwork("RA25")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.Patterns = &table.Table{}
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

//////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *FSAEnv
	if len(ss.Envs) == 0 {
		trn = &FSAEnv{}
		tst = &FSAEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*FSAEnv)
		tst = ss.Envs.ByMode(etime.Test).(*FSAEnv)
	}

	if ss.Config.InputNameMap == nil {
		ss.Config.InputNameMap = make(map[string]int, len(ss.Config.InputNames))
		for i, nm := range ss.Config.InputNames {
			ss.Config.InputNameMap[nm] = i
		}
	}

	// note: names must be standard here!
	trn.Name = etime.Train.String()
	trn.Seq.Max = 25 // 25 sequences per epoch training
	trn.TMatReber()

	tst.Name = etime.Test.String()
	tst.Seq.Max = 10
	tst.TMatReber() // todo: random

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	in := net.AddLayer2D("Input", 1, 7, leabra.InputLayer)
	hid, hidct, hidp := net.AddDeep2D("Hidden", 8, 8)

	hidp.Shape.CopyShape(&in.Shape)
	hidp.Drivers.Add("Input")

	trg := net.AddLayer2D("Targets", 1, 7, leabra.InputLayer) // just for visualization

	in.AddClass("Input")
	hidp.AddClass("Input")
	trg.AddClass("Input")

	hidct.PlaceRightOf(hid, 2)
	hidp.PlaceRightOf(in, 2)
	trg.PlaceBehind(hidp, 2)

	full := paths.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all

	net.ConnectLayers(in, hid, full, leabra.ForwardPath)

	// for this small localist model with longer-term dependencies,
	// these additional context pathways turn out to be essential!
	// larger models in general do not require them, though it might be
	// good to check
	net.ConnectCtxtToCT(hidct, hidct, full)
	// net.LateralConnectLayer(hidct, full) // note: this does not work AT ALL -- essential to learn from t-1
	net.ConnectCtxtToCT(in, hidct, full)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()
	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.Config.GUI {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trls := ss.Config.Run.NTrials

	ls.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	ls.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, trls).
		AddTime(etime.Cycle, 100)

	leabra.LooperStdPhases(ls, &ss.Context, ss.Net, 75, 99)                // plus phase timing
	leabra.LooperSimCycleAndLearn(ls, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	ls.Stacks[etime.Train].OnInit.Add("Init", func() { ss.Init() })

	for m, _ := range ls.Stacks {
		stack := ls.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	ls.Loop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Train stop early condition
	ls.Loop(etime.Train, etime.Epoch).IsDone.AddBool("NZeroStop", func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Config.Run.NZero
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
	})

	// Add Testing
	trainEpoch := ls.Loop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	ls.Loop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		leabra.LogTestErrors(&ss.Logs)
	})
	ls.Loop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := ls.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if ss.Config.Run.PCAInterval > 0 && trnEpc%ss.Config.Run.PCAInterval == 0 {
			leabra.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	ls.AddOnEndToAll("Log", func(mode, time enums.Enum) {
		ss.Log(mode.(etime.Modes), time.(etime.Times))
	})
	leabra.LooperResetLogBelow(ls, &ss.Logs)

	ls.Loop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := ls.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	ls.Loop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file, to look at later
	ls.Loop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		leabra.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		if ss.Config.Log.NetData {
			ls.Loop(etime.Test, etime.Trial).OnEnd.Add("NetDataRecord", func() {
				ss.GUI.NetDataRecord(ss.ViewUpdate.Text)
			})
		}
	} else {
		leabra.LooperUpdateNetView(ls, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
		leabra.LooperUpdatePlots(ls, &ss.GUI)
		ls.Stacks[etime.Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
		ls.Stacks[etime.Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	net.InitExt()

	ev := ss.Envs.ByMode(ctx.Mode).(*FSAEnv)
	ev.Step()
	ss.Stats.SetString("TrialName", ev.String())

	in := ss.Net.LayerByName("Input")
	trg := ss.Net.LayerByName("Targets")
	clrmsk, setmsk, _ := in.ApplyExtFlags()
	ns := ev.NNext.Values[0]
	for i := 0; i < ns; i++ {
		lbl := ev.NextLabels.Values[i]
		li, ok := ss.Config.InputNameMap[lbl]
		if !ok {
			log.Printf("Input label: %v not found in InputNames list of labels\n", lbl)
			continue
		}
		if i == 0 {
			in.ApplyExtValue(li, 1, clrmsk, setmsk, false)
		}
		trg.ApplyExtValue(li, 1, clrmsk, setmsk, false)
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.Loop(etime.Train, etime.Run).Counter.Cur)
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("UnitErr", 0.0)
	ss.Stats.SetFloat("CorSim", 0.0)
	ss.Stats.SetString("TrialName", "")
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	if tm == etime.Trial {
		ss.TrialStats() // get trial stats for current di
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "UnitErr", "TrlErr", "CorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	inp := ss.Net.LayerByName("HiddenP")
	trg := ss.Net.LayerByName("Targets")
	ss.Stats.SetFloat("CorSim", float64(inp.CosDiff.Cos))
	sse := 0.0
	gotOne := false
	for ni := range inp.Neurons {
		inn := &inp.Neurons[ni]
		tgn := &trg.Neurons[ni]
		if tgn.Act > 0.5 {
			if inn.ActM > 0.4 {
				gotOne = true
			}
		} else {
			if inn.ActM > 0.5 {
				sse += float64(inn.ActM)
			}
		}
	}
	if !gotOne {
		sse += 1
	}
	ss.Stats.SetFloat("SSE", sse)
	ss.Stats.SetFloat("AvgSSE", sse)
	if sse > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("CorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddCopyFromFloatItems(etime.Train, []etime.Times{etime.Epoch, etime.Run}, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	layers := ss.Net.LayersByType(leabra.SuperLayer, leabra.CTLayer, leabra.TargetLayer)
	leabra.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	leabra.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	leabra.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "TargetLayer")

	ss.Logs.PlotItems("CorSim", "PctCor", "FirstZero", "LastZero")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		ss.TrialStats()
		ss.StatCounters()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Leabra Random Associator"
	ss.GUI.MakeBody(ss, "ra25", title, `This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	// nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	// nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    icons.Reset,
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/emer/leabra/blob/main/examples/ra25/README.md")
		},
	})
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.TestEpoch, etime.Test, etime.Epoch, "tst_epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.TestTrial, etime.Test, etime.Trial, "tst_trl", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.Loop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	if ss.Config.Run.StartWts != "" { // this is just for testing -- not usually needed
		ss.Loops.Step(etime.Train, 1, etime.Trial) // get past NewRun
		ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWts))
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWts)
	}

	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
