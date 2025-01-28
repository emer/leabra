// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build not

// hip_bench runs a hippocampus model for testing parameters and new learning ideas
package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/vecint"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/etensor/plot"
	"github.com/emer/etensor/tensor"
	"github.com/emer/etensor/tensor/stats/metric"
	"github.com/emer/etensor/tensor/stats/simat"
	"github.com/emer/etensor/tensor/stats/split"
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

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// see def_params.go for the default params, and params.go for user-saved versions
// from the gui.

// see bottom of file for multi-factor testing params

// HipParams have the hippocampus size and connectivity parameters
type HipParams struct {

	// size of EC in terms of overall pools (outer dimension)
	ECSize vecint.Vector2i

	// size of one EC pool
	ECPool vecint.Vector2i

	// size of one CA1 pool
	CA1Pool vecint.Vector2i

	// size of CA3
	CA3Size vecint.Vector2i

	// size of DG / CA3
	DGRatio float32

	// size of DG
	DGSize vecint.Vector2i `edit:"-"`

	// percent connectivity into DG
	DGPCon float32

	// percent connectivity into CA3
	CA3PCon float32

	// percent connectivity into CA3 from DG
	MossyPCon float32

	// percent activation in EC pool
	ECPctAct float32

	// delta in mossy effective strength between minus and plus phase
	MossyDel float32

	// delta in mossy strength for testing (relative to base param)
	MossyDelTest float32
}

func (hp *HipParams) Update() {
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)
}

// PatParams have the pattern parameters
type PatParams struct {

	// number of A-B, A-C patterns each
	ListSize int

	// minimum difference between item random patterns, as a proportion (0-1) of total active
	MinDiffPct float32

	// use drifting context representations -- otherwise does bit flips from prototype
	DriftCtxt bool

	// proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting
	CtxtFlipPct float32
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	//
	Net *leabra.Network `new-window:"+" display:"no-inline"`

	// hippocampus sizing parameters
	Hip HipParams

	// parameters for the input patterns
	Pat PatParams

	// pool patterns vocabulary
	PoolVocab patgen.Vocab `display:"no-inline"`

	// AB training patterns to use
	TrainAB *table.Table `display:"no-inline"`

	// AC training patterns to use
	TrainAC *table.Table `display:"no-inline"`

	// AB testing patterns to use
	TestAB *table.Table `display:"no-inline"`

	// AC testing patterns to use
	TestAC *table.Table `display:"no-inline"`

	// Lure pretrain patterns to use
	PreTrainLure *table.Table `display:"no-inline"`

	// Lure testing patterns to use
	TestLure *table.Table `display:"no-inline"`

	// all training patterns -- for pretrain
	TrainAll *table.Table `display:"no-inline"`

	// training trial-level log data for pattern similarity
	TrnCycPatSimLog *table.Table `display:"no-inline"`

	// training trial-level log data
	TrnTrlLog *table.Table `display:"no-inline"`

	// training epoch-level log data
	TrnEpcLog *table.Table `display:"no-inline"`

	// testing epoch-level log data
	TstEpcLog *table.Table `display:"no-inline"`

	// testing trial-level log data
	TstTrlLog *table.Table `display:"no-inline"`

	// testing cycle-level log data
	TstCycLog *table.Table `display:"no-inline"`

	// summary log of each run
	RunLog *table.Table `display:"no-inline"`

	// aggregate stats on all runs
	RunStats *table.Table `display:"no-inline"`

	// testing stats
	TstStats *table.Table `display:"no-inline"`

	// similarity matrix results for layers
	SimMats map[string]*simat.SimMat `display:"no-inline"`

	// full collection of param sets
	Params params.Sets `display:"no-inline"`

	// which set of *additional* parameters to use -- always applies Base and optionaly this next if set
	ParamSet string

	// extra tag string to add to any file names output from sim (e.g., weights files, log files, params)
	Tag string

	// current batch run number, for generating different seed
	BatchRun int

	// maximum number of model runs to perform
	MaxRuns int

	// maximum number of epochs to run per model run
	MaxEpcs int

	// number of epochs to run for pretraining
	PreTrainEpcs int

	// if a positive number, training will stop after this many epochs with zero mem errors
	NZeroStop int

	// Training environment -- contains everything about iterating over input / output patterns over training
	TrainEnv env.FixedTable

	// Testing environment -- manages iterating over testing
	TestEnv env.FixedTable

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

	// threshold to use for memory test -- if error proportion is below this number, it is scored as a correct trial
	MemThr float64

	// slice of slice for logging DG patterns every trial
	dgCycPats [100][]float32

	// slice of slice for logging CA3 patterns every trial
	ca3CycPats [100][]float32

	// slice of slice for logging CA1 patterns every trial
	ca1CycPats [100][]float32

	// what set of patterns are we currently testing
	TestNm string `edit:"-"`

	// whether current trial's ECout met memory criterion
	Mem float64 `edit:"-"`

	// current trial's proportion of bits where target = on but ECout was off ( < 0.5), for all bits
	TrgOnWasOffAll float64 `edit:"-"`

	// current trial's proportion of bits where target = on but ECout was off ( < 0.5), for only completion bits that were not active in ECin
	TrgOnWasOffCmp float64 `edit:"-"`

	// current trial's proportion of bits where target = off but ECout was on ( > 0.5)
	TrgOffWasOn float64 `edit:"-"`

	// current trial's sum squared error
	TrlSSE float64 `edit:"-"`

	// current trial's average sum squared error
	TrlAvgSSE float64 `edit:"-"`

	// current trial's cosine difference
	TrlCosDiff float64 `edit:"-"`

	// last epoch's total sum squared error
	EpcSSE float64 `edit:"-"`

	// last epoch's average sum squared error (average over trials, and over units within layer)
	EpcAvgSSE float64 `edit:"-"`

	// last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)
	EpcPctErr float64 `edit:"-"`

	// last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)
	EpcPctCor float64 `edit:"-"`

	// last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)
	EpcCosDiff float64 `edit:"-"`

	// how long did the epoch take per trial in wall-clock milliseconds
	EpcPerTrlMSec float64 `edit:"-"`

	// epoch at when Mem err first went to zero
	FirstZero int `edit:"-"`

	// number of epochs in a row with zero Mem err
	NZero int `edit:"-"`

	// sum to increment as we go through epoch
	SumSSE float64 `display:"-" edit:"-"`

	// sum to increment as we go through epoch
	SumAvgSSE float64 `display:"-" edit:"-"`

	// sum to increment as we go through epoch
	SumCosDiff float64 `display:"-" edit:"-"`

	// sum of errs to increment as we go through epoch
	CntErr int `display:"-" edit:"-"`

	// main GUI window
	Win *core.Window `display:"-"`

	// the network viewer
	NetView *netview.NetView `display:"-"`

	// the master toolbar
	ToolBar *core.ToolBar `display:"-"`

	// the training trial plot
	TrnTrlPlot *plot.Plot2D `display:"-"`

	// the training epoch plot
	TrnEpcPlot *plot.Plot2D `display:"-"`

	// the testing epoch plot
	TstEpcPlot *plot.Plot2D `display:"-"`

	// the test-trial plot
	TstTrlPlot *plot.Plot2D `display:"-"`

	// the test-cycle plot
	TstCycPlot *plot.Plot2D `display:"-"`

	// the run plot
	RunPlot *plot.Plot2D `display:"-"`

	// the run stats plot - ABmem
	RunStatsPlot1 *plot.Plot2D `display:"-"`

	// the run stats plot - learning time
	RunStatsPlot2 *plot.Plot2D `display:"-"`

	// log file
	TrnCycPatSimFile *os.File `display:"-"`

	// headers written
	TrnCycPatSimHdrs bool `display:"-"`

	// log file
	TstEpcFile *os.File `display:"-"`

	// headers written
	TstEpcHdrs bool `display:"-"`

	// log file
	RunFile *os.File `display:"-"`

	// headers written
	RunHdrs bool `display:"-"`

	// temp slice for holding values -- prevent mem allocs
	TmpValues []float32 `display:"-"`

	// names of layers to collect more detailed stats on (avg act, etc)
	LayStatNms []string `display:"-"`

	// names of test tables
	TstNms []string `display:"-"`

	// names of sim mat stats
	SimMatStats []string `display:"-"`

	// names of test stats
	TstStatNms []string `display:"-"`

	// for holding layer values
	ValuesTsrs map[string]*tensor.Float32 `display:"-"`

	// for command-line run only, auto-save final weights after each run
	SaveWeights bool `display:"-"`

	// pretrained weights file
	PreTrainWts []byte `display:"-"`

	// if true, pretraining is done
	PretrainDone bool `display:"-"`

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
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.PoolVocab = patgen.Vocab{}
	ss.TrainAB = &table.Table{}
	ss.TrainAC = &table.Table{}
	ss.TestAB = &table.Table{}
	ss.TestAC = &table.Table{}
	ss.PreTrainLure = &table.Table{}
	ss.TestLure = &table.Table{}
	ss.TrainAll = &table.Table{}
	ss.TrnCycPatSimLog = &table.Table{}
	ss.TrnTrlLog = &table.Table{}
	ss.TrnEpcLog = &table.Table{}
	ss.TstEpcLog = &table.Table{}
	ss.TstTrlLog = &table.Table{}
	ss.TstCycLog = &table.Table{}
	ss.RunLog = &table.Table{}
	ss.RunStats = &table.Table{}
	ss.SimMats = make(map[string]*simat.SimMat)
	ss.Params = ParamSets // in def_params -- current best params
	//ss.Params = OrigParamSets // key for original param in Ketz et al. 2013
	//ss.Params = SavedParamsSets // user-saved gui params
	ss.RndSeed = 2
	ss.ViewOn = true
	ss.TrainUpdate = leabra.AlphaCycle
	ss.TestUpdate = leabra.Cycle
	ss.TestInterval = 1
	ss.LogSetParams = false
	ss.MemThr = 0.34
	ss.LayStatNms = []string{"Input", "ECin", "ECout", "DG", "CA3", "CA1"}
	ss.TstNms = []string{"AB", "AC", "Lure"}
	ss.TstStatNms = []string{"Mem", "TrgOnWasOff", "TrgOffWasOn"}
	ss.SimMatStats = []string{"WithinAB", "WithinAC", "Between"}

	ss.Defaults()
}

func (pp *PatParams) Defaults() {
	pp.ListSize = 175
	pp.MinDiffPct = 0.5
	pp.CtxtFlipPct = .25
}

func (hp *HipParams) Defaults() {
	// size
	hp.ECSize.Set(2, 3)
	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(20, 20) // using BigHip for default
	hp.CA3Size.Set(40, 40) // using BigHip for default
	hp.DGRatio = 2.236     // sqrt(5) = 2.236 c.f. Ketz et al., 2013

	// ratio
	hp.DGPCon = 0.25
	hp.CA3PCon = 0.25
	hp.MossyPCon = 0.02
	hp.ECPctAct = 0.2

	hp.MossyDel = 4     // 4 > 2 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 3 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount..
}

func (ss *Sim) Defaults() {
	ss.Hip.Defaults()
	ss.Pat.Defaults()
	ss.BatchRun = 0        // for initializing envs if using Gui
	ss.Time.CycPerQtr = 25 // note: key param - 25 seems like it is actually fine?
	ss.Update()
}

func (ss *Sim) Update() {
	ss.Hip.Update()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnCycPatSimLog(ss.TrnCycPatSimLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 30
		ss.NZeroStop = 1
		ss.PreTrainEpcs = 5 // seems sufficient? increase?
	}

	ss.TrainEnv.Name = "TrainEnv"
	ss.TrainEnv.Table = table.NewIndexView(ss.TrainAB)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Name = "TestEnv"
	ss.TestEnv.Table = table.NewIndexView(ss.TestAB)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(ss.BatchRun)
	ss.TestEnv.Init(ss.BatchRun)
}

// SetEnv select which set of patterns to train on: AB or AC
func (ss *Sim) SetEnv(trainAC bool) {
	if trainAC {
		ss.TrainEnv.Table = table.NewIndexView(ss.TrainAC)
	} else {
		ss.TrainEnv.Table = table.NewIndexView(ss.TrainAB)
	}
	ss.TrainEnv.Init(ss.BatchRun)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "Hip_bench")
	hp := &ss.Hip
	in := net.AddLayer4D("Input", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, leabra.InputLayer)
	ecin := net.AddLayer4D("ECin", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, leabra.SuperLayer)
	ecout := net.AddLayer4D("ECout", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, leabra.TargetLayer) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", hp.ECSize.Y, hp.ECSize.X, hp.CA1Pool.Y, hp.CA1Pool.X, leabra.SuperLayer)
	dg := net.AddLayer2D("DG", hp.DGSize.Y, hp.DGSize.X, leabra.SuperLayer)
	ca3 := net.AddLayer2D("CA3", hp.CA3Size.Y, hp.CA3Size.X, leabra.SuperLayer)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	ecin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	ecout.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ECin", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "DG", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 2})

	onetoone := paths.NewOneToOne()
	pool1to1 := paths.NewPoolOneToOne()
	full := paths.NewFull()

	net.ConnectLayers(in, ecin, onetoone, leabra.ForwardPath)
	net.ConnectLayers(ecout, ecin, onetoone, BackPath)

	// EC <-> CA1 encoder pathways
	pj := net.ConnectLayersPath(ecin, ca1, pool1to1, leabra.ForwardPath, &leabra.EcCa1Path{})
	pj.SetClass("EcCa1Path")
	pj = net.ConnectLayersPath(ca1, ecout, pool1to1, leabra.ForwardPath, &leabra.EcCa1Path{})
	pj.SetClass("EcCa1Path")
	pj = net.ConnectLayersPath(ecout, ca1, pool1to1, BackPath, &leabra.EcCa1Path{})
	pj.SetClass("EcCa1Path")

	// Perforant pathway
	ppathDG := paths.NewUnifRnd()
	ppathDG.PCon = hp.DGPCon
	ppathCA3 := paths.NewUnifRnd()
	ppathCA3.PCon = hp.CA3PCon

	pj = net.ConnectLayersPath(ecin, dg, ppathDG, leabra.ForwardPath, &leabra.CHLPath{})
	pj.SetClass("HippoCHL")

	if true { // key for bcm vs. ppath, zycyc: must use false for orig_param, true for def_param
		pj = net.ConnectLayersPath(ecin, ca3, ppathCA3, leabra.ForwardPath, &leabra.EcCa1Path{})
		pj.SetClass("PPath")
		pj = net.ConnectLayersPath(ca3, ca3, full, emer.Lateral, &leabra.EcCa1Path{})
		pj.SetClass("PPath")
	} else {
		// so far, this is sig worse, even with error-driven MinusQ1 case (which is better than off)
		pj = net.ConnectLayersPath(ecin, ca3, ppathCA3, leabra.ForwardPath, &leabra.CHLPath{})
		pj.SetClass("HippoCHL")
		pj = net.ConnectLayersPath(ca3, ca3, full, emer.Lateral, &leabra.CHLPath{})
		pj.SetClass("HippoCHL")
	}

	// always use this for now:
	if true {
		pj = net.ConnectLayersPath(ca3, ca1, full, leabra.ForwardPath, &leabra.CHLPath{})
		pj.SetClass("HippoCHL")
	} else {
		// note: this requires lrate = 1.0 or maybe 1.2, doesn't work *nearly* as well
		pj = net.ConnectLayers(ca3, ca1, full, leabra.ForwardPath) // default con
		// pj.SetClass("HippoCHL")
	}

	// Mossy fibers
	mossy := paths.NewUnifRnd()
	mossy.PCon = hp.MossyPCon
	pj = net.ConnectLayersPath(dg, ca3, mossy, leabra.ForwardPath, &leabra.CHLPath{}) // no learning
	pj.SetClass("HippoCHL")

	// using 4 threads total (rest on 0)
	dg.(leabra.LeabraLayer).SetThread(1)
	ca3.(leabra.LeabraLayer).SetThread(2)
	ca1.(leabra.LeabraLayer).SetThread(3) // this has the most

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
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

func (ss *Sim) ReConfigNet() {
	ss.Update()
	ss.ConfigPats()
	ss.Net = &leabra.Network{} // start over with new network
	ss.ConfigNet(ss.Net)
	if ss.NetView != nil {
		ss.NetView.SetNet(ss.Net)
		ss.NetView.Update() // issue #41 closed
	}
	ss.ConfigTrnCycPatSimLog(ss.TrnCycPatSimLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	ss.SetParams("", ss.LogSetParams) // all sheets

	ss.ReConfigNet()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.NewRun()
	ss.UpdateView(true)
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.RecordSyns()
	}
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
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)
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

	dg := ss.Net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.SendName("ECin").(leabra.LeabraPath).AsLeabra()
	ca1FmCa3 := ca1.SendName("CA3").(leabra.LeabraPath).AsLeabra()
	ca3FmDg := ca3.SendName("DG").(leabra.LeabraPath).AsLeabra()
	_ = ecin
	_ = input

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	thetaLow := float32(0.3)
	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = thetaLow

	dgwtscale := ca3FmDg.WtScale.Rel
	ca3FmDg.WtScale.Rel = dgwtscale - ss.Hip.MossyDel // 0 for the first quarter, comment out if NoEDL and orig, zycyc. NoEDL key

	if train {
		ecout.SetType(leabra.TargetLayer) // clamp a plus phase during testing
	} else {
		ecout.SetType(emer.Compare) // don't clamp
	}
	ecout.UpdateExtFlags() // call this after updating type

	ss.Net.AlphaCycInit(train)
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			} else if ss.PretrainDone { // zycyc Pat Sim log
				var dgCycPat []float32
				var ca3CycPat []float32
				var ca1CycPat []float32
				dg.UnitValues(&dgCycPat, "Act")
				ca3.UnitValues(&ca3CycPat, "Act")
				ca1.UnitValues(&ca1CycPat, "Act")
				ss.dgCycPats[cyc+qtr*ss.Time.CycPerQtr] = dgCycPat
				ss.ca3CycPats[cyc+qtr*ss.Time.CycPerQtr] = ca3CycPat
				ss.ca1CycPats[cyc+qtr*ss.Time.CycPerQtr] = ca1CycPat
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
		switch qtr + 1 {
		case 1: // Second, Third Quarters: CA1 is driven by CA3 recall
			ca1FmECin.WtScale.Abs = thetaLow
			ca1FmCa3.WtScale.Abs = 1
			//ca3FmDg.WtScale.Rel = dgwtscale //zycyc, orig
			if train { // def
				ca3FmDg.WtScale.Rel = dgwtscale
			} else {
				ca3FmDg.WtScale.Rel = dgwtscale - ss.Hip.MossyDelTest // testing, NoDynMF key
			}
			ss.Net.GScaleFromAvgAct() // update computed scaling factors
			ss.Net.InitGInc()         // scaling params change, so need to recompute all netins
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = thetaLow
			ss.Net.GScaleFromAvgAct() // update computed scaling factors
			ss.Net.InitGInc()         // scaling params change, so need to recompute all netins
			if train {                // clamp ECout from ECin
				ecin.UnitValues(&ss.TmpValues, "Act") // note: could use input instead -- not much diff
				ecout.ApplyExt1D32(ss.TmpValues)
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
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

	ca3FmDg.WtScale.Rel = dgwtscale // restore
	ca1FmCa3.WtScale.Abs = 1

	if train {
		ss.Net.DWt()
		if ss.NetView != nil && ss.NetView.IsVisible() {
			ss.NetView.RecordSyns()
		}
		ss.Net.WtFromDWt() // so testing is based on updated weights
	}
	if ss.ViewOn && viewUpdate == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		if ss.TstCycPlot != nil {
			ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
		}
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "ECout"}
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

		// zycyc, fixed epoch num -- half AB half AC
		//if ss.TrainEnv.Table.Table == ss.TrainAB && (epc == ss.MaxEpcs/2) {
		//	ss.TrainEnv.Table = table.NewIndexView(ss.TrainAC)
		//}
		//if epc >= ss.MaxEpcs { // done with training..
		//	ss.RunEnd()
		//	if ss.TrainEnv.Run.Incr() { // we are done!
		//		ss.StopNow = true
		//		return
		//	} else {
		//		ss.NeedsNewRun = true
		//		return
		//	}
		//}

		// zycyc, half / learned (default)
		learned := (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop)
		if ss.TrainEnv.Table.Table == ss.TrainAB && (learned || epc == ss.MaxEpcs/2) { // switch to AC
			ss.TrainEnv.Table = table.NewIndexView(ss.TrainAC)

			// set names after updating epochs to get correct names for the next env
			ss.TrainEnv.SetTrialName()
			ss.TrainEnv.SetGroupName()

			learned = false
		}
		if learned || epc >= ss.MaxEpcs { // done with training..
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
	ss.LogTrnCycPatSim(ss.TrnCycPatSimLog) // zycyc pat sim log
}

// PreTrainTrial runs one trial of pretraining using TrainEnv
func (ss *Sim) PreTrainTrial() {
	//if ss.NeedsNewRun {
	//	ss.NewRun()
	//}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		//ss.LogTrnEpc(ss.TrnEpcLog) // zycyc, don't log pretraining
		if ss.ViewOn && ss.TrainUpdate > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if epc >= ss.PreTrainEpcs { // done with training..
			ss.StopNow = true
			return
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
	ss.TrainEnv.Table = table.NewIndexView(ss.TrainAB)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWeights()
	ss.LoadPretrainedWts()
	ss.InitStats()
	ss.TrnCycPatSimLog.SetNumRows(0)
	ss.TrnTrlLog.SetNumRows(0)
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

func (ss *Sim) LoadPretrainedWts() bool {
	if ss.PreTrainWts == nil {
		return false
	}
	b := bytes.NewReader(ss.PreTrainWts)
	err := ss.Net.ReadWtsJSON(b)
	if err != nil {
		log.Println(err)
		// } else {
		// 	fmt.Printf("loaded pretrained wts\n")
	}
	return true
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
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.Mem = 0
	ss.TrgOnWasOffAll = 0
	ss.TrgOnWasOffCmp = 0
	ss.TrgOffWasOn = 0
	ss.TrlSSE = 0
	ss.TrlAvgSSE = 0
	ss.EpcSSE = 0
	ss.EpcAvgSSE = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// MemStats computes ActM vs. Target on ECout with binary counts
// must be called at end of 3rd quarter so that Targ values are
// for the entire full pattern as opposed to the plus-phase target
// values clamped from ECin activations
func (ss *Sim) MemStats(train bool) {
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	nn := ecout.Shape.Len()
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	actMi, _ := ecout.UnitVarIndex("ActM")
	targi, _ := ecout.UnitVarIndex("Targ")
	actQ1i, _ := ecout.UnitVarIndex("ActQ1")
	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitValue1D(actMi, ni)
		trg := ecout.UnitValue1D(targi, ni) // full pattern target
		inact := ecin.UnitValue1D(actQ1i, ni)
		if trg < 0.5 { // trgOff
			trgOffN += 1
			if actm > 0.5 {
				trgOffWasOn += 1
			}
		} else { // trgOn
			trgOnN += 1
			if inact < 0.5 { // missing in ECin -- completion target
				cmpN += 1
				if actm < 0.5 {
					trgOnWasOffAll += 1
					trgOnWasOffCmp += 1
				}
			} else {
				if actm < 0.5 {
					trgOnWasOffAll += 1
				}
			}
		}
	}
	trgOnWasOffAll /= trgOnN
	trgOffWasOn /= trgOffN
	if train { // no cmp
		if trgOnWasOffAll < ss.MemThr && trgOffWasOn < ss.MemThr {
			ss.Mem = 1
		} else {
			ss.Mem = 0
		}
	} else { // test
		if cmpN > 0 { // should be
			trgOnWasOffCmp /= cmpN
			if trgOnWasOffCmp < ss.MemThr && trgOffWasOn < ss.MemThr {
				ss.Mem = 1
			} else {
				ss.Mem = 0
			}
		}
	}
	ss.TrgOnWasOffAll = trgOnWasOffAll
	ss.TrgOnWasOffCmp = trgOnWasOffCmp
	ss.TrgOffWasOn = trgOffWasOn
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) (sse, avgsse, cosdiff float64) {
	outLay := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
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

// SetDgCa3Off sets the DG and CA3 layers off (or on)
func (ss *Sim) SetDgCa3Off(net *leabra.Network, off bool) {
	ca3 := net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	dg := net.LayerByName("DG").(leabra.LeabraLayer).AsLeabra()
	ca3.Off = off
	dg.Off = off
}

// PreTrain runs pre-training, saves weights to PreTrainWts
func (ss *Sim) PreTrain() {
	ss.SetDgCa3Off(ss.Net, true)
	ss.TrainEnv.Table = table.NewIndexView(ss.TrainAll)
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(curRun) // need this after changing num of rows in tables
	for {
		ss.PreTrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	b := &bytes.Buffer{}
	ss.Net.WriteWtsJSON(b)
	ss.PreTrainWts = b.Bytes()
	ss.TrainEnv.Table = table.NewIndexView(ss.TrainAB)
	ss.SetDgCa3Off(ss.Net, false)
	ss.Stopped()
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
	ss.TestNm = "AB"
	ss.TestEnv.Table = table.NewIndexView(ss.TestAB)
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on chg
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
	if !ss.StopNow {
		ss.TestNm = "AC"
		ss.TestEnv.Table = table.NewIndexView(ss.TestAC)
		ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
		for {
			ss.TestTrial(true)
			_, _, chg := ss.TestEnv.Counter(env.Epoch)
			if chg || ss.StopNow {
				break
			}
		}
		if !ss.StopNow {
			ss.TestNm = "Lure"
			ss.TestEnv.Table = table.NewIndexView(ss.TestLure)
			ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
			for {
				ss.TestTrial(true)
				_, _, chg := ss.TestEnv.Counter(env.Epoch)
				if chg || ss.StopNow {
					break
				}
			}
		}
	}
	// log only at very end
	ss.LogTstEpc(ss.TstEpcLog)
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
		ss.Params.ValidateSheets([]string{"Network", "Sim", "Hip", "Pat"})
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

	if sheet == "" || sheet == "Hip" {
		simp, ok := pset.Sheets["Hip"]
		if ok {
			simp.Apply(&ss.Hip, setMsg)
		}
	}

	if sheet == "" || sheet == "Pat" {
		simp, ok := pset.Sheets["Pat"]
		if ok {
			simp.Apply(&ss.Pat, setMsg)
		}
	}

	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) OpenPat(dt *table.Table, fname, name, desc string) {
	err := dt.OpenCSV(core.Filename(fname), table.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
}

func (ss *Sim) ConfigPats() {
	hp := &ss.Hip
	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X
	plY := hp.ECPool.Y // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X // makes much more readable
	npats := ss.Pat.ListSize
	pctAct := hp.ECPctAct
	minDiff := ss.Pat.MinDiffPct
	nOn := patgen.NFromPct(pctAct, plY*plX)
	ctxtflip := patgen.NFromPct(ss.Pat.CtxtFlipPct, nOn)
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt", 3, plY, plX, pctAct, minDiff) // totally diff

	for i := 0; i < (ecY-1)*ecX*3; i++ { // 12 contexts! 1: 1 row of stimuli pats; 3: 3 diff ctxt bases
		list := i / ((ecY - 1) * ecX)
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(ss.PoolVocab, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
		//todo: also support drifting
		//solution 2: drift based on last trial (will require sequential learning)
		//patgen.VocabDrift(ss.PoolVocab, ss.NFlipBits, "ctxt"+strconv.Itoa(i+1))
	}

	patgen.InitPats(ss.TrainAB, "TrainAB", "TrainAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "Input", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TrainAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TestAB, "TestAB", "TestAB Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})
	patgen.MixPats(ss.TestAB, ss.PoolVocab, "ECout", []string{"A", "B", "ctxt1", "ctxt2", "ctxt3", "ctxt4"})

	patgen.InitPats(ss.TrainAC, "TrainAC", "TrainAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "Input", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TrainAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.TestAC, "TestAC", "TestAC Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "Input", []string{"A", "empty", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})
	patgen.MixPats(ss.TestAC, ss.PoolVocab, "ECout", []string{"A", "C", "ctxt5", "ctxt6", "ctxt7", "ctxt8"})

	patgen.InitPats(ss.PreTrainLure, "PreTrainLure", "PreTrainLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "Input", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.PreTrainLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here

	patgen.InitPats(ss.TestLure, "TestLure", "TestLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA", "empty", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})    // arbitrary ctxt here

	ss.TrainAll = ss.TrainAB.Clone()
	ss.TrainAll.AppendRows(ss.TrainAC)
	ss.TrainAll.AppendRows(ss.PreTrainLure)
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
		pnm := ss.ParamsName()
		if pnm == "Base" {
			return ss.Tag
		} else {
			return ss.Tag + "_" + pnm
		}
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
//  TrnCycPatSimLog

// LogTrnCycPatSim adds data from current trial to the TrnCycPatSimLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnCycPatSim(dt *table.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur

	var spltparams []string
	if len(os.Args) > 1 {
		params := ss.RunName() // includes tag
		spltparams = strings.Split(params, "_")
	} else {
		spltparams = append(spltparams, "Default")
		spltparams = append(spltparams, strconv.Itoa(ss.Pat.ListSize))
	}

	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}

	if ss.TrnCycPatSimFile != nil {
		if !ss.TrnCycPatSimHdrs {
			dt.WriteCSVHeaders(ss.TrnCycPatSimFile, table.Tab)
			ss.TrnCycPatSimHdrs = true
		}
		for iCyc := 0; iCyc < 100; iCyc += 1 { // zycyc: step control
			row += 1
			dt.SetNumRows(row + 1)
			//dt.SetCellString("Params", row, params)
			dt.SetCellString("NetSize", row, spltparams[0])
			dt.SetCellString("ListSize", row, spltparams[1])
			dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
			dt.SetCellFloat("Epoch", row, float64(epc))
			dt.SetCellFloat("Trial", row, float64(trl))
			dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur)
			dt.SetCellFloat("Cycle", row, float64(iCyc))
			dt.SetCellFloat("DG", row, float64(metric.Correlation32(ss.dgCycPats[iCyc], ss.dgCycPats[99])))
			dt.SetCellFloat("CA3", row, float64(metric.Correlation32(ss.ca3CycPats[iCyc], ss.ca3CycPats[99])))
			dt.SetCellFloat("CA1", row, float64(metric.Correlation32(ss.ca1CycPats[iCyc], ss.ca1CycPats[99])))
			dt.WriteCSVRow(ss.TrnCycPatSimFile, row, table.Tab)
		}
	}
}

func (ss *Sim) ConfigTrnCycPatSimLog(dt *table.Table) {
	dt.SetMetaData("name", "TrnCycLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := table.Schema{
		//{"Params", tensor.STRING, nil, nil},
		{"NetSize", tensor.STRING, nil, nil},
		{"ListSize", tensor.STRING, nil, nil},
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"Trial", tensor.INT64, nil, nil},
		{"TrialName", tensor.STRING, nil, nil},
		{"Cycle", tensor.INT64, nil, nil},
		{"DG", tensor.FLOAT64, nil, nil},
		{"CA3", tensor.FLOAT64, nil, nil},
		{"CA1", tensor.FLOAT64, nil, nil},
	}
	//for iCyc := 0; iCyc < 100; iCyc++ {
	//	sch = append(sch, table.Column{"CA3Cyc"+strconv.Itoa(iCyc), tensor.FLOAT64, nil, nil})
	//}
	dt.SetFromSchema(sch, nt)
}

//////////////////////////////////////////////
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *table.Table) {
	epc := ss.TrainEnv.Epoch.Cur
	trl := ss.TrainEnv.Trial.Cur

	row := dt.Rows
	if trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	dt.SetCellFloat("Mem", row, ss.Mem)
	dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffAll)
	dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)

	// note: essential to use Go version of update when called from another goroutine
	if ss.TrnTrlPlot != nil {
		ss.TrnTrlPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTrnTrlLog(dt *table.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := table.Schema{
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"Trial", tensor.INT64, nil, nil},
		{"TrialName", tensor.STRING, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
		{"Mem", tensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", tensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", tensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Hippocampus Train Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Trial", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("TrialName", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)

	plt.SetColParams("Mem", plot.On, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("TrgOnWasOff", plot.On, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("TrgOffWasOn", plot.On, plot.FixMin, 0, plot.FixMax, 1)

	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *table.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv           // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Table.Len()) // number of trials in view

	var spltparams []string
	if len(os.Args) > 1 {
		params := ss.RunName() // includes tag
		spltparams = strings.Split(params, "_")
	} else {
		spltparams = append(spltparams, "Default")
		spltparams = append(spltparams, strconv.Itoa(ss.Pat.ListSize))
	}

	ss.EpcSSE = ss.SumSSE / nt
	ss.SumSSE = 0
	ss.EpcAvgSSE = ss.SumAvgSSE / nt
	ss.SumAvgSSE = 0
	ss.EpcPctErr = float64(ss.CntErr) / nt
	ss.CntErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0

	trlog := ss.TrnTrlLog
	tix := table.NewIndexView(trlog)

	//dt.SetCellString("Params", row, params)
	dt.SetCellString("NetSize", row, spltparams[0])
	dt.SetCellString("ListSize", row, spltparams[1])
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	mem := stats.Mean(tix, "Mem")[0]
	dt.SetCellFloat("Mem", row, mem)
	dt.SetCellFloat("TrgOnWasOff", row, stats.Mean(tix, "TrgOnWasOff")[0])
	dt.SetCellFloat("TrgOffWasOn", row, stats.Mean(tix, "TrgOffWasOn")[0])

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Name+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TrnEpcPlot != nil {
		ss.TrnEpcPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *table.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		//{"Params", tensor.STRING, nil, nil},
		{"NetSize", tensor.STRING, nil, nil},
		{"ListSize", tensor.STRING, nil, nil},
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"PctErr", tensor.FLOAT64, nil, nil},
		{"PctCor", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
		{"Mem", tensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", tensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", tensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, table.Column{lnm + " ActAvg", tensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Hippocampus Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("PctErr", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("PctCor", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)

	plt.SetColParams("Mem", plot.On, plot.FixMin, 0, plot.FixMax, 1)         // default plot
	plt.SetColParams("TrgOnWasOff", plot.On, plot.FixMin, 0, plot.FixMax, 1) // default plot
	plt.SetColParams("TrgOffWasOn", plot.On, plot.FixMin, 0, plot.FixMax, 1) // default plot

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", plot.Off, plot.FixMin, 0, plot.FixMax, 0.5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *table.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TestEnv.Trial.Cur

	row := dt.Rows
	if ss.TestNm == "AB" && trl == 0 { // reset at start
		row = 0
	}
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("TestNm", row, ss.TestNm)
	dt.SetCellFloat("Trial", row, float64(row))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	dt.SetCellFloat("Mem", row, ss.Mem)
	dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffCmp)
	dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Name+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		tsr := ss.ValuesTsr(lnm)
		ly.UnitValuesTensor(tsr, "Act")
		dt.SetCellTensor(lnm+"Act", row, tsr)
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstTrlPlot != nil {
		ss.TstTrlPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstTrlLog(dt *table.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := table.Schema{
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"TestNm", tensor.STRING, nil, nil},
		{"Trial", tensor.INT64, nil, nil},
		{"TrialName", tensor.STRING, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
		{"Mem", tensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", tensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", tensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, table.Column{lnm + " ActM.Avg", tensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, table.Column{lnm + "Act", tensor.FLOAT64, ly.Shape.Sizes, nil})
	}

	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Hippocampus Test Trial Plot"
	plt.Params.XAxisCol = "TrialName"
	plt.Params.Type = plot.Bar
	plt.SetTable(dt) // this sets defaults so set params after
	plt.Params.XAxisRot = 45
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("TestNm", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Trial", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("TrialName", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)

	plt.SetColParams("Mem", plot.On, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("TrgOnWasOff", plot.On, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("TrgOffWasOn", plot.On, plot.FixMin, 0, plot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", plot.Off, plot.FixMin, 0, plot.FixMax, 0.5)
	}
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"Act", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	}

	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

// RepsAnalysis analyzes representations
func (ss *Sim) RepsAnalysis() {
	acts := table.NewIndexView(ss.TstTrlLog)
	for _, lnm := range ss.LayStatNms {
		sm, ok := ss.SimMats[lnm]
		if !ok {
			sm = &simat.SimMat{}
			ss.SimMats[lnm] = sm
		}
		sm.TableCol(acts, lnm+"Act", "TrialName", true, metric.Correlation64)
	}
}

// SimMatStatFull returns full triangular matrix for sim mat statistics
func (ss *Sim) SimMatStatFull(lnm string) *tensor.Float64 {
	sm := ss.SimMats[lnm]
	smat := sm.Mat
	nitm := smat.DimSize(0)
	ncat := nitm / len(ss.TstNms) // i.e., list size
	newTsr := tensor.NewFloat64([]int{2 * ncat, 2 * ncat}, nil, []string{"Y", "X"})

	for y := 0; y < nitm*2/3; y++ { // only taking AB and AC, not Lure
		newTsr.SubSpace([]int{y}).CopyFrom(smat.SubSpace([]int{y}))
	}
	return newTsr
}

// SimMatStat returns within, between for sim mat statistics
func (ss *Sim) SimMatStat(lnm string) (float64, float64, float64) {
	sm := ss.SimMats[lnm]
	smat := sm.Mat
	nitm := smat.DimSize(0)
	ncat := nitm / len(ss.TstNms) // i.e., list size
	win_sum_ab := float64(0)
	win_n_ab := 0
	win_sum_ac := float64(0)
	win_n_ac := 0
	btn_sum := float64(0)
	btn_n := 0
	for y := 0; y < nitm*2/3; y++ { // only taking AB and AC, not Lure
		for x := 0; x < y; x++ {
			val := smat.Float([]int{y, x})
			same := (y / ncat) == (x / ncat) // i.e., same list or not
			if same {
				if y < nitm/3 {
					win_sum_ab += val
					win_n_ab++
				} else {
					win_sum_ac += val
					win_n_ac++
				}
			} else if (y % ncat) == (x % ncat) { // between list, only when same A (i.e., TrainAB11 vs. Train AC11)!
				btn_sum += val
				btn_n++
			}
		}
	}
	if win_n_ab > 0 {
		win_sum_ab /= float64(win_n_ab)
	}
	if win_n_ac > 0 {
		win_sum_ac /= float64(win_n_ac)
	}
	if btn_n > 0 {
		btn_sum /= float64(btn_n)
	}
	return win_sum_ab, win_sum_ac, btn_sum
}

func (ss *Sim) LogTstEpc(dt *table.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	ss.RepsAnalysis()

	trl := ss.TstTrlLog
	tix := table.NewIndexView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	var spltparams []string
	if len(os.Args) > 1 {
		params := ss.RunName() // includes tag
		spltparams = strings.Split(params, "_")
	} else {
		spltparams = append(spltparams, "Default")
		spltparams = append(spltparams, strconv.Itoa(ss.Pat.ListSize))
	}

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		nt := ss.TrainAB.Rows * 4 // 1 train and 3 tests
		ss.EpcPerTrlMSec = float64(iv) / (float64(nt) * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	//dt.SetCellString("Params", row, params)
	dt.SetCellString("NetSize", row, spltparams[0])
	dt.SetCellString("ListSize", row, spltparams[1])
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)
	dt.SetCellFloat("SSE", row, stats.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, stats.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, stats.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val > 0
	})[0])
	dt.SetCellFloat("PctCor", row, stats.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val == 0
	})[0])
	dt.SetCellFloat("CosDiff", row, stats.Mean(tix, "CosDiff")[0])

	trix := table.NewIndexView(trl)
	spl := split.GroupBy(trix, []string{"TestNm"})
	for _, ts := range ss.TstStatNms {
		split.Agg(spl, ts, stats.AggMean)
	}
	ss.TstStats = spl.AggsToTable(table.ColNameOnly)

	for ri := 0; ri < ss.TstStats.Rows; ri++ {
		tst := ss.TstStats.CellString("TestNm", ri)
		for _, ts := range ss.TstStatNms {
			dt.SetCellFloat(tst+" "+ts, row, ss.TstStats.CellFloat(ts, ri))
		}
	}

	for _, lnm := range ss.LayStatNms {
		win_ab, win_ac, btn := ss.SimMatStat(lnm)
		for _, ts := range ss.SimMatStats {
			if ts == "WithinAB" {
				dt.SetCellFloat(lnm+" "+ts, row, win_ab)
			} else if ts == "WithinAC" {
				dt.SetCellFloat(lnm+" "+ts, row, win_ac)
			} else {
				dt.SetCellFloat(lnm+" "+ts, row, btn)
			}
		}
	}
	// RS Matrix
	//for _, lnm := range ss.LayStatNms {
	//	rsm := ss.SimMatStatFull(lnm)
	//	dt.SetCellTensor(lnm+" RSM", row, rsm)
	//}

	// base zero on testing performance!
	curAB := ss.TrainEnv.Table.Table == ss.TrainAB
	var mem float64
	if curAB {
		mem = dt.CellFloat("AB Mem", row)
	} else {
		mem = dt.CellFloat("AC Mem", row)
	}
	if ss.FirstZero < 0 && mem == 1 {
		ss.FirstZero = epc
	}
	if mem == 1 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstEpcPlot != nil {
		ss.TstEpcPlot.GoUpdate()
	}
	if ss.TstEpcFile != nil {
		if !ss.TstEpcHdrs {
			dt.WriteCSVHeaders(ss.TstEpcFile, table.Tab)
			ss.TstEpcHdrs = true
		}
		dt.WriteCSVRow(ss.TstEpcFile, row, table.Tab)
	}
}

func (ss *Sim) ConfigTstEpcLog(dt *table.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		//{"Params", tensor.STRING, nil, nil},
		{"NetSize", tensor.STRING, nil, nil},
		{"ListSize", tensor.STRING, nil, nil},
		{"Run", tensor.INT64, nil, nil},
		{"Epoch", tensor.INT64, nil, nil},
		{"PerTrlMSec", tensor.FLOAT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"PctErr", tensor.FLOAT64, nil, nil},
		{"PctCor", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, table.Column{tn + " " + ts, tensor.FLOAT64, nil, nil})
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			sch = append(sch, table.Column{lnm + " " + ts, tensor.FLOAT64, nil, nil})
		}
	}
	// RS Matrix
	//for _, lnm := range ss.LayStatNms {
	//	ncat := ss.Pat.ListSize
	//	sch = append(sch, table.Column{lnm + " RSM", tensor.FLOAT64, []int{2 * ncat, 2 * ncat}, nil})
	//}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Hippocampus Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt) // this sets defaults so set params after
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Epoch", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("PerTrlMSec", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("PctErr", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("PctCor", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			if ts == "Mem" {
				plt.SetColParams(tn+" "+ts, plot.On, plot.FixMin, 0, plot.FixMax, 1)
			} else {
				plt.SetColParams(tn+" "+ts, plot.Off, plot.FixMin, 0, plot.FixMax, 1)
			}
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			plt.SetColParams(lnm+" "+ts, plot.Off, plot.FixMin, 0, plot.FloatMax, 1)
		}
	}
	// RS Matrix
	//for _, lnm := range ss.LayStatNms {
	//	plt.SetColParams(lnm+" RSM", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	//}
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
		if ss.TstCycPlot != nil {
			ss.TstCycPlot.GoUpdate()
		}
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
	plt.Params.Title = "Hippocampus Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", plot.On, plot.FixMin, 0, plot.FixMax, .5)
		plt.SetColParams(lnm+" Act.Avg", plot.On, plot.FixMin, 0, plot.FixMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *table.Table) {
	epclog := ss.TstEpcLog
	epcix := table.NewIndexView(epclog)
	if epcix.Len() == 0 {
		return
	}

	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Indexes = epcix.Indexes[epcix.Len()-nlast:]

	var spltparams []string
	if len(os.Args) > 1 {
		params := ss.RunName() // includes tag
		spltparams = strings.Split(params, "_")
	} else {
		spltparams = append(spltparams, "Default")
		spltparams = append(spltparams, strconv.Itoa(ss.Pat.ListSize))
	}

	fzero := ss.FirstZero
	if fzero < 0 {
		fzero = ss.MaxEpcs
	}

	//dt.SetCellString("Params", row, params)
	dt.SetCellString("NetSize", row, spltparams[0])
	dt.SetCellString("ListSize", row, spltparams[1])
	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellFloat("NEpochs", row, float64(ss.TstEpcLog.Rows))
	dt.SetCellFloat("FirstZero", row, float64(fzero))
	dt.SetCellFloat("SSE", row, stats.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, stats.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, stats.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, stats.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, stats.Mean(epcix, "CosDiff")[0])

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			nm := tn + " " + ts
			dt.SetCellFloat(nm, row, stats.Mean(epcix, nm)[0])
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			nm := lnm + " " + ts
			dt.SetCellFloat(nm, row, stats.Mean(epcix, nm)[0])
		}
	}
	ss.LogRunStats()

	// note: essential to use Go version of update when called from another goroutine
	if ss.RunPlot != nil {
		ss.RunPlot.GoUpdate()
	}
	if ss.RunFile != nil {
		if !ss.RunHdrs {
			dt.WriteCSVHeaders(ss.RunFile, table.Tab)
			ss.RunHdrs = true
		}
		dt.WriteCSVRow(ss.RunFile, row, table.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *table.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := table.Schema{
		//{"Params", tensor.STRING, nil, nil},
		{"NetSize", tensor.STRING, nil, nil},
		{"ListSize", tensor.STRING, nil, nil},
		{"Run", tensor.INT64, nil, nil},
		{"NEpochs", tensor.FLOAT64, nil, nil},
		{"FirstZero", tensor.FLOAT64, nil, nil},
		{"SSE", tensor.FLOAT64, nil, nil},
		{"AvgSSE", tensor.FLOAT64, nil, nil},
		{"PctErr", tensor.FLOAT64, nil, nil},
		{"PctCor", tensor.FLOAT64, nil, nil},
		{"CosDiff", tensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, table.Column{tn + " " + ts, tensor.FLOAT64, nil, nil})
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			sch = append(sch, table.Column{lnm + " " + ts, tensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *plot.Plot2D, dt *table.Table) *plot.Plot2D {
	plt.Params.Title = "Hippocampus Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("NetSize", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("ListSize", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("Run", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("NEpochs", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("FirstZero", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("SSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("AvgSSE", plot.Off, plot.FixMin, 0, plot.FloatMax, 0)
	plt.SetColParams("PctErr", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("PctCor", plot.Off, plot.FixMin, 0, plot.FixMax, 1)
	plt.SetColParams("CosDiff", plot.Off, plot.FixMin, 0, plot.FixMax, 1)

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			if ts == "Mem" {
				plt.SetColParams(tn+" "+ts, plot.On, plot.FixMin, 0, plot.FixMax, 1) // default plot
			} else {
				plt.SetColParams(tn+" "+ts, plot.Off, plot.FixMin, 0, plot.FixMax, 1)
			}
		}
	}
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			plt.SetColParams(lnm+" "+ts, plot.Off, plot.FixMin, 0, plot.FloatMax, 1)
		}
	}
	return plt
}

//////////////////////////////////////////////
//  RunStats

// LogRunStats computes RunStats from RunLog data -- can be used for looking at prelim results
func (ss *Sim) LogRunStats() {
	dt := ss.RunLog
	runix := table.NewIndexView(dt)
	//spl := split.GroupBy(runix, []string{"Params"})
	spl := split.GroupBy(runix, []string{"NetSize", "ListSize"})
	//spl := split.GroupBy(runix, []string{"NetSize", "ListSize", "Condition"})
	for _, tn := range ss.TstNms {
		nm := tn + " " + "Mem"
		split.Desc(spl, nm)
	}
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "NEpochs")
	for _, lnm := range ss.LayStatNms {
		for _, ts := range ss.SimMatStats {
			split.Desc(spl, lnm+" "+ts)
		}
	}
	ss.RunStats = spl.AggsToTable(table.AddAggName)
	if ss.RunStatsPlot1 != nil {
		ss.ConfigRunStatsPlot(ss.RunStatsPlot1, ss.RunStats, 1)
	}
	if ss.RunStatsPlot2 != nil {
		ss.ConfigRunStatsPlot(ss.RunStatsPlot2, ss.RunStats, 2)
	}
}

func (ss *Sim) ConfigRunStatsPlot(plt *plot.Plot2D, dt *table.Table, plotidx int) *plot.Plot2D {
	plt.Params.Title = "Comparison between Hippocampus Models"
	//plt.Params.XAxisCol = "Params"
	plt.Params.XAxisCol = "ListSize"
	plt.Params.LegendCol = "NetSize"
	//plt.Params.LegendCol = "Condition"
	plt.SetTable(dt)

	//plt.Params.BarWidth = 10
	//plt.Params.Type = plot.Bar
	plt.Params.LineWidth = 1
	plt.Params.Scale = 2
	plt.Params.Type = plot.XY
	plt.Params.XAxisRot = 45

	if plotidx == 1 {
		cp := plt.SetColParams("AB Mem:Mean", plot.On, plot.FixMin, 0, plot.FixMax, 1) // interference
		cp.ErrCol = "AB Mem:Sem"
		plt.Params.YAxisLabel = "AB Memory"
	} else if plotidx == 2 {
		cp := plt.SetColParams("NEpochs:Mean", plot.On, plot.FixMin, 0, plot.FixMax, 30) // total learning time
		cp.ErrCol = "NEpochs:Sem"
		plt.Params.YAxisLabel = "Learning Time"
	}

	//cp = plt.SetColParams("AC Mem:Mean", plot.On, plot.FixMin, 0, plot.FixMax, 1)
	//cp.ErrCol = "AC Mem:Sem"
	//cp = plt.SetColParams("FirstZero:Mean", plot.On, plot.FixMin, 0, plot.FixMax, 30)
	//cp.ErrCol = "FirstZero:Sem"

	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() *core.Window {
	width := 1600
	height := 1200

	core.SetAppName("hip_bench")
	core.SetAppAbout(`This demonstrates a basic Hippocampus model in Leabra. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := core.NewMainWindow("hip_bench", "Hippocampus AB-AC", width, height)
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
	nv.ViewDefaults()

	plt := tv.AddNewTab(plot.KiT_Plot2D, "TrnTrlPlot").(*plot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TrnEpcPlot").(*plot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TstTrlPlot").(*plot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TstEpcPlot").(*plot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "TstCycPlot").(*plot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "RunPlot").(*plot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	plt = tv.AddNewTab(plot.KiT_Plot2D, "RunStatsPlot1").(*plot.Plot2D)
	ss.RunStatsPlot1 = plt

	plt = tv.AddNewTab(plot.KiT_Plot2D, "RunStatsPlot2").(*plot.Plot2D)
	ss.RunStatsPlot2 = plt

	split.SetSplits(.2, .8)

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

	tbar.AddAction(core.ActOpts{Label: "Pre Train", Icon: "fast-fwd", Tooltip: "Does full pretraining.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.PreTrain()
		}
	})

	tbar.AddAction(core.ActOpts{Label: "New Run", Icon: "reset", Tooltip: "After PreTrain, init things and reload the pretrain weights"}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			ss.NewRun()
		})

	tbar.AddSeparator("test")

	tbar.AddAction(core.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *core.Action) {
		act.SetActiveStateUpdate(!ss.IsRunning)
	}}, win.This(), func(recv, send tree.Node, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on trial -- wrap
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

	tbar.AddAction(core.ActOpts{Label: "Env", Icon: "gear", Tooltip: "select training input patterns: AB or AC."}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			views.CallMethod(ss, "SetEnv", vp)
		})

	tbar.AddSeparator("log")

	tbar.AddAction(core.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddAction(core.ActOpts{Label: "Rebuild Net", Icon: "reset", Tooltip: "Rebuild network with current params"}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			ss.ReConfigNet()
		})

	tbar.AddAction(core.ActOpts{Label: "Run Stats", Icon: "file-data", Tooltip: "compute stats from run log -- avail in plot"}, win.This(),
		func(recv, send tree.Node, sig int64, data interface{}) {
			ss.LogRunStats()
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
		{"SetEnv", tree.Props{
			"desc": "select which set of patterns to train on: AB or AC",
			"icon": "gear",
			"Args": tree.PropSlice{
				{"Train on AC", tree.Props{}},
			},
		}},
	},
}

// zycyc
// OuterLoopParams are the parameters to run for outer crossed factor testing
//var OuterLoopParams = []string{"SmallHip"}

var OuterLoopParams = []string{"BigHip"}

//var OuterLoopParams = []string{"SmallHip", "MedHip", "BigHip"}

// InnerLoopParams are the parameters to run for inner crossed factor testing
//var InnerLoopParams = []string{"List020", "List040"}

//var InnerLoopParams = []string{"List150", "List175", "List200"}

var InnerLoopParams = []string{"List100", "List125", "List150", "List175", "List200"}

//var InnerLoopParams = []string{"List020", "List040", "List060", "List080", "List100"}

// TwoFactorRun runs outer-loop crossed with inner-loop params
func (ss *Sim) TwoFactorRun() {
	tag := ss.Tag
	usetag := tag
	if usetag != "" {
		usetag += "_"
	}
	for _, otf := range OuterLoopParams {
		for _, inf := range InnerLoopParams {
			ss.Tag = usetag + otf + "_" + inf
			rand.Seed(ss.RndSeed + int64(ss.BatchRun)) // TODO: non-parallel running should resemble parallel running results, now not
			ss.SetParamsSet(otf, "", ss.LogSetParams)
			ss.SetParamsSet(inf, "", ss.LogSetParams)
			ss.ReConfigNet() // note: this applies Base params to Network
			ss.ConfigEnv()
			ss.StopNow = false
			ss.PretrainDone = false
			ss.PreTrain() // zycyc, NoPretrain key
			ss.PretrainDone = true
			ss.NewRun()
			ss.Train()
		}
	}
	ss.Tag = tag
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveCycPatSimLog bool
	var saveEpcLog bool
	var saveRunLog bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.BatchRun, "run", 0, "current batch run")
	flag.IntVar(&ss.MaxRuns, "runs", 1, "number of runs to do, i.e., subjects")
	flag.IntVar(&ss.MaxEpcs, "epcs", 30, "maximum number of epochs to run (split between AB / AC)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWeights, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveCycPatSimLog, "cycpatsimlog", false, "if true, save train cycle similarity log to file") // zycyc, pat sim key
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save test epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName(strconv.Itoa(ss.BatchRun) + "tstepc")
		ss.TstEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstEpcFile = nil
		} else {
			fmt.Printf("Saving test epoch log to: %v\n", fnm)
			defer ss.TstEpcFile.Close()
		}
	}
	if saveCycPatSimLog {
		var err error
		fnm := ss.LogFileName(strconv.Itoa(ss.BatchRun) + "trncycpatsim")
		ss.TrnCycPatSimFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnCycPatSimFile = nil
		} else {
			fmt.Printf("Saving train cycle pattern similarity log to: %v\n", fnm)
			defer ss.TrnCycPatSimFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName(strconv.Itoa(ss.BatchRun) + "run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if ss.SaveWeights {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Batch No. %d\n", ss.BatchRun)
	fmt.Printf("Running %d Runs\n", ss.MaxRuns-ss.BatchRun)
	// ss.Train()
	ss.TwoFactorRun()
	//fnm := ss.LogFileName("runs")
	//ss.RunStats.SaveCSV(core.Filename(fnm), table.Tab, table.Headers) // not usable for batch runs
}
