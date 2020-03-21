// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
	"time"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/emer/leabra/hip"
	"github.com/emer/leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// see def_params.go for the default params, and params.go for user-saved versions
// from the gui.

// see bottom of file for multi-factor testing params

// HipParams have the hippocampus size and connectivity parameters
type HipParams struct {
	ECSize       evec.Vec2i `desc:"size of EC in terms of overall pools (outer dimension)"`
	ECPool       evec.Vec2i `desc:"size of one EC pool"`
	CA1Pool      evec.Vec2i `desc:"size of one CA1 pool"`
	CA3Size      evec.Vec2i `desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	DGSize       evec.Vec2i `inactive:"+" desc:"size of DG"`
	DGPCon       float32    `desc:"percent connectivity into DG"`
	CA3PCon      float32    `desc:"percent connectivity into CA3"`
	MossyPCon    float32    `desc:"percent connectivity into CA3 from DG"`
	ECPctAct     float32    `desc:"percent activation in EC pool"`
	MossyDel     float32    `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32    `desc:"delta in mossy strength for testing (relative to base param)"`
}

func (hp *HipParams) Update() {
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)
}

// PatParams have the pattern parameters
type PatParams struct {
	ListSize    int     `desc:"number of A-B, A-C patterns each"`
	MinDiffPct  float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`
	DriftCtxt   bool    `desc:"use drifting context representations -- otherwise does bit flips from prototype"`
	CtxtFlipPct float32 `desc:"proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting"`
	DriftPct    float32 `desc:"percentage of active bits that drift, per step, for drifting context"`
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *leabra.Network             `view:"no-inline"`
	Hip          HipParams                   `desc:"hippocampus sizing parameters"`
	Pat          PatParams                   `desc:"parameters for the input patterns"`
	PoolVocab    map[string]*etensor.Float32 `view:"no-inline" desc:"pool patterns vocabulary"`
	TrainAB      *etable.Table               `view:"no-inline" desc:"AB training patterns to use"`
	TrainAC      *etable.Table               `view:"no-inline" desc:"AC training patterns to use"`
	TestAB       *etable.Table               `view:"no-inline" desc:"AB testing patterns to use"`
	TestAC       *etable.Table               `view:"no-inline" desc:"AC testing patterns to use"`
	TestLure     *etable.Table               `view:"no-inline" desc:"Lure testing patterns to use"`
	TrainAll     *etable.Table               `view:"no-inline" desc:"all training patterns -- for pretrain"`
	TrnTrlLog    *etable.Table               `view:"no-inline" desc:"training trial-level log data"`
	TrnEpcLog    *etable.Table               `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table               `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog    *etable.Table               `view:"no-inline" desc:"testing trial-level log data"`
	TstCycLog    *etable.Table               `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog       *etable.Table               `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table               `view:"no-inline" desc:"aggregate stats on all runs"`
	TstStats     *etable.Table               `view:"no-inline" desc:"testing stats"`
	Params       params.Sets                 `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string                      `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	Tag          string                      `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params)"`
	MaxRuns      int                         `desc:"maximum number of model runs to perform"`
	MaxEpcs      int                         `desc:"maximum number of epochs to run per model run"`
	PreTrainEpcs int                         `desc:"number of epochs to run for pretraining"`
	NZeroStop    int                         `desc:"if a positive number, training will stop after this many epochs with zero mem errors"`
	TrainEnv     env.FixedTable              `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable              `desc:"Testing environment -- manages iterating over testing"`
	Time         leabra.Time                 `desc:"leabra timing parameters and state"`
	ViewOn       bool                        `desc:"whether to update the network view while running"`
	TrainUpdt    leabra.TimeScales           `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     leabra.TimeScales           `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int                         `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	MemThr       float64                     `desc:"threshold to use for memory test -- if error proportion is below this number, it is scored as a correct trial"`

	// statistics: note use float64 as that is best for etable.Table
	TestNm         string  `inactive:"+" desc:"what set of patterns are we currently testing"`
	Mem            float64 `inactive:"+" desc:"whether current trial's ECout met memory criterion"`
	TrgOnWasOffAll float64 `inactive:"+" desc:"current trial's proportion of bits where target = on but ECout was off ( < 0.5), for all bits"`
	TrgOnWasOffCmp float64 `inactive:"+" desc:"current trial's proportion of bits where target = on but ECout was off ( < 0.5), for only completion bits that were not active in ECin"`
	TrgOffWasOn    float64 `inactive:"+" desc:"current trial's proportion of bits where target = off but ECout was on ( > 0.5)"`
	TrlSSE         float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE      float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff     float64 `inactive:"+" desc:"current trial's cosine difference"`

	EpcSSE        float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE     float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance)"`
	EpcPctCor     float64 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when Mem err first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with zero Mem err"`

	// internal state - view:"-"
	SumSSE       float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64          `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	CntErr       int              `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	Win          *gi.Window       `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar      `view:"-" desc:"the master toolbar"`
	TrnTrlPlot   *eplot.Plot2D    `view:"-" desc:"the training trial plot"`
	TrnEpcPlot   *eplot.Plot2D    `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D    `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D    `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D    `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D    `view:"-" desc:"the run plot"`
	RunStatsPlot *eplot.Plot2D    `view:"-" desc:"the run stats plot"`
	TrnEpcFile   *os.File         `view:"-" desc:"log file"`
	TrnEpcHdrs   bool             `view:"-" desc:"headers written"`
	TstEpcFile   *os.File         `view:"-" desc:"log file"`
	TstEpcHdrs   bool             `view:"-" desc:"headers written"`
	RunFile      *os.File         `view:"-" desc:"log file"`
	TmpVals      []float32        `view:"-" desc:"temp slice for holding values -- prevent mem allocs"`
	LayStatNms   []string         `view:"-" desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	TstNms       []string         `view:"-" desc:"names of test tables"`
	TstStatNms   []string         `view:"-" desc:"names of test stats"`
	SaveWts      bool             `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	PreTrainWts  []byte           `view:"-" desc:"pretrained weights file"`
	NoGui        bool             `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool             `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool             `view:"-" desc:"true if sim is running"`
	StopNow      bool             `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool             `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeed      int64            `view:"-" desc:"the current random seed"`
	LastEpcTime  time.Time        `view:"-" desc:"timer for last epoch"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.PoolVocab = map[string]*etensor.Float32{}
	ss.TrainAB = &etable.Table{}
	ss.TrainAC = &etable.Table{}
	ss.TestAB = &etable.Table{}
	ss.TestAC = &etable.Table{}
	ss.TestLure = &etable.Table{}
	ss.TrainAll = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets // in def_params -- current best params
	// ss.Params = OrigParamSets // original, previous model
	// ss.Params = SavedParamsSets // current user-saved gui params
	ss.RndSeed = 2
	ss.ViewOn = true
	ss.TrainUpdt = leabra.AlphaCycle
	ss.TestUpdt = leabra.AlphaCycle
	ss.TestInterval = 1
	ss.LogSetParams = false
	ss.MemThr = 0.34
	ss.LayStatNms = []string{"ECin", "DG", "CA3", "CA1"}
	ss.TstNms = []string{"AB", "AC", "Lure"}
	ss.TstStatNms = []string{"Mem", "TrgOnWasOff", "TrgOffWasOn"}

	ss.Defaults()
}

func (pp *PatParams) Defaults() {
	pp.ListSize = 20 // 10 is too small to see issues..
	pp.MinDiffPct = 0.5
	pp.CtxtFlipPct = .25
	pp.DriftPct = .2
}

func (hp *HipParams) Defaults() {
	// size
	hp.ECSize.Set(2, 3)
	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(10, 10)
	hp.CA3Size.Set(20, 20)
	hp.DGRatio = 1.5

	// ratio
	hp.DGPCon = 0.25 // .35 is sig worse, .2 learns faster but AB recall is worse
	hp.CA3PCon = 0.25
	hp.MossyPCon = 0.02 // .02 > .05 > .01 (for small net)
	hp.ECPctAct = 0.2

	hp.MossyDel = 4     // 4 > 2 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 3 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount..
}

func (ss *Sim) Defaults() {
	ss.Hip.Defaults()
	ss.Pat.Defaults()
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
		ss.PreTrainEpcs = 5 // seems sufficient?
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.TestAB)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

// SetEnv select which set of patterns to train on: AB or AC
func (ss *Sim) SetEnv(trainAC bool) {
	if trainAC {
		ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAC)
	} else {
		ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	}
	ss.TrainEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "Hip_bench")
	hp := &ss.Hip
	in := net.AddLayer4D("Input", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Input)
	ecin := net.AddLayer4D("ECin", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Hidden)
	ecout := net.AddLayer4D("ECout", hp.ECSize.Y, hp.ECSize.X, hp.ECPool.Y, hp.ECPool.X, emer.Target) // clamped in plus phase
	ca1 := net.AddLayer4D("CA1", hp.ECSize.Y, hp.ECSize.X, hp.CA1Pool.Y, hp.CA1Pool.X, emer.Hidden)
	dg := net.AddLayer2D("DG", hp.DGSize.Y, hp.DGSize.X, emer.Hidden)
	ca3 := net.AddLayer2D("CA3", hp.CA3Size.Y, hp.CA3Size.X, emer.Hidden)

	ecin.SetClass("EC")
	ecout.SetClass("EC")

	ecin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Input", YAlign: relpos.Front, Space: 2})
	ecout.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ECin", YAlign: relpos.Front, Space: 2})
	dg.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Input", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca3.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "DG", YAlign: relpos.Front, XAlign: relpos.Left, Space: 0})
	ca1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "CA3", YAlign: relpos.Front, Space: 2})

	onetoone := prjn.NewOneToOne()
	pool1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	net.ConnectLayers(in, ecin, onetoone, emer.Forward)
	net.ConnectLayers(ecout, ecin, onetoone, emer.Back)

	// EC <-> CA1 encoder pathways
	pj := net.ConnectLayersPrjn(ecin, ca1, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ca1, ecout, pool1to1, emer.Forward, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")
	pj = net.ConnectLayersPrjn(ecout, ca1, pool1to1, emer.Back, &hip.EcCa1Prjn{})
	pj.SetClass("EcCa1Prjn")

	// Perforant pathway
	ppathDG := prjn.NewUnifRnd()
	ppathDG.PCon = hp.DGPCon
	ppathCA3 := prjn.NewUnifRnd()
	ppathCA3.PCon = hp.CA3PCon

	pj = net.ConnectLayersPrjn(ecin, dg, ppathDG, emer.Forward, &hip.CHLPrjn{})
	pj.SetClass("HippoCHL")

	if true { // toggle for bcm vs. ppath
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.EcCa1Prjn{})
		pj.SetClass("PPath")
	} else {
		// so far, this is sig worse, even with error-driven MinusQ1 case (which is better than off)
		pj = net.ConnectLayersPrjn(ecin, ca3, ppathCA3, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("PPath")
		pj = net.ConnectLayersPrjn(ca3, ca3, full, emer.Lateral, &hip.CHLPrjn{})
		pj.SetClass("PPath")
	}

	// always use this for now:
	if true {
		pj = net.ConnectLayersPrjn(ca3, ca1, full, emer.Forward, &hip.CHLPrjn{})
		pj.SetClass("HippoCHL")
	} else {
		// note: this requires lrate = 1.0 or maybe 1.2, doesn't work *nearly* as well
		pj = net.ConnectLayers(ca3, ca1, full, emer.Forward) // default con
		// pj.SetClass("HippoCHL")
	}

	// Mossy fibers
	mossy := prjn.NewUnifRnd()
	mossy.PCon = hp.MossyPCon
	pj = net.ConnectLayersPrjn(dg, ca3, mossy, emer.Forward, &hip.CHLPrjn{}) // no learning
	pj.SetClass("HippoCHL")

	// using 4 threads total (rest on 0)
	dg.SetThread(1)
	ca3.SetThread(2)
	ca1.SetThread(3) // this has the most

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
	net.InitWts()
}

func (ss *Sim) ReConfigNet() {
	ss.ConfigPats()
	ss.Net = &leabra.Network{} // start over with new network
	ss.ConfigNet(ss.Net)
	if ss.NetView != nil {
		ss.NetView.SetNet(ss.Net)
		ss.NetView.Update() // issue #41 closed
	}
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
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
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
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}

	ca1 := ss.Net.LayerByName("CA1").(leabra.LeabraLayer).AsLeabra()
	ca3 := ss.Net.LayerByName("CA3").(leabra.LeabraLayer).AsLeabra()
	input := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	ecin := ss.Net.LayerByName("ECin").(leabra.LeabraLayer).AsLeabra()
	ecout := ss.Net.LayerByName("ECout").(leabra.LeabraLayer).AsLeabra()
	ca1FmECin := ca1.RcvPrjns.SendName("ECin").(leabra.LeabraPrjn).AsLeabra()
	ca1FmCa3 := ca1.RcvPrjns.SendName("CA3").(leabra.LeabraPrjn).AsLeabra()
	ca3FmDg := ca3.RcvPrjns.SendName("DG").(leabra.LeabraPrjn).AsLeabra()
	_ = ecin
	_ = input

	// First Quarter: CA1 is driven by ECin, not by CA3 recall
	// (which is not really active yet anyway)
	ca1FmECin.WtScale.Abs = 1
	ca1FmCa3.WtScale.Abs = 0

	dgwtscale := ca3FmDg.WtScale.Rel
	ca3FmDg.WtScale.Rel = dgwtscale - ss.Hip.MossyDel

	if train {
		ecout.SetType(emer.Target) // clamp a plus phase during testing
	} else {
		ecout.SetType(emer.Compare) // don't clamp
	}
	ecout.UpdateExtFlags() // call this after updating type

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if !train {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
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
			ca1FmECin.WtScale.Abs = 0
			ca1FmCa3.WtScale.Abs = 1
			if train {
				ca3FmDg.WtScale.Rel = dgwtscale
			} else {
				ca3FmDg.WtScale.Rel = dgwtscale - ss.Hip.MossyDelTest // testing
			}
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
		case 3: // Fourth Quarter: CA1 back to ECin drive only
			ca1FmECin.WtScale.Abs = 1
			ca1FmCa3.WtScale.Abs = 0
			ss.Net.GScaleFmAvgAct() // update computed scaling factors
			ss.Net.InitGInc()       // scaling params change, so need to recompute all netins
			if train {              // clamp ECout from ECin
				ecin.UnitVals(&ss.TmpVals, "Act") // note: could use input instead -- not much diff
				ecout.ApplyExt1D32(ss.TmpVals)
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		if qtr+1 == 3 {
			ss.MemStats(train) // must come after QuarterFinal
		}
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
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
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
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
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "ECout"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := en.State(ly.Nm)
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
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		learned := (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop)
		if ss.TrainEnv.Table.Table == ss.TrainAB && (learned || epc == ss.MaxEpcs/2) {
			ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAC)
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
}

// PreTrainTrial runs one trial of pretraining using TrainEnv
func (ss *Sim) PreTrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
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
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.LoadPretrainedWts()
	ss.InitStats()
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
	nn := ecout.Shape().Len()
	trgOnWasOffAll := 0.0 // all units
	trgOnWasOffCmp := 0.0 // only those that required completion, missing in ECin
	trgOffWasOn := 0.0    // should have been off
	cmpN := 0.0           // completion target
	trgOnN := 0.0
	trgOffN := 0.0
	for ni := 0; ni < nn; ni++ {
		actm := ecout.UnitVal1D("ActM", ni)
		trg := ecout.UnitVal1D("Targ", ni) // full pattern target
		inact := ecin.UnitVal1D("ActQ1", ni)
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

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
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
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAll)
	// todo: pretrain on all patterns!
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.PreTrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	b := &bytes.Buffer{}
	ss.Net.WriteWtsJSON(b)
	ss.PreTrainWts = b.Bytes()
	ss.TrainEnv.Table = etable.NewIdxView(ss.TrainAB)
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
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
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
	ss.TestEnv.Table = etable.NewIdxView(ss.TestAB)
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
		ss.TestEnv.Table = etable.NewIdxView(ss.TestAC)
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
			ss.TestEnv.Table = etable.NewIdxView(ss.TestLure)
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

func (ss *Sim) OpenPat(dt *etable.Table, fname, name, desc string) {
	err := dt.OpenCSV(gi.FileName(fname), etable.Tab)
	if err != nil {
		log.Println(err)
		return
	}
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", desc)
}

func (ss *Sim) ConfigPats() {
	hp := &ss.Hip
	plY := hp.ECPool.Y // good idea to get shorter vars when used frequently
	plX := hp.ECPool.X // makes much more readable
	npats := ss.Pat.ListSize
	pctAct := hp.ECPctAct
	minDiff := ss.Pat.MinDiffPct
	nOn := patgen.NFmPct(pctAct, plY*plX)
	ctxtflip := patgen.NFmPct(ss.Pat.CtxtFlipPct, nOn)
	patgen.AddVocabEmpty(ss.PoolVocab, "empty", npats, plY, plX)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "A", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "B", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "C", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lA", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "lB", npats, plY, plX, pctAct, minDiff)
	patgen.AddVocabPermutedBinary(ss.PoolVocab, "ctxt", 3, plY, plX, pctAct, minDiff) // totally diff

	for i := 0; i < 12; i++ { // 12 contexts!
		list := i / 4
		ctxtNm := fmt.Sprintf("ctxt%d", i+1)
		tsr, _ := patgen.AddVocabRepeat(ss.PoolVocab, ctxtNm, npats, "ctxt", list)
		patgen.FlipBitsRows(tsr, ctxtflip, ctxtflip, 1, 0)
		// todo: also support drifting
		// solution 2: drift based on last trial (will require sequential learning)
		// patgen.VocabDrift(ss.PoolVocab, ss.NFlipBits, "ctxt"+strconv.Itoa(i+1))
	}

	ecY := hp.ECSize.Y
	ecX := hp.ECSize.X

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

	patgen.InitPats(ss.TestLure, "TestLure", "TestLure Pats", "Input", "ECout", npats, ecY, ecX, plY, plX)
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "Input", []string{"lA", "empty", "ctxt9", "ctxt10", "ctxt11", "ctxt12"}) // arbitrary ctxt here
	patgen.MixPats(ss.TestLure, ss.PoolVocab, "ECout", []string{"lA", "lB", "ctxt9", "ctxt10", "ctxt11", "ctxt12"})    // arbitrary ctxt here

	ss.TrainAll = ss.TrainAB.Clone()
	ss.TrainAll.AppendRows(ss.TrainAC)
	ss.TrainAll.AppendRows(ss.TestLure)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

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
//  TrnTrlLog

// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnTrl(dt *etable.Table) {
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
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("SSE", row, ss.TrlSSE)
	dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	dt.SetCellFloat("Mem", row, ss.Mem)
	dt.SetCellFloat("TrgOnWasOff", row, ss.TrgOnWasOffAll)
	dt.SetCellFloat("TrgOffWasOn", row, ss.TrgOffWasOn)

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTrnTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Train Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	plt.SetColParams("Mem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOnWasOff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOffWasOn", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	return plt
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

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

	trlog := ss.TrnTrlLog
	tix := etable.NewIdxView(trlog)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("SSE", row, ss.EpcSSE)
	dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)

	mem := agg.Mean(tix, "Mem")[0]
	dt.SetCellFloat("Mem", row, mem)
	dt.SetCellFloat("TrgOnWasOff", row, agg.Mean(tix, "TrgOnWasOff")[0])
	dt.SetCellFloat("TrgOffWasOn", row, agg.Mean(tix, "TrgOffWasOn")[0])

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if !ss.TrnEpcHdrs {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
			ss.TrnEpcHdrs = true
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	plt.SetColParams("Mem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)         // default plot
	plt.SetColParams("TrgOnWasOff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("TrgOffWasOn", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
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
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	// inLay := ss.Net.LayerByName("Input").(leabra.LeabraLayer).AsLeabra()
	// outLay := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"TestNm", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"Mem", etensor.FLOAT64, nil, nil},
		{"TrgOnWasOff", etensor.FLOAT64, nil, nil},
		{"TrgOffWasOn", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	// sch = append(sch, etable.Schema{
	// 	{"InAct", etensor.FLOAT64, inLay.Shp.Shp, nil},
	// 	{"OutActM", etensor.FLOAT64, outLay.Shp.Shp, nil},
	// 	{"OutActP", etensor.FLOAT64, outLay.Shp.Shp, nil},
	// }...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Test Trial Plot"
	plt.Params.XAxisCol = "TrialName"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt) // this sets defaults so set params after
	plt.Params.BarWidth = 5
	plt.Params.XAxisRot = 45
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TestNm", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	plt.SetColParams("Mem", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOnWasOff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("TrgOffWasOn", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 0.5)
	}

	// plt.SetColParams("InAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	// plt.SetColParams("OutActM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	// plt.SetColParams("OutActP", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

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
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)
	dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val > 0
	})[0])
	dt.SetCellFloat("PctCor", row, agg.PropIf(tix, "SSE", func(idx int, val float64) bool {
		return val == 0
	})[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	trix := etable.NewIdxView(trl)
	spl := split.GroupBy(trix, []string{"TestNm"})
	for _, ts := range ss.TstStatNms {
		split.Agg(spl, ts, agg.AggMean)
	}
	ss.TstStats = spl.AggsToTable(etable.ColNameOnly)

	for ri := 0; ri < ss.TstStats.Rows; ri++ {
		tst := ss.TstStats.CellString("TestNm", ri)
		for _, ts := range ss.TstStatNms {
			dt.SetCellFloat(tst+" "+ts, row, ss.TstStats.CellFloat(ts, ri))
		}
	}

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
	ss.TstEpcPlot.GoUpdate()
	if ss.TstEpcFile != nil {
		if !ss.TstEpcHdrs {
			dt.WriteCSVHeaders(ss.TstEpcFile, etable.Tab)
			ss.TstEpcHdrs = true
		}
		dt.WriteCSVRow(ss.TstEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt) // this sets defaults so set params after
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			if ts == "Mem" {
				plt.SetColParams(tn+" "+ts, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
			} else {
				plt.SetColParams(tn+" "+ts, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
			}
		}
	}
	return plt
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", eplot.On, eplot.FixMin, 0, eplot.FixMax, .5)
		plt.SetColParams(lnm+" Act.Avg", eplot.On, eplot.FixMin, 0, eplot.FixMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TstEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 1
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	fzero := ss.FirstZero
	if fzero < 0 {
		fzero = ss.MaxEpcs
	}

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("NEpochs", row, float64(ss.TstEpcLog.Rows))
	dt.SetCellFloat("FirstZero", row, float64(fzero))
	dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			nm := tn + " " + ts
			dt.SetCellFloat(nm, row, agg.Mean(epcix, nm)[0])
		}
	}

	ss.LogRunStats()

	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"NEpochs", etensor.FLOAT64, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			sch = append(sch, etable.Column{tn + " " + ts, etensor.FLOAT64, nil, nil})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("NEpochs", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, tn := range ss.TstNms {
		for _, ts := range ss.TstStatNms {
			if ts == "Mem" {
				plt.SetColParams(tn+" "+ts, eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
			} else {
				plt.SetColParams(tn+" "+ts, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
			}
		}
	}
	return plt
}

//////////////////////////////////////////////
//  RunStats

// LogRunStats computes RunStats from RunLog data -- can be used for looking at prelim results
func (ss *Sim) LogRunStats() {
	dt := ss.RunLog
	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	for _, tn := range ss.TstNms {
		nm := tn + " " + "Mem"
		split.Desc(spl, nm)
	}
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "NEpochs")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)
	if ss.RunStatsPlot != nil {
		ss.ConfigRunStatsPlot(ss.RunStatsPlot, ss.RunStats)
	}
}

func (ss *Sim) ConfigRunStatsPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Hippocampus Run Stats Plot"
	plt.Params.XAxisCol = "Params"
	plt.SetTable(dt)
	plt.Params.BarWidth = 10
	plt.Params.Type = eplot.Bar
	plt.Params.XAxisRot = 45

	cp := plt.SetColParams("AB Mem:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	cp.ErrCol = "AB Mem:Sem"
	cp = plt.SetColParams("AC Mem:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	cp.ErrCol = "AC Mem:Sem"
	cp = plt.SetColParams("FirstZero:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 30)
	cp.ErrCol = "FirstZero:Sem"
	cp = plt.SetColParams("NEpochs:Mean", eplot.On, eplot.FixMin, 0, eplot.FixMax, 30)
	cp.ErrCol = "NEpochs:Sem"
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("hip_bench")
	gi.SetAppAbout(`This demonstrates a basic Hippocampus model in Leabra. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("hip_bench", "Hippocampus AB-AC", width, height)
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

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html
	nv.SetNet(ss.Net)
	ss.NetView = nv
	nv.ViewDefaults()

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnTrlPlot").(*eplot.Plot2D)
	ss.TrnTrlPlot = ss.ConfigTrnTrlPlot(plt, ss.TrnTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunStatsPlot").(*eplot.Plot2D)
	// ss.RunStatsPlot = ss.ConfigRunStatsPlot(plt, ss.RunStats)
	ss.RunStatsPlot = plt

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Pre Train", Icon: "fast-fwd", Tooltip: "Does full pretraining.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.PreTrain()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on trial -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
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

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Env", Icon: "gear", Tooltip: "select training input patterns: AB or AC."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			giv.CallMethod(ss, "SetEnv", vp)
		})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddAction(gi.ActOpts{Label: "Rebuild Net", Icon: "reset", Tooltip: "Rebuild network with current params"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.ReConfigNet()
		})

	tbar.AddAction(gi.ActOpts{Label: "Run Stats", Icon: "file-data", Tooltip: "compute stats from run log -- avail in plot"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.LogRunStats()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
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
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
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
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
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
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
		{"SetEnv", ki.Props{
			"desc": "select which set of patterns to train on: AB or AC",
			"icon": "gear",
			"Args": ki.PropSlice{
				{"Train on AC", ki.Props{}},
			},
		}},
	},
}

// OuterLoopParams are the parameters to run for outer crossed factor testing
// var OuterLoopParams = []string{"SmallHip", "MedHip"} //, "BigHip"}
var OuterLoopParams = []string{"MedHip"} //, "BigHip"}

// InnerLoopParams are the parameters to run for inner crossed factor testing
// var InnerLoopParams = []string{"List020", "List040", "List050", "List060", "List070", "List080"} // , "List100"}
var InnerLoopParams = []string{"List040", "List080", "List120", "List160", "List200"} // , "List100"}
//var InnerLoopParams = []string{"List010", "List020", "List030", "List040", "List50"} // , "List100"}

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
			rand.Seed(ss.RndSeed) // each run starts at same seed..
			ss.SetParamsSet(otf, "", ss.LogSetParams)
			ss.SetParamsSet(inf, "", ss.LogSetParams)
			ss.ReConfigNet() // note: this applies Base params to Network
			ss.ConfigEnv()
			ss.StopNow = false
			ss.PreTrain()
			ss.NewRun()
			ss.Train()
		}
	}
	ss.Tag = tag
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do")
	flag.IntVar(&ss.MaxEpcs, "epcs", 30, "maximum number of epochs to run (split between AB / AC)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
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
		fnm := ss.LogFileName("epc")
		ss.TstEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstEpcFile = nil
		} else {
			fmt.Printf("Saving test epoch log to: %v\n", fnm)
			defer ss.TstEpcFile.Close()
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
	// ss.Train()
	ss.TwoFactorRun()
	fnm := ss.LogFileName("runs")
	ss.RunStats.SaveCSV(gi.FileName(fnm), etable.Tab, etable.Headers)
}
