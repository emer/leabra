// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"flag"
	"fmt"
	"github.com/emer/emergent/env"
	_ "github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	_ "github.com/emer/etable/split"
	"github.com/goki/gi/giv"
	"github.com/goki/mat32"
	"log"
	"math/rand"
	"os"
	_ "reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	//"github.com/emer/leabra/pbwm"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"

	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/leabra/examples/pvlv/stepper"
	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/pvlv"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"

	"github.com/emer/leabra/examples/pvlv/data"
)

var TheSim Sim

func main() {
	// TheSim is the overall state for this simulation
	TheSim.VerboseInit, TheSim.LayerThreads = TheSim.CmdArgs() // doesn't return if nogui command line arg set
	TheSim.New()
	TheSim.Config()
	gimain.Main(func() { // this starts gui
		guirun(&TheSim)
	})
}

func guirun(ss *Sim) {
	//ss.Init(&ss.TrainEnv, false)
	win := ss.ConfigGui()
	win.StartEventLoop()
	//ss.RunLoop()
}

func (ss *Sim) RunLoop() {
	for {
		ss.Stepper.WaitToGo()
	}
}

type MonitorDatum []string

type MonitorVal interface {
	GetMonitorVal([]string) float64
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
// TODO this is not done.

type Sim struct { // trying to duplicate the cemer version as closely as possible
	//gi.Layout
	ActivateParams bool               `desc:"whether to activate selected param sets at start of train -- otherwise just uses current values as-is"`
	SeqParams      *data.RunSeqParams `desc:"For sequences of runs (i.e., this is above the RunParams level)"`
	SeqParamsName  string             `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	RunParams      *data.RunParams    `desc:"for running Train directly"`
	Tag            string             `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	//RunConfig      *RunConfig  `view:"-" desc:"for running Train directly"`
	Params         params.Sets `view:"no-inline" desc:"pvlv-specific network parameters"`
	ParamSet       string
	StableParams   params.Set `view:"no-inline" desc:"shouldn't need to change these'"`
	MiscParams     params.Set `view:"no-inline" desc:"misc params -- network specs"`
	AnalysisParams params.Set `view:"no-inline" desc:"??"`
	TrainEnv       PVLVEnv    `desc:"Training environment -- PVLV environment"`
	TestEnv        PVLVEnv    `desc:"Testing environment -- PVLV environment"`

	StepsToRun                   int          `desc:"number of StopStepGrain steps to execute before stopping"`
	OrigSteps                    int          `view:"-" desc:"saved number of StopStepGrain steps to execute before stopping"`
	StepGrain                    StepGrain    `desc:"granularity for the Step command"`
	StopStepCondition            StopStepCond `desc:"granularity for conditional stop"`
	StopConditionTrialNameString string       `desc:"if StopStepCond is TrialName or NotTrialName, this string is used for matching the current AlphaTrialName"`
	StopStepCounter              env.Ctr      `inactive:"+" view:"-" desc:"number of times we've hit whatever StopStepGrain is set to'"`
	StepMode                     bool         `view:"-" desc:"running from Step command?"`
	TestMode                     bool         `inactive:"+" desc:"testing mode, no training"`

	CycleUpdateNetView           bool `desc:"determines whether network views will be updated on a cycle-by-cycle basis (slow, but often quite useful for seeing how processing is proceeding)"`
	NetTimesCycleQtr             bool `desc:"turn this OFF to see cycle-level updating"`
	TrialAnalysisTimeLogInterval int
	TrialAnalUpdateCmpGraphs     bool          `desc:"turn off to preserve existing cmp graphs - else saves cur as cmp for new run"`
	Net                          *pvlv.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`

	CycleOutputDataRows int                 `desc:"maximum number of rows for CycleOutputData"`
	CycleOutputData     *etable.Table       `view:"no-inline" desc:"Cycle-level output data"`
	CycleDataPlot       *eplot.Plot2D       `view:"no-inline" desc:"Fine-grained trace data"`
	CycleOutputMetadata map[string][]string `view:"-"`
	//TrialOutputData             *etable.Table     `view:"-" desc:"Trial-level output data"`
	//EpochOutputData             *etable.Table     `view:"-" desc:"EpochCt-level output data"`
	//EpochOutputDataCmp             *etable.Table     `view:"-" desc:"EpochCt-level output data copy"`
	TrialTypeEpochFirstLog    *etable.Table `view:"-" desc:"TODO: FILL THIS IN"`
	TrialTypeEpochFirstLogCmp *etable.Table `view:"-" desc:"TODO: FILL THIS IN"`
	//HistoryGraphData          *etable.Table     `view:"-" desc:"data for history graph"`
	//RealTimeDataLog           *etable.Table     `view:"-"`
	//TrnEpcLog                 *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	//RunLog                    *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	//RunStats                  *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Time       leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn     bool              `desc:"whether to update the network view while running"`
	TrainUpdt  leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt   leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TstRecLays []string          `desc:"names of layers to record activations etc of during testing"`

	// internal state - view:"-"
	Win                    *gi.Window         `view:"-" desc:"main GUI window"`
	NetView                *netview.NetView   `view:"-" desc:"the network viewer"`
	ToolBar                *gi.ToolBar        `view:"-" desc:"the master toolbar"`
	WtsGrid                *etview.TensorGrid `view:"-" desc:"the weights grid view"`
	TrialTypeData          *etable.Table      `view:"no-inline" desc:"data for the TrialTypeData plot"`
	TrialTypeDataPlot      *eplot.Plot2D      `view:"no-inline" desc:"multiple views for different type of trials"`
	TrialTypeSet           map[string]int     `view:"-"`
	TrialTypeSetCounter    int                `view:"-"`
	TrialTypeEpochFirst    *eplot.Plot2D      `view:"-" desc:"epoch plot"`
	TrialTypeEpochFirstCmp *eplot.Plot2D      `view:"-" desc:"epoch plot"`
	HistoryGraph           *eplot.Plot2D      `view:"-" desc:"trial history"`
	RealTimeData           *eplot.Plot2D      `view:"-" desc:"??"`

	SaveWts bool `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui   bool `view:"-" desc:"if true, runing in no GUI mode"`
	//RunState    runcontrol.RunState `view:"-" desc:"flag to stop running"`
	RndSeed int64 `desc:"the current random seed"`
	//StateMux    sync.Mutex          `view:"-"`
	Stepper      *stepper.Stepper `view:"-"`
	SimHasRun    bool             `view:"-"`
	InitHasRun   bool             `view:"-"`
	VerboseInit  bool             `view:"-"`
	LayerThreads bool             `desc:"use per-layer threads"`

	// internal state - view:"-"
	SumDA        float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAbsDA     float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumRewPred   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumErr       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumSSE       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	LastEpcTime  leabra.Time                 `view:"-" desc:"timer for last epoch"`
	Interactive  bool                        `view:"-" desc:"true iff running through the GUI"`
	//TrialName         string
	//StopStepTrialName string

	// master lists of various kinds of parameters
	MasterRunParams    data.RunParamsMap  `view:"no-inline" desc:"master list of RunParams records"`
	MasterRunConfigs   data.RunConfigsMap `view:"-" desc:"master list of RunParams records"` // not used?
	MasterRunSeqParams data.SeqParamsMap  `view:"no-inline" desc:"master list of RunSeqParams records"`
	MasterEnvParams    data.EnvParamsMap  `desc:"master list of EnvEpochParams (trial groups) records"`

	InputShapes map[string][]int
	//AnalysisData *PVLVAnalysisData `view:"no-inline"`

	MaxSeqSteps int `view:"-" desc:"Maximum number of high-level sequences to run"`
	MaxRuns     int `view:"-" desc:"maximum number of model runs to perform"` // for non-GUI runs

	LHbInLayers []string `view:"-" desc:"layers that project to LHbRMTg"`
}

//type PVLVAnalysisData struct {
//	CycleOutputData *etable.Table
//	TrialOutputData *etable.Table
//	EpochOutputData *etable.Table
//	//AllTrialData *etable.Table
//}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

//func MySetRelPos (layer leabra.Layer, rel relpos.Rel) {
//	pos := layer.Pos()
//
//}

func (ss *Sim) OpenCemerWeights(fName string) {
	err := ss.Net.OpenWtsCpp(gi.FileName(fName))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}
}

var simOneTimeInit sync.Once

func (ss *Sim) New() {
	ss.InputShapes = map[string][]int{
		"StimIn":    data.StimInShape,
		"ContextIn": data.ContextInShape,
		"USTimeIn":  data.USTimeInShape, // valence, cs, time, us
		"PosPV":     data.USInShape,
		"NegPV":     data.USInShape,
	}
	ss.Net = &pvlv.Network{}
	ss.CycleOutputData = &etable.Table{}
	//ss.TrialOutputData = &etable.Table{}
	//ss.EpochOutputData = &etable.Table{}
	ss.TrialTypeData = &etable.Table{}
	ss.TrialTypeSet = map[string]int{}
	ss.TrialTypeSetCounter = 0
	ss.TrialTypeEpochFirstLog = &etable.Table{}
	ss.TrialTypeEpochFirstLogCmp = &etable.Table{}
	//ss.HistoryGraphData = &etable.Table{}
	//ss.RealTimeDataLog = &etable.Table{}
	// TODO: fix these
	//ss.RunControl = make(chan runcontrol.RunState)
	ss.ActivateParams = true
	simOneTimeInit.Do(func() {
		ss.MasterRunParams = data.AllRunParams()
		ss.MasterRunSeqParams = data.AllSeqParams()
		ss.MasterRunConfigs = data.AllRunConfigs()
		ss.MasterEnvParams = data.AllEnvParams()
		ss.TrainEnv = PVLVEnv{Nm: "Train", Dsc: "training environment"}
		ss.TrainEnv.New(ss)
		ss.Stepper = stepper.New()
		ss.Stepper.RegisterStopChecker(CheckStopCondition, SimState{ss, &ss.TrainEnv})
		ss.Stepper.RegisterPauseNotifier(NotifyPause, SimState{ss, &ss.TrainEnv})
	})
	ss.Defaults()
	ss.Params = ParamSets
	ss.CycleOutputDataRows = 8000
	//ss.AnalysisData = &PVLVAnalysisData{TrialOutputData: &etable.Table{
	//	Cols:       nil,
	//	ColNames:   nil,
	//	Rows:       0,
	//	ColNameMap: nil,
	//	MetaData:   nil,
	//}}
	//ss.Experiments = Experiments()
	//ss.RunSeqParams = ParamSets
	//ss.RunParams = ParamSets
	//ss.PvlvParams = ParamSets
	//ss.StableParams = ParamSets
	//ss.MiscParams = ParamSets
	//ss.AnalysisParams = ParamSets
	ss.InitHasRun = false

}

func (ss *Sim) Defaults() {
	//defaultRunSeqNm := "USDebug"
	defaultRunSeqNm := "PosAcq"
	ss.SeqParamsName = defaultRunSeqNm
	err := ss.SetRunSeqParams()
	if err != nil {
		panic(err)
	}
	ss.TrainUpdt = leabra.Quarter
	ss.CycleUpdateNetView = false
	ss.NetTimesCycleQtr = true
	ss.TrialAnalysisTimeLogInterval = 1
	ss.TrialAnalUpdateCmpGraphs = true
	ss.StepsToRun = 1
	ss.StepGrain = SGTrial
	ss.StopStepCondition = SSNone
	ss.StopConditionTrialNameString = "_t3"
	ss.ViewOn = true
	ss.RndSeed = 1
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Top-level Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigOutputData()
	ss.InitSim(&ss.TrainEnv)
}

func (ss *Sim) ConfigEnv() {
	//if ss.MaxRuns == 0 { // allow user override
	//	ss.MaxRuns = 1
	//}
	//if ss.MaxEpcs == 0 { // allow user override
	//	ss.MaxEpcs = 30
	//}
	//if ss.MaxTrls == 0 { // allow user override
	//	ss.MaxTrls = 100
	//}
	//
	ss.TrainEnv.Init(ss)
}

////////////////////////////////////////////////////////////////////////////////
// Init, utils

func (ss *Sim) Init(aki ki.Ki) {
	//ss.Layout.Init(aki)
}

func (ss *Sim) Ki() *Sim {
	return ss
}

type StopStepCond int

const (
	SSNone              StopStepCond = iota
	SSError                          // Error
	SSCorrect                        // Correct
	SSTrialNameMatch                 // Trial Name
	SSTrialNameNonmatch              // Not Trial Name
	StopStepCondN
)

////go:generate stringer -type=StopStepCond -linecomment // moved to stringers.go
var KiT_StopStepCond = kit.Enums.AddEnum(StopStepCondN, kit.NotBitFlag, nil)

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) InitSim(ev *PVLVEnv) {
	rand.Seed(ss.RndSeed)
	ss.Stepper.Init()
	ev.TrialInstances = new(TrialInstanceList).Init()
	err := ss.SetParams("", ss.VerboseInit) // all sheets
	if err != nil {
		fmt.Println(err)
	}
	_ = ss.InitRun(ev)
	ss.Net.InitWts()
	ss.UpdateView(true)
	ss.InitHasRun = true
	ss.VerboseInit = false
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
	var ev *PVLVEnv
	if train {
		ev = &ss.TrainEnv
	} else {
		ev = &ss.TestEnv
	}
	return fmt.Sprintf("Run:\t%d\tEpoch:\t%03d\tTrial:\t%02d\tAlpha:\t%01d\tCycle:\t%03d\t\tName:\t%12v\t\t\t",
		ev.RunCt.Cur, ev.EpochCt.Cur, ev.TrialCt.Cur, ev.AlphaCycle.Cur, ss.Time.Cycle, ev.AlphaTrialName) //, ev.USTimeInStr)
	//return fmt.Sprintf("Run:\t%d\tEpoch:\t%2d\tTrial:\t%3d\tCycle:\t%03d\t\tName:\t%v\tUSTimeInStr:\t%v\t",
	//	ev.RunCt.Cur, ev.EpochCt.Cur, ev.AlphaCycle.Cur, ss.Time.Cycle, ev.AlphaTrialName, ev.USTimeInStr)
	//return "TBD"
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate()
	}
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) NotifyStopped() {
	ss.Stepper.Stop()
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
	fmt.Println("stopped")
}

// configure output data tables
func (ss *Sim) ConfigCycleOutputData(dt *etable.Table) {
	dt.SetMetaData("name", "CycleOutputData")
	dt.SetMetaData("desc", "Cycle-level output data")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	floatCols := []string{
		"VSPatchPosD1_0_Act", "VSPatchPosD2_0_Act",
		"VSPatchNegD1_0_Act", "VSPatchNegD2_0_Act",
		"VSMatrixPosD1_0_Act", "VSMatrixPosD2_0_Act",
		"VSMatrixNegD1_0_Act", "VSMatrixNegD2_0_Act",

		"VSMatrixPosD1_0_ModNet", "VSMatrixPosD1_0_DA",
		"VSMatrixPosD2_0_ModNet", "VSMatrixPosD2_0_DA",
		//"PPTg_0_TotalAct", "LHbRMTg_0_TotalAct", "PosPV_0_TotalAct", "NegPV_0_TotalAct",
		//"VSPatchPosD1_0_TotalAct", "VSPatchPosD2_0_TotalAct", "VSPatchNegD1_0_TotalAct", "VSPatchNegD2_0_TotalAct",
		//"VSMatrixPosD1_0_TotalAct", "VSMatrixPosD2_0_TotalAct", "VSMatrixNegD1_0_TotalAct", "VSMatrixNegD2_0_TotalAct",

		"PosPV_0_Act", "StimIn_0_Act", "ContextIn_0_Act", "USTimeIn_0_Act",

		"VTAp_0_Act", "LHbRMTg_0_Act", "PPTg_0_Act",
		"VTAp_0_PPTgDApt", "VTAp_0_LHbDA", "VTAp_0_PosPVAct", "VTAp_0_VSPosPVI", "VTAp_0_VSNegPVI", "VTAp_0_BurstLHbDA",
		"VTAp_0_DipLHbDA", "VTAp_0_TotBurstDA", "VTAp_0_TotDipDA", "VTAp_0_NetDipDA", "VTAp_0_NetDA", "VTAp_0_SendVal",

		"CElAcqPosD1_0_ModAct", "CElAcqPosD1_0_PVAct",
		"CElAcqPosD1_0_ModLevel", "CElAcqPosD1_0_ModLrn",
		"CElAcqPosD1_0_Act", "CElAcqPosD1_0_ActP", "CElAcqPosD1_0_ActQ0", "CElAcqPosD1_0_ActM",
		"CElAcqPosD1_1_ModPoolAvg", "CElAcqPosD1_1_PoolActAvg", "CElAcqPosD1_1_PoolActMax",

		"CElExtPosD2_0_ModAct", "CElExtPosD2_0_ModLevel", "CElExtPosD2_0_ModNet", "CElExtPosD2_0_ModLrn",
		"CElExtPosD2_0_Act", "CElExtPosD2_0_Ge", "CElExtPosD2_0_Gi", "CElExtPosD2_0_Inet", "CElExtPosD2_0_GeRaw",

		"BLAmygPosD1_3_Act", "BLAmygPosD1_3_ModAct", "BLAmygPosD1_3_ActDiff", "BLAmygPosD1_3_ActQ0",
		"BLAmygPosD1_3_ModLevel", "BLAmygPosD1_3_ModNet", "BLAmygPosD1_3_ModLrn", "BLAmygPosD1_3_DA",
		"BLAmygPosD1_1_PoolActAvg", "BLAmygPosD1_1_PoolActMax", "BLAmygPosD1_2_PoolActAvg", "BLAmygPosD1_2_PoolActMax",

		"BLAmygPosD2_5_Act", "BLAmygPosD2_5_ModAct", "BLAmygPosD2_5_ActDiff", "BLAmygPosD2_5_ActQ0",
		"BLAmygPosD2_5_ModLevel", "BLAmygPosD2_5_ModNet", "BLAmygPosD2_5_ModLrn", "BLAmygPosD2_5_DA",
		//"BLAmygPosD2_1_PoolActAvg", "BLAmygPosD2_1_PoolActMax",
		//"BLAmygPosD1_2_ModAct", "BLAmygPosD2_2_Act",

		"CEmPos_0_Act",
		//"CEmPos_1_PoolActAvg", "CEmPos_1_PoolActMax",
	}
	ss.CycleOutputMetadata = make(map[string][]string, len(floatCols))

	sch := etable.Schema{}
	sch = append(sch, etable.Column{Name: "Cycle", Type: etensor.INT32})
	sch = append(sch, etable.Column{Name: "GlobalStep", Type: etensor.INT32})
	for _, colName := range floatCols {
		parts := strings.Split(colName, "_")
		//lyNm := parts[0]
		idx := parts[1]
		val := parts[2]
		data := []string{}
		sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64})
		data = append(data, val)
		data = append(data, idx)
		ss.CycleOutputMetadata[colName] = data
	}
	dt.SetFromSchema(sch, ss.CycleOutputDataRows)
}

func (ss *Sim) ConfigCycleOutputDataPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "CycleOutputData"
	plt.Params.XAxisCol = "GlobalStep"
	plt.Params.XAxisLabel = "Cycle"
	plt.Params.Type = eplot.XY
	plt.SetTable(dt)

	for iCol := range dt.ColNames {
		colName := dt.ColNames[iCol]
		var colOnOff, colFixMin, colFixMax bool
		var colMin, colMax float64
		switch colName {
		case "Cycle":
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = 0
			colFixMax = eplot.FixMax
			colMax = 99
		case "GlobalStep":
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = 0
			colFixMax = eplot.FixMax
			colMax = float64(dt.Rows - 1)
		case "StimIn_0_Act", "ContextIn_0_Act", "PosPV_0_Act":
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = -1.25
			colFixMax = eplot.FixMax
			colMax = 1.25
		default:
			colOnOff = eplot.Off
			colFixMin = eplot.FixMin
			colMin = -1.25
			colFixMax = eplot.FixMax
			colMax = 1.25
		}
		plt.SetColParams(colName, colOnOff, colFixMin, colMin, colFixMax, colMax)
	}
	return plt
}

//func (ss *Sim) ConfigTrialOutputData(dt *etable.Table) {
//	dt.SetMetaData("name", "TrialOutputData")
//	dt.SetMetaData("desc", "Trial-level output data")
//	dt.SetMetaData("read-only", "true")
//	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
//
//	floatCols := []string{
//		"RTCycles", "SSE", "NormErr", "ExtRew", "VTAp_act", "VTAn_act", "LHbRMTg_act",
//		"PPTg_US0_act", "PosPV_US0_act", "PosPV_US1_act", "NegPV_US0_act", "CEmPos_US0_act", "CEmPos_US1_act",
//		"CEmNeg_US0_act", "BLAmygD1_US0_act", "BLAmygD2_US0_act", "BLAmygD1_US1_act", "BLAmygD2_US1_act",
//		"BLAmygD1_delta", "BLAmygD1_absdif", "BLAmygD2_delta", "BLAmygD2_absdif", "BLAmygD1_fmstim_maxwt",
//		"CElAcqPosD1_US0_act", "CElExtPosD2_US0_act", "CElAcqPosD1_US1_act", "CElExtPosD2_US1_act",
//		"CElAcqNegD2_US0_act", "CElExtNegD1_US0_act", "CElAcqPosD1_US0_act_dif", "CElExtPosD2_US0_act_dif",
//		"VSPatchPosD1_US0_act", "VSPatchPosD2_US0_act", "VSPatchNegD2_US0_act", "VSPatchNegD1_US0_act",
//		"VSMatrixPosD1_US0_act", "VSMatrixPosD2_US0_act", "VSMatrixPosD1_US1_act", "VSMatrixPosD2_US1_act",
//		"VSMatrixNegD2_US0_act", "VSMatrixNegD1_US0_act", "vta_glut_input", "vta_input_to_rmtg", "vta_pvi_input",
//		"vta_net_da", "LHbRMT_netin_avg", "LHbRMT_acts_eq_avg", "LHbRMT_user_data_net_lhb", "VTAp_user_data_lhb_da",
//	}
//	sch := etable.Schema{
//		{Name: "Batch", Type: etensor.INT},
//		{Name: "Epoch", Type: etensor.INT},
//		{Name: "Step", Type: etensor.FLOAT64},
//		{Name: "TrainMode", Type: etensor.STRING},
//		{Name: "Trial", Type: etensor.INT},
//		{Name: "Tick", Type: etensor.INT},
//		{Name: "Time", Type: etensor.INT},
//		{Name: "Stimulus", Type: etensor.STRING},
//		{Name: "TrialName", Type: etensor.STRING},
//		{Name: "GroupName", Type: etensor.STRING},
//	}
//	for _, colName := range floatCols { sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64}) }
//	dt.SetFromSchema(sch, 0)
//
//	//dt.SetCellString("TrialName", 1, "") // prevent crash when event loop starts
//}

func (ss *Sim) ConfigEpochOutputData(dt *etable.Table) {
	colNames := []string{
		"AvgSSE", "CntErr", "AvgNormErr", "AvgExtRew", "AvgCycles", "EpochTimeTot", "EpochTimeUsr", "AvgTick",
		"BLAmygPosD1_Marker_Fm_VTAp_netrel", "BLAmygPosD1_Fm_PosPV_netrel", "BLAmygPosD1_Fm_Stim_In_netrel",
		"BLAmygPosD1_Inhib_Fm_BLAmygPosD2_netrel", "BLAmygNegD2_Marker_Fm_VTAp_netrel", "BLAmygNegD2_Fm_NegPV_netrel",
		"BLAmygNegD2_Fm_Stim_In_netrel", "BLAmygNegD2_Inhib_Fm_BLAmygNegD1_netrel", "BLAmygPosD2_Marker_Fm_VTAp_netrel",
		"BLAmygPosD2_Fm_Context_In_netrel", "BLAmygPosD2_Deep_Mod_Fm_BLAmygPosD1_netrel",
		"BLAmygNegD1_Marker_Fm_VTAp_netrel", "BLAmygNegD1_Fm_Context_In_netrel",
		"BLAmygNegD1_Deep_Mod_Fm_BLAmygNegD2_netrel", "CElAcqPosD1_Deep_Raw_Fm_PosPV_netrel",
		"CElAcqPosD1_Marker_Fm_VTAp_netrel", "CElAcqPosD1_Inhib_Fm_CElExtPosD2_netrel",
		"CElAcqPosD1_Fm_BLAmygPosD1_netrel", "CElAcqPosD1_Fm_Stim_In_netrel",
		"CElExtPosD2_Deep_Mod_Fm_CElAcqPosD1_netrel", "CElExtPosD2_Marker_Fm_VTAp_netrel",
		"CElExtPosD2_Inhib_Fm_CElAcqPosD1_netrel", "CElExtPosD2_Fm_BLAmygPosD2_netrel",
		"CElAcqNegD2_Deep_Raw_Fm_NegPV_netrel", "CElAcqNegD2_Marker_Fm_VTAp_netrel",
		"CElAcqNegD2_Inhib_Fm_CElExtNegD1_netrel", "CElAcqNegD2_Fm_BLAmygNegD2_netrel",
		"CElAcqNegD2_Fm_Stim_In_netrel", "CElExtNegD1_Deep_Mod_Fm_CElAcqNegD2_netrel",
		"CElExtNegD1_Marker_Fm_VTAp_netrel", "CElExtNegD1_Inhib_Fm_CElAcqNegD2_netrel",
		"CElExtNegD1_Fm_BLAmygNegD1_netrel", "CEmPos_Fm_CElAcqPosD1_netrel", "CEmPos_Inhib_Fm_CElExtPosD2_netrel",
		"CEmNeg_Fm_CElAcqNegD2_netrel", "CEmNeg_Inhib_Fm_CElExtNegD1_netrel", "VSPatchPosD1_Marker_Fm_VTAp_netrel",
		"VSPatchPosD1_Deep_Mod_Fm_BLAmygPosD1_netrel", "VSPatchPosD1_Fm_USTime_In_netrel",
		"VSPatchPosD2_Marker_Fm_VTAp_netrel", "VSPatchPosD2_Deep_Mod_Fm_BLAmygPosD1_netrel",
		"VSPatchPosD2_Fm_USTime_In_netrel", "VSPatchNegD2_Marker_Fm_VTAp_netrel",
		"VSPatchNegD2_Deep_Mod_Fm_BLAmygNegD2_netrel", "VSPatchNegD2_Fm_USTime_In_netrel",
		"VSPatchNegD1_Marker_Fm_VTAp_netrel", "VSPatchNegD1_Deep_Mod_Fm_BLAmygNegD2_netrel",
		"VSPatchNegD1_Fm_USTime_In_netrel", "VSMatrixPosD1_Marker_Fm_VTAp_netrel",
		"VSMatrixPosD1_Deep_Mod_Fm_BLAmygPosD1_netrel", "VSMatrixPosD1_Fm_Stim_In_netrel",
		"VSMatrixPosD2_Marker_Fm_VTAp_netrel", "VSMatrixPosD2_Deep_Mod_Fm_VSMatrixPosD1_netrel",
		"VSMatrixPosD2_Fm_Stim_In_netrel", "VSMatrixNegD2_Marker_Fm_VTAp_netrel",
		"VSMatrixNegD2_Deep_Mod_Fm_BLAmygNegD2_netrel", "VSMatrixNegD2_Fm_Stim_In_netrel",
		"VSMatrixNegD1_Marker_Fm_VTAp_netrel", "VSMatrixNegD1_Deep_Mod_Fm_VSMatrixNegD2_netrel",
		"VSMatrixNegD1_Fm_Stim_In_netrel", "PPTg_Fm_CEmPos_netrel", "LHbRMTg_Marker_Fm_PosPV_netrel",
		"LHbRMTg_Marker_Fm_NegPV_netrel", "LHbRMTg_Marker_Fm_VSPatchPosD1_netrel",
		"LHbRMTg_Marker_Fm_VSPatchPosD2_netrel", "LHbRMTg_Marker_Fm_VSPatchNegD2_netrel",
		"LHbRMTg_Marker_Fm_VSPatchNegD1_netrel", "LHbRMTg_Marker_Fm_VSMatrixPosD1_netrel",
		"LHbRMTg_Marker_Fm_VSMatrixPosD2_netrel", "LHbRMTg_Marker_Fm_VSMatrixNegD2_netrel",
		"LHbRMTg_Marker_Fm_VSMatrixNegD1_netrel", "VTAp_Marker_Fm_PPTg_p_netrel", "VTAp_Marker_Fm_LHbRMTg_netrel",
		"VTAp_Marker_Fm_PosPV_netrel", "VTAp_Marker_Fm_VSPatchPosD1_netrel", "VTAp_Marker_Fm_VSPatchPosD2_netrel",
		"VTAp_Marker_Fm_VSPatchNegD1_netrel", "VTAp_Marker_Fm_VSPatchNegD2_netrel", "VTAn_Marker_Fm_LHbRMTg_netrel",
		"VTAn_Marker_Fm_NegPV_netrel", "VTAn_Marker_Fm_VSPatchNegD2_netrel", "VTAn_Marker_Fm_VSPatchNegD1_netrel",
		"VTAn_Marker_Fm_PPTg_n_netrel", "PPTg_n_Fm_CEmPos_netrel", "netmax",
	}
	dt.SetMetaData("name", "EpochOutputData")
	dt.SetMetaData("desc", "Epoch-level output data")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	sch := etable.Schema{
		{Name: "Batch", Type: etensor.INT},
		{Name: "Epoch", Type: etensor.INT},
		{Name: "TrainMode", Type: etensor.STRING},
	}
	for _, colName := range colNames {
		sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigOutputData() {
	ss.ConfigCycleOutputData(ss.CycleOutputData)
	//ss.ConfigTrialOutputData(ss.TrialOutputData)
	//ss.ConfigEpochOutputData(ss.EpochOutputData)
	ss.ConfigTrialTypeEpochFirstLog(ss.TrialTypeEpochFirstLog, "TrialTypeEpochFirst")
	ss.ConfigTrialTypeEpochFirstLog(ss.TrialTypeEpochFirstLogCmp, "TrialTypeEpochFirstCmp")
	ss.ConfigTrialTypeData(ss.TrialTypeData)
}

// end output data config

// configure plots
func (ss *Sim) ConfigTrialTypeEpochFirstLog(dt *etable.Table, name string) {
	dt.SetMetaData("name", name)
	dt.SetMetaData("desc", "Multi-epoch monitor")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))
	sch := etable.Schema{}

	colNames := []string{
		"TrialName", "EpochAll", "Tick", "Time", "Stimulus", "GroupName", "RTCycles", "SSE", "NormErr",
		"ExtRew", "VTAp_act", "VTAn_act", "LHbRMTg_act", "PosPV_US0_act", "PosPV_US1_act", "NegPV_US0_act",
		"CEmPos_US0_act", "CEmPos_US1_act", "CEmNeg_US0_act", "BLAmygD1_US0_act", "BLAmygD2_US0_act",
		"BLAmygD1_US1_act", "BLAmygD2_US1_act", "BLAmygD1_delta", "BLAmygD1_absdif", "BLAmygD2_delta",
		"BLAmygD2_absdif", "BLAmygD1_fmstim_maxwt", "CElAcqPosD1_US0_act", "CElExtPosD2_US0_act", "CElAcqPosD1_US1_act",
		"CElExtPosD2_US1_act", "CElAcqNegD2_US0_act", "CElExtNegD1_US0_act", "CElAcqPosD1_US0_act_dif",
		"CElExtPosD2_US0_act_dif", "VSPatchPosD1_US0_act", "VSPatchPosD2_US0_act", "VSPatchNegD2_US0_act",
		"VSPatchNegD1_US0_act", "VSMatrixPosD1_US0_act", "VSMatrixPosD2_US0_act", "VSMatrixPosD1_US1_act",
		"VSMatrixPosD2_US1_act", "VSMatrixNegD2_US0_act", "VSMatrixNegD1_US0_act", "VTA_glut_input", "VTA_input_to_rmtg",
		"VTA_pvi_input", "VTA_net_DA", "LHbRMT_netin_avg", "LHbRMT_acts_eq_avg", "LHbRMT_user_data_net_LHb",
		"VTAp_user_data_LHb_DA", "TrialType", "Batch", "Epoch",
	}

	for _, colName := range colNames {
		switch colName {
		case "trial_name", "Time", "Stimulus", "group_name", "trial_typ":
			sch = append(sch, etable.Column{Name: colName, Type: etensor.STRING})
		default:
			sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64})
		}
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrialTypeData(dt *etable.Table) {
	dt.SetMetaData("name", "TrialTypeData")
	dt.SetMetaData("desc", "Plot of activations for different trial types")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	colNames := []string{
		"TrialType",
		"VTAp_act", "VTAn_act",
		"CElAcqPosD1_US0_act", "CElExtPosD2_US0_act",
		"VSPatchPosD1_US0_act", "VSPatchPosD2_US0_act",
		"VSPatchNegD1_US0_act", "VSPatchNegD2_US0_act",
		"VSMatrixPosD1_US0_act", "VSMatrixPosD2_US0_act",
		"VSMatrixNegD1_US0_act", "VSMatrixNegD2_US0_act",
		"CElAcqNegD2_US0_act", "CElExtNegD1_US0_act",
		"CEmPos_US0_act", "LHbRMTg_act",
		//"BLAmygD1_US0_act", "BLAmygD2_US0_act",
		//"BLAmygD1_US1_act", "BLAmygD2_US1_act",
	}
	sch := etable.Schema{}

	for _, colName := range colNames {
		if colName == "TrialType" {
			sch = append(sch, etable.Column{Name: colName, Type: etensor.STRING})
		} else {
			sch = append(sch, etable.Column{Name: colName, Type: etensor.FLOAT64})
		}
	}
	dt.SetFromSchema(sch, 1)
	for _, colName := range colNames {
		if colName == "TrialType" {
			dt.SetCellString("TrialType", 0, "B_Rf_POS_omit_t0")
		} else {
			dt.SetCellFloat(colName, 0, 0)
		}
	}
}

func (ss *Sim) ConfigTrialTypeDataPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "TrialTypeData"
	plt.Params.XAxisCol = "TrialType"
	plt.Params.XAxisLabel = " "
	plt.Params.XAxisRot = 45.0
	plt.Params.Type = eplot.XY
	plt.Params.LineWidth = 1.25
	plt.SetTable(dt)

	for _, colName := range dt.ColNames {
		var colOnOff, colFixMin, colFixMax bool
		var colMin, colMax float64
		switch colName {
		case "VTAp_act":
			colOnOff = eplot.On
			colFixMin = eplot.FixMin
			colMin = -2
			colFixMax = eplot.FixMax
			colMax = 2
		case "CElAcqPosD1_US0_act", "VSPatchPosD1_US0_act", "VSPatchPosD2_US0_act", "LHbRMTg_act":
			colOnOff = eplot.On
			colFixMin = eplot.FloatMin
			colMin = 0
			colFixMax = eplot.FloatMax
			colMax = 0
		default:
			colOnOff = eplot.Off
			colFixMin = eplot.FloatMin
			colMin = 0
			colFixMax = eplot.FloatMax
			colMax = 0
		}
		plt.SetColParams(colName, colOnOff, colFixMin, colMin, colFixMax, colMax)
	}
	return plt
}

func (ss *Sim) ConfigTrialTypeEpochFirstPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "All Epochs"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Epoch", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("VTAp_act", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)
	plt.SetColParams("BLAmygD2_US0_act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("BLAmygD1_US0_act", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	plt.SetColParams("CElAcqPosD1_US0_act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CElExtPosD2_US0_act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("VSMatrixPosD1_US0_act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("VSMatrixPosD2_US0_act", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CElAcqNegD2_US0_act", eplot.Off, eplot.FloatMin, 0, eplot.FloatMin, 1)

	return plt
}

func (ss *Sim) ConfigHistoryGraph(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "History Graph"
	plt.Params.XAxisCol = "GroupName"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("GroupName", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("LHbRMTg_act", eplot.On, eplot.FloatMin, 0, eplot.FloatMax, 0)
	return plt
}

func (ss *Sim) ConfigRealTimeData(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Real Time Data"
	plt.Params.XAxisCol = "GroupNumber"
	plt.SetTable(dt)
	plt.SetColParams("GroupNumber", eplot.Off, eplot.FloatMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("VTAAct", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)
	return plt
}

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	pos := nv.Scene().Camera.Pose.Pos
	nv.Scene().Camera.Pose.Pos.Set(pos.X, pos.Y, 2)
	nv.Scene().Camera.LookAt(mat32.Vec3{Y: 0.5, Z: 1}, mat32.Vec3{Y: 1})
	ctrs := nv.Counters()
	ctrs.SetProp("font-family", "Go Mono")
}

func (ss *Sim) Running() bool {
	return ss.Stepper.Active()
}

func (ss *Sim) Stopped() bool {
	runState, _ := ss.Stepper.CheckStates()
	return runState == stepper.Stopped
}

func (ss *Sim) Paused() bool {
	runState, _ := ss.Stepper.CheckStates()
	return runState == stepper.Paused
}

func (ss *Sim) Stop() {
	ss.Stepper.Enter(stepper.Stopped)
}

var CemerWtsFname = ""

func FileViewLoadCemerWts(vp *gi.Viewport2D) {
	giv.FileViewDialog(vp, CemerWtsFname, ".svg", giv.DlgOpts{Title: "Open SVG"}, nil,
		vp.Win, func(recv, send ki.Ki, sig int64, data interface{}) {
			if sig == int64(gi.DialogAccepted) {
				dlg, _ := send.(*gi.Dialog)
				CemerWtsFname = giv.FileViewDialogValue(dlg)
				err := TheSim.Net.OpenWtsCpp(gi.FileName(CemerWtsFname))
				if err != nil {
					fmt.Println(err)
				}
			}
		})
}

func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1600
	gi.SetAppName("BVPVLV")
	gi.SetAppAbout(`A bi-valent version of the Primary Value Learned Value model of the phasic dopamine signaling system. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch7/pvlv/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("bvpvlv", "BVPVLV", width, height)
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
	nv.Params.LayNmSize = 0.02
	nv.SetNet(ss.Net)
	ss.NetView = nv
	ss.ConfigNetView(nv)

	eplot.PlotColorNames = []string{
		"black", "SkyBlue", "red", "ForestGreen", "lawngreen", "purple", "orange", "brown", "chartreuse", "navy",
		"cyan", "magenta", "tan", "salmon", "yellow", "blue", "SkyBlue", "pink"}

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrialTypeData").(*eplot.Plot2D)
	ss.TrialTypeDataPlot = ss.ConfigTrialTypeDataPlot(plt, ss.TrialTypeData)
	plt = tv.AddNewTab(eplot.KiT_Plot2D, "CycleOutputData").(*eplot.Plot2D)
	ss.CycleDataPlot = ss.ConfigCycleOutputDataPlot(plt, ss.CycleOutputData)

	frm := gi.AddNewFrame(tv, "TrialTypeEpochFirst", gi.LayoutVert)
	frm.SetStretchMax()
	pltCmp := frm.AddNewChild(eplot.KiT_Plot2D, "TrialTypeEpochFirst_cmp").(*eplot.Plot2D)
	pltCmp.SetStretchMax()
	pltLower := frm.AddNewChild(eplot.KiT_Plot2D, "TrialTypeEpochFirst").(*eplot.Plot2D)
	pltLower.SetStretchMax()
	tv.AddTab(frm, "TrialTypeEpochFirst")
	ss.TrialTypeEpochFirst = ss.ConfigTrialTypeEpochFirstPlot(pltLower, ss.TrialTypeEpochFirstLog)
	ss.TrialTypeEpochFirstCmp = ss.ConfigTrialTypeEpochFirstPlot(pltCmp, ss.TrialTypeEpochFirstLogCmp)

	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "HistoryGraph").(*eplot.Plot2D)
	//ss.HistoryGraph = ss.ConfigHistoryGraph(plt, ss.HistoryGraphData)
	//
	//plt = tv.AddNewTab(eplot.KiT_Plot2D, "RealTimeData").(*eplot.Plot2D)
	//ss.RealTimeData = ss.ConfigRealTimeData(plt, ss.RealTimeDataLog)

	input := tv.AddNewTab(etview.KiT_TableView, "StdInputData").(*etview.TableView)
	input.SetName("StdInputData")
	input.SetTable(ss.TrainEnv.StdInputData, nil)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Run init code. Global variables retain current values unless reset in the init code", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.Running())
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stepper.RequestStop(true)
		if !ss.InitHasRun {
			ss.InitSim(&ss.TrainEnv)
		}
		if ss.SimHasRun {
			gi.ChoiceDialog(ss.Win.Viewport, gi.DlgOpts{Title: "Init weights?", Prompt: "Initialize network weights?"},
				[]string{"Yes", "No"}, ss.Win.This(),
				func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == 0 {
						fmt.Println("initializing weights")
						ss.Net.InitWts()
						ss.SimHasRun = false
					}
				})
		}
		_ = ss.InitRun(&ss.TrainEnv)
		ss.UpdateView(true)
		ss.Win.Viewport.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Run", Icon: "run", Tooltip: "Run the currently selected sequence. If not initialized, will run initialization first",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.Running())
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.Running() {
			if ss.Stopped() {
				if !ss.InitHasRun {
					fmt.Println("Initializing...")
					ss.InitSim(&ss.TrainEnv)
				}
				ss.SimHasRun = true
				ss.Stepper.Enter(stepper.Running)
				ss.ToolBar.UpdateActions()
				go ss.TrainMultiRun()
			} else if ss.Paused() {
				ss.Stepper.PleaseEnter(stepper.Running)
				ss.ToolBar.UpdateActions()
			}
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "StepCyc", Icon: "run", Tooltip: "Step by Cycles.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.Running())
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunSteps(Cycle)
	})

	tbar.AddAction(gi.ActOpts{Label: "StepQtr", Icon: "run", Tooltip: "Step by Quarters.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.Running())
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunSteps(Quarter)
	})

	tbar.AddAction(gi.ActOpts{Label: "StepMinus", Icon: "run", Tooltip: "Step to the end of the Minus Phase.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.Running())
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunSteps(SettleMinus)
	})

	tbar.AddAction(gi.ActOpts{Label: "StepPlus", Icon: "run", Tooltip: "Step to the end of the Plus Phase.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.Running())
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunSteps(SettlePlus)
	})

	tbar.AddAction(gi.ActOpts{Label: "StepAlpha", Icon: "run", Tooltip: "Step by Alpha Cycles.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.Running())
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunSteps(AlphaCycle)
	})

	tbar.AddAction(gi.ActOpts{Label: "Step", Icon: "run", Tooltip: "Step by whatever unit is set.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.Running())
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunSteps(ss.StepGrain)
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Stop the current program at its next natural stopping point (i.e., cleanly stopping when appropriate chunks of computation have completed).", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.Running())
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		fmt.Println("STOP!")
		ss.Stepper.RequestStop(true)
		ss.ToolBar.UpdateActions()
		ss.Win.Viewport.SetNeedsFullRender()
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

	fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)

	fmen.Menu.AddAction(gi.ActOpts{Label: "Load CEmer weights", Tooltip: "load a CEmer weights file", Data: ss}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		FileViewLoadCemerWts(vp)
	})

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

func (ss *Sim) RunSteps(grain StepGrain) {
	if !ss.Running() {
		if ss.Stopped() {
			if !ss.InitHasRun {
				fmt.Println("Initializing...")
				ss.InitSim(&ss.TrainEnv)
			}
			ss.SimHasRun = true
			ss.OrigSteps = ss.StepsToRun
			ss.Stepper.StartStepping(int(grain), ss.StepsToRun)
			ss.ToolBar.UpdateActions()
			go ss.TrainMultiRun()
		} else if ss.Paused() {
			ss.Stepper.SetStepGrain(int(grain))
			ss.Stepper.SetNSteps(ss.StepsToRun)
			ss.Stepper.PleaseEnter(stepper.Stepping)
			ss.ToolBar.UpdateActions()
		}
	}
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		err := ss.Params.ValidateSheets([]string{"Network"})
		if err != nil {
			fmt.Printf("error in validate sheets for Network: %v\n", err)
		}
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)

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
			applied, err := ss.Net.ApplyParams(netp, setMsg)
			if err != nil {
				fmt.Printf("error when applying %v, applied=%v, err=%v\n", netp, applied, err)
			}
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			applied, err := simp.Apply(ss, setMsg)
			if err != nil {
				fmt.Printf("error when applying %v, applied=%v, err=%v\n", simp, applied, err)
			}
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TrainEnv" etc
	return err
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"max-width":  -1,
	"max-height": -1,
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
		{"OpenCemerWeights", ki.Props{
			"desc": "open network weights from CEmer-format file",
			"icon": "file-open",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) RunEnd() {
	//ss.LogRun(ss.RunLog)
}

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.SeqParams.Nm
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
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
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.RunCt.Cur, ss.TrainEnv.EpochCt.Cur) + ".wts.gz"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(ev *PVLVEnv) {
	//row := dt.Rows
	//dt.SetNumRows(row + 1)
	//
	//epc := ss.TrainEnv.EpochCt.Prv         // this is triggered by increment so use previous value
	//nt := float64(ss.TrainEnv.AlphaCycle.Max) // number of trials in view
	//
	//ss.EpcDA = ss.SumDA / nt
	//ss.SumDA = 0
	//ss.EpcAbsDA = ss.SumAbsDA / nt
	//ss.SumAbsDA = 0
	//ss.EpcRewPred = ss.SumRewPred / nt
	//ss.SumRewPred = 0
	//ss.EpcSSE = ss.SumSSE / nt
	//ss.SumSSE = 0
	//ss.EpcAvgSSE = ss.SumAvgSSE / nt
	//ss.SumAvgSSE = 0
	//ss.EpcPctErr = float64(ss.SumErr) / nt
	//ss.SumErr = 0
	//ss.EpcPctCor = 1 - ss.EpcPctErr
	//ss.EpcCosDiff = ss.SumCosDiff / nt
	//ss.SumCosDiff = 0
	//if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
	//	ss.FirstZero = epc
	//}
	//if ss.EpcPctErr == 0 {
	//	ss.NZero++
	//} else {
	//	ss.NZero = 0
	//}
	//
	//if ss.LastEpcTime.IsZero() {
	//	ss.EpcPerTrlMSec = 0
	//} else {
	//	iv := time.Now().Sub(ss.LastEpcTime)
	//	ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	//}
	//ss.LastEpcTime = time.Now()
	//
	//dt.SetCellFloat("Run", row, float64(ss.TrainEnv.RunCt.Cur))
	//dt.SetCellFloat("Epoch", row, float64(epc))
	//dt.SetCellFloat("SSE", row, ss.EpcSSE)
	//dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
	//dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	//dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	//dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	//dt.SetCellFloat("DA", row, ss.EpcDA)
	//dt.SetCellFloat("AbsDA", row, ss.EpcAbsDA)
	//dt.SetCellFloat("RewPred", row, ss.EpcRewPred)
	//dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)
	//
	//// note: essential to use Go version of update when called from another goroutine
	//ss.TrnEpcPlot.GoUpdate()
	//if ss.TrnEpcFile != nil {
	//	if ss.TrainEnv.RunCt.Cur == 0 && epc == 0 {
	//		dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
	//	}
	//	dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	//}
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
		{"DA", etensor.FLOAT64, nil, nil},
		{"AbsDA", etensor.FLOAT64, nil, nil},
		{"RewPred", etensor.FLOAT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SIR Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DA", eplot.Off, eplot.FixMin, -1, eplot.FixMax, 1)
	plt.SetColParams("AbsDA", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("RewPred", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	//epc := ss.TestEnv.EpochCt.Prv // this is triggered by increment so use previous value

	trl := ss.TrainEnv.AlphaCycle.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	//dt.SetCellFloat("Run", row, float64(ss.TestEnv.Run.Cur))
	//dt.SetCellFloat("Epoch", row, float64(epc))
	//dt.SetCellFloat("Trial", row, float64(trl))
	//dt.SetCellString("TrialName", row, ss.TestEnv.String())
	//dt.SetCellFloat("Err", row, ss.TrlErr)
	//dt.SetCellFloat("SSE", row, ss.TrlSSE)
	//dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
	//dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)
	//dt.SetCellFloat("DA", row, ss.TrlDA)
	//dt.SetCellFloat("AbsDA", row, ss.TrlAbsDA)
	//dt.SetCellFloat("RewPred", row, ss.TrlRewPred)
	//
	//for _, lnm := range ss.TstRecLays {
	//	tsr := ss.ValsTsr(lnm)
	//	ly := ss.Net.LayerByName(lnm).(deep.DeepLayer).AsDeep()
	//	ly.UnitValsTensor(tsr, "ActM")
	//	dt.SetCellTensor(lnm, row, tsr)
	//}

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TrainEnv.AlphaCycle.Max
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"AlphaCycle", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"SSE", etensor.FLOAT64, nil, nil},
		{"AvgSSE", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"DA", etensor.FLOAT64, nil, nil},
		{"AbsDA", etensor.FLOAT64, nil, nil},
		{"RewPred", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		sch = append(sch, etable.Column{Name: lnm, Type: etensor.FLOAT64, CellShape: ly.Shp.Shp})
	}
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Test AlphaCycle Plot"
	plt.Params.XAxisCol = "AlphaCycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AlphaCycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Err", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("DA", eplot.On, eplot.FixMin, -1, eplot.FixMax, 1)
	plt.SetColParams("AbsDA", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("RewPred", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.TstRecLays {
		plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// trl := ss.TstTrlLog
	// tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.EpochCt.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.RunCt.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	dt.SetFromSchema(etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
	}, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SIR Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	//run := ss.TrainEnv.RunCt.Cur // this is NOT triggered by increment yet -- use Cur
	//row := dt.Rows
	//dt.SetNumRows(row + 1)
	//
	//epclog := ss.TrnEpcLog
	//epcix := etable.NewIdxView(epclog)
	//// compute mean over last N epochs for run level
	//nlast := 1
	//if nlast > epcix.Len()-1 {
	//	nlast = epcix.Len() - 1
	//}
	//epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]
	//
	//params := "Std"
	//// if ss.AvgLGain != 2.5 {
	//// 	params += fmt.Sprintf("_AvgLGain=%v", ss.AvgLGain)
	//// }
	//// if ss.InputNoise != 0 {
	//// 	params += fmt.Sprintf("_InVar=%v", ss.InputNoise)
	//// }
	//
	//dt.SetCellFloat("Run", row, float64(run))
	//dt.SetCellString("Params", row, params)
	////dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	//dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
	//dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
	//dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	//dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	//dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])
	//
	//runix := etable.NewIdxView(dt)
	//spl := split.GroupBy(runix, []string{"Params"})
	//split.Desc(spl, "FirstZero")
	//split.Desc(spl, "PctCor")
	//ss.RunStats = spl.AggsToTable(etable.AddAggName)
	//
	//// note: essential to use Go version of update when called from another goroutine
	//ss.RunPlot.GoUpdate()
	//if ss.RunFile != nil {
	//	if row == 0 {
	//		dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
	//	}
	//	dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	//}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

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

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "SIR Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	plt.SetColParams("SSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

func (ss *Sim) SetRunSeqParams() error {
	var err error = nil
	if ss.SeqParams == nil || ss.SeqParamsName != ss.SeqParams.Nm {
		oldSeqParams := ss.SeqParams
		newSeqParams, found := ss.GetRunSeqParams(ss.SeqParamsName)
		if !found {
			err = errors.New(fmt.Sprintf("RunSeq \"%v\" was not found!", ss.SeqParamsName))
			fmt.Println(err)
			return err
		} else {
			ss.SeqParams = newSeqParams
			newRunParams, found := ss.GetRunParams(ss.SeqParams.SeqStep1)
			if !found {
				err = errors.New(fmt.Sprintf("SeqParams step 1 \"%v\" was not found!", ss.SeqParams.SeqStep1))
				gi.PromptDialog(nil, gi.DlgOpts{Title: "SeqParams step not found", Prompt: err.Error()}, gi.AddOk, gi.NoCancel, nil, nil)
				ss.SeqParams = oldSeqParams
				return err
			} else {
				ss.RunParams = newRunParams
				return nil
			}
		}
	}
	return nil
}

// InitRun intializes a new run of the model, using the TrainEnv.RunCt counter
// for the new run value
func (ss *Sim) InitRun(ev *PVLVEnv) error {
	err := ss.SetRunSeqParams()
	if err != nil {
		return err
	}
	ev.Init(ss)
	ss.Time.Reset()
	ss.Net.InitActs()
	ss.InitStats()
	ss.ClearCycleData()
	return nil
}

//func (ss *Sim) RunSelectedLevel(ev *PVLVEnv) {
//	ss.StepCounter.Init()
//	ss.StopStepCounter.Init()
//	ss.MidstreamPoint = ss.StepGrain
//	for {
//		switch ss.StepGrain {
//		case MultiRunSequence:
//			ss.MultiRunSequence()
//		case MultiTrial:
//			ss.MultiTrial(ev, false)
//		}
//	}
//}

// MultiRunSequence
// Run the currently selected sequence
func (ss *Sim) TrainMultiRun() bool {
	ss.Net.InitActs()
	allDone := false
	var found bool
	ev := &ss.TrainEnv
	seqSteps := [5]*data.RunParams{}
	seqSteps[0], found = ss.GetRunParams(ss.SeqParams.SeqStep1)
	if !found {
		fmt.Println("SeqStep", ss.SeqParams.SeqStep1, "was not found")
	}
	seqSteps[1], found = ss.GetRunParams(ss.SeqParams.SeqStep2)
	if !found {
		fmt.Println("SeqStep", ss.SeqParams.SeqStep2, "was not found")
	}
	seqSteps[2], found = ss.GetRunParams(ss.SeqParams.SeqStep3)
	if !found {
		fmt.Println("SeqStep", ss.SeqParams.SeqStep3, "was not found")
	}
	seqSteps[3], found = ss.GetRunParams(ss.SeqParams.SeqStep4)
	if !found {
		fmt.Println("SeqStep", ss.SeqParams.SeqStep4, "was not found")
	}
	seqSteps[4], found = ss.GetRunParams(ss.SeqParams.SeqStep5)
	if !found {
		fmt.Println("SeqStep", ss.SeqParams.SeqStep5, "was not found")
	}
	activateStep := func(i int, runParams *data.RunParams) {
		ev.CurRunParams = runParams
		ss.RunParams = ev.CurRunParams
	}
	for i, seqStep := range seqSteps {
		activateStep(i, seqStep)
		if ev.CurRunParams.Nm == "NullStep" {
			allDone = true
			break
		}
		ss.TrainMultiGroup(ev, true)
		if allDone || ss.Stepper.StopRequested() || ss.Stopped() {
			break
		}
		if ss.TrainUpdt >= leabra.Run {
			ss.UpdateView(true)
		}
	}
	ss.Stepper.Enter(stepper.Stopped)
	return allDone
}

// end MultiRunSequence

// MultiTrial
func (ss *Sim) TrainMultiGroup(ev *PVLVEnv, seqRun bool) {
	if !seqRun {
		ev.CurRunParams = ss.RunParams
	}
	nDone := 0
	ev.TrialInstances = new(TrialInstanceList).Init()
	ev.TrialCt.Init()
	for i := 0; i < ev.CurRunParams.TrainEpochs; i++ {
		ev.RunOneEpoch(ss)
		nDone++
	}
}

// end MultiTrial

//SingleTrial
func (ss *Sim) MasterRun(ev *PVLVEnv) {
	ss.TrainMultiGroup(ev, false)
}

type SimState struct {
	ss *Sim
	ev *PVLVEnv
}

func CheckStopCondition(st interface{}, _ int) bool {
	ss := st.(SimState).ss
	ev := st.(SimState).ev
	ret := false
	switch ss.StopStepCondition {
	case SSNone:
		return false
	case SSError:
		ret = ss.SumSSE > 0.0
	case SSCorrect:
		ret = ss.SumSSE == 0.0
	case SSTrialNameMatch:
		ret = strings.Contains(ev.AlphaTrialName, ss.StopConditionTrialNameString)
	case SSTrialNameNonmatch:
		ret = !strings.Contains(ev.AlphaTrialName, ss.StopConditionTrialNameString)
	default:
		ret = false
	}
	return ret
}

func NotifyPause(st interface{}) {
	ss := st.(SimState).ss
	if int(ss.StepGrain) != ss.Stepper.Grain() {
		ss.Stepper.SetStepGrain(int(ss.StepGrain))
	}
	if ss.StepsToRun != ss.OrigSteps {
		ss.Stepper.SetNSteps(ss.StepsToRun)
		ss.OrigSteps = ss.StepsToRun
	}
	ss.ToolBar.UpdateActions()
	ss.UpdateView(true)
	ss.Win.Viewport.SetNeedsFullRender()
}

// end Epoch and functions

// Monitors //

// SaveLogFiles and functions
func (ss *Sim) SaveLogFiles() {
	//if ss.TrnEpcLog != nil {
	//	_ = ss.TrnEpcLog.SaveCSV("BVPVLVEpochLog.csv", etable.Delims(etable.Comma), true)
	//}
	/* globals added to hardvars:
	Program::RunState run_state; // our program's run state
	int ret_val;
	// args: global script parameters (arguments)
	LeabraNetwork* network;
	// vars: global (non-parameter) variables
	//String tag;
	//bool log_trials;
	//String log_dir;
	//String log_file_nm;
	//DataTable* epoch_output_data;
	//DataTable* trial_output_data;
	//DataTable* all_trial_data;
	*/
	// tag = _pos_cond_inhib_PVLVMaster (String) -- init from MasterStartup
	// log_trials = false
	// log_dir = ""
	// log_file_nm = ""
	// epoch_output_data = ss.AnalysisData.EpochOutputData
	// ss.AnalysisData.TrialOutputData
	// ss.AnalysisData.AllTrialData
	// vars: global (non-parameter) variables

	//if ss.AnalysisData.EpochOutputData. {
	//	return
	//}
	//logFileNm := ss.AnalysisData.EpochOutputData.SaveDataLog(".trn_epc.dat")
}

// end SaveLogFiles functions

// TrialAnalysis and its functions
func (ss *Sim) TrialAnalysis() {
	//if !ss.Interactive {
	//	ss.AllTrialData.ResetData()
	//	ss.FirstRun = true
	//	ss.FirstTimeRunLog = true
	//}
	//ss.AllTrialData.ClearDataFlag()
	//if ss.existingDataAnalysis {
	//	ss.AnalyzeTicksExistingData(ev)
	//} else {
	//	if ss.Interactive {
	//		ss.GetNewData(ev)
	//	}
	//	if ss.DoAnalysis {
	//		ss.AnalyzeTicks(ev)
	//	}
	//}
}

func (ss *Sim) TrialMonitor() {

}

func (ss *Sim) GetNewData() {
	//trlRows := ss.AnalysisData.TrialOutputData.Rows
	//oldRows := 0
	//if ss.AnalysisData.AllTrialData.Rows == 0 {
	//	ss.AnalysisData.AllTrialData.Cop
	//}
}

func (ss *Sim) GetExistingData(ev *PVLVEnv) {

}

func (ss *Sim) ConfigGroupSpec(ev *PVLVEnv) {

}

func (ss *Sim) AnalyzeTicks(ev *PVLVEnv) {

}

func (ss *Sim) AnalyzeTicksExistingData(ev *PVLVEnv) {

}

//func (ss *Sim) InitTrialData() {
//	dt := ss.TrialTypeData
//	dt.SetCellString("TrialName",1, "trial name")
//}

//type Strings []string
//
//func (ss Strings) Less(i, j int) bool {
//	return ss[i] < ss[j ]
//}
//func (ss Strings) Swap(i, j int) {
//	ss[i], ss[j] = ss[j], ss[i]
//}
//func (ss Strings) Len() int {
//	return len(ss)
//}

func (ss *Sim) SetTrialTypeDataXLabels(ev *PVLVEnv) {
	tgNmMap := map[string]string{}
	ev.TrialInstances.Reset()
	ticks := 0
	for !ev.TrialInstances.AtEnd() {
		tg := ev.TrialInstances.ReadNext()
		//name := tg.TrialName + "_" + tg.ValenceContext.String() + "_t"
		name := tg.TrialName + "_t"
		if ticks == 0 {
			ticks = tg.AlphaTicksPerTrialGp
		}
		tgNmMap[name] = name
	}
	ev.TrialInstances.Reset()
	names := sort.StringSlice{}
	//names := []string{}
	for _, val := range tgNmMap {
		names = append(names, val)
	}
	sort.Sort(names)
	dt := ss.TrialTypeData
	dt.SetNumRows(0)
	ss.TrialTypeSet = map[string]int{}
	//dt.SetNumRows(names.Len() * ticks)
	dt.SetNumRows(len(names) * ticks)
	for i, name := range names {
		for j := 0; j < ticks; j++ {
			row := i*ticks + j
			fullName := name + strconv.Itoa(j)
			ss.TrialTypeSet[fullName] = row
			dt.SetCellString("TrialType", row, fullName)
		}
	}
	//ss.TrialTypeSetCounter = names.Len() * ticks
	ss.TrialTypeSetCounter = len(names) * ticks
}

func (ss *Sim) LogTrialTypeData(ev *PVLVEnv) {
	dt := ss.TrialTypeData
	//epc := ev.EpochCt.Cur
	//row := ss.TrialTypeSet[ev.PrevAlphaTrialName]
	row, ok := ss.TrialTypeSet[ev.AlphaTrialName]
	if !ok {
		row = ss.TrialTypeSetCounter
		ss.TrialTypeSetCounter++
		ss.TrialTypeSet[ev.AlphaTrialName] = ss.TrialTypeSetCounter
		if dt.Rows <= row {
			dt.SetNumRows(row + 1)
		}
		dt.SetCellString("TrialType", row, ev.AlphaTrialName)
	}
	for _, colNm := range dt.ColNames {
		if colNm != "TrialType" {
			parts := strings.Split(colNm, "_")
			lnm := parts[0]
			if parts[1] != "act" {

			}
			tsr := ss.ValsTsr(lnm)
			ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
			err := ly.UnitValsTensor(tsr, "Act") // get minus phase act
			if err == nil {
				dt.SetCellTensor(colNm, row, tsr)
			} else {
				fmt.Println(err)
			}
		}
	}
	ss.TrialTypeDataPlot.GoUpdate()
}

func GetLeabraMonitorVal(ly *leabra.Layer, data []string) float64 {
	var val float32
	var err error
	var varIdx int
	valType := data[0]
	varIdx, err = pvlv.NeuronVarByName(valType)
	if err != nil {
		varIdx, err = leabra.NeuronVarIdxByName(valType)
		if err != nil {
			fmt.Printf("index lookup failed for %v_%v_%v_%v: \n", ly.Name(), data[1], valType, err)
		}
	}
	unitIdx, err := strconv.Atoi(data[1])
	if err != nil {
		fmt.Printf("string to int conversion failed for %v_%v_%v%v: \n", ly.Name(), data[1], valType, err)
	}
	//switch valType {
	//case "Act":
	//	val = ly.Neurons[unitIdx].Act
	//default:
	val = ly.UnitVal1D(varIdx, unitIdx)
	//}
	return float64(val)
}

func (ss *Sim) ClearCycleData() {
	for i := 0; i < ss.CycleOutputData.Rows; i++ {
		for _, colName := range ss.CycleOutputData.ColNames {
			ss.CycleOutputData.SetCellFloat(colName, i, 0)
		}
	}
}

func (ss *Sim) LogCycleData(ev *PVLVEnv) {
	var val float64
	dt := ss.CycleOutputData
	row := ev.GlobalStep
	alphaStep := ss.Time.Cycle + ev.AlphaCycle.Cur*100
	for _, colNm := range dt.ColNames {
		if colNm == "GlobalStep" {
			dt.SetCellFloat("GlobalStep", row, float64(ev.GlobalStep))
		} else if colNm == "Cycle" {
			dt.SetCellFloat(colNm, row, float64(alphaStep))
		} else {
			monData := ss.CycleOutputMetadata[colNm]
			parts := strings.Split(colNm, "_")
			lnm := parts[0]
			ly := ss.Net.LayerByName(lnm)
			switch ly.(type) {
			case *leabra.Layer:
				val = GetLeabraMonitorVal(ly.(*leabra.Layer), monData)
			default:
				val = ly.(MonitorVal).GetMonitorVal(monData)
			}
			//lly := ly.(leabra.LeabraLayer).AsLeabra()
			dt.SetCellFloat(colNm, row, val)
		}
	}
	label := fmt.Sprintf("%20s: %3d", ev.AlphaTrialName, row)
	ss.CycleDataPlot.Params.XAxisLabel = label
	if ss.CycleUpdateNetView || row%25 == 0 {
		ss.CycleDataPlot.GoUpdate()
	}
}

func (ss *Sim) TimeAggTickData(ev *PVLVEnv) {
}

// end TrialAnalysis functions

func (ss *Sim) EpochMonitor(ev *PVLVEnv) {
	ss.LogTrnEpc(ev)
}

//func (ss *Sim) Train() {
//	ev := &ss.TrainEnv
//	ss.StopNow = false
//	for {
//		ss.TrainTrial(ev)
//		if ev.TrialNameStopTest(ss) || ss.StopNow {
//			ss.Stop()
//			fmt.Println("STOPPING")
//			break
//		}
//	}
//}

func (ss *Sim) TrialStats(accum bool) {
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
}

func (ss *Sim) CmdArgs() (verbose, threads bool) {
	var nogui bool
	//var verbose bool
	var saveEpcLog bool
	var saveRunLog bool
	//var threads bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&nogui, "nogui", false, "if not passing any other args and want to run nogui, use nogui")
	flag.BoolVar(&verbose, "verbose", false, "give more feedback during initialization")
	flag.BoolVar(&threads, "threads", false, "use per-layer threads")
	flag.Parse()

	if !nogui {
		return verbose, threads
	}

	ss.NoGui = nogui
	ss.InitSim(&ss.TrainEnv)

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	//if ss.ParamSet != "" {
	//	fmt.Printf("Using ParamSet: %s\n", ss.Params)
	//}

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
	ss.TrainMultiRun()
	return verbose, threads
}

func (ss *Sim) GetEnvParams(nm string) (*data.EnvParams, bool) {
	groups, ok := ss.MasterEnvParams[nm]
	ret := new(data.EnvParams).New(groups)
	return ret, ok
}

func (ev *PVLVEnv) GetEpochTrial(n int) *data.Trial {
	ret := (*ev.EpochParamsList.Trials)[n]
	return ret
}

func (ss *Sim) GetRunConfig(nm string) data.RunConfig {
	ret := ss.MasterRunConfigs[nm]
	return ret
}

func (ss *Sim) GetRunParams(nm string) (*data.RunParams, bool) {
	ret, found := ss.MasterRunParams[nm]
	return &ret, found
}

func (ss *Sim) GetRunSeqParams(nm string) (*data.RunSeqParams, bool) {
	ret, found := ss.MasterRunSeqParams[nm]
	return &ret, found
}
