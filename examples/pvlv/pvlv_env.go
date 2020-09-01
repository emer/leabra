// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	_ "fmt"
	_ "github.com/alecthomas/assert"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/params"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/examples/pvlv/data"
	rlist "github.com/emer/leabra/examples/pvlv/recordlist"
	"github.com/emer/leabra/pvlv"
	"github.com/goki/ki/kit"
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"
)

type PVLVEnv struct {
	Nm              string              `inactive:"+" desc:"name of this environment"`
	Dsc             string              `inactive:"+" desc:"description of this environment"`
	PVLVParams      *params.Params      `desc:"PVLV-specific params"`
	GlobalStep      int                 `desc:"cycle counter, cleared by Init, otherwise increments on every Cycle"`
	MultiRunCt      env.Ctr             `inactive:"+" view:"inline" desc:"top-level counter for multi-run sequence"`
	RunCt           env.Ctr             `inactive:"+" view:"inline" desc:"top-level counter for multi-trial group run"`
	EpochCt         env.Ctr             `inactive:"+" view:"inline" desc:"epoch within a run"`
	TrialCt         env.Ctr             `inactive:"+" view:"inline" desc:"trial group within a set of trial groups"`
	AlphaCycle      env.Ctr             `inactive:"+" view:"inline" desc:"step within a trial"`
	AlphaTrialName  string              `inactive:"+" desc:"name of current alpha trial step"`
	USTimeInStr     string              `inactive:"+" desc:"decoded value of USTimeIn"`
	EpochParamsList *data.EnvParams     `desc:"AKA trial group list. A set of trial groups to be run together"`
	TrialInstances  *TrialInstanceList  `view:"no-inline" desc:"instantiated trial groups, further unpacked into StdInputData from this"`
	StdInputData    *etable.Table       `desc:"Completely instantiated input data for a single epoch"`
	ContextModel    ContextModel        `inactive:"+" desc:"One at a time, conjunctive, or a mix"`
	SeqRun          bool                `view:"-" desc:"running from a top-level sequence?"`
	CurRunParams    *data.RunParams     `view:"-" desc:"params for currently executing run, whether from selection or sequence"`
	TrialsPerEpoch  int                 `inactive:"+"`
	DataLoopOrder   rlist.DataLoopOrder `inactive:"+"`
	EpochEnded      bool                `view:"-"`

	// Input data tensors
	TsrStimIn    etensor.Float64
	TsrPosPV     etensor.Float64
	TsrNegPV     etensor.Float64
	TsrContextIn etensor.Float64
	TsrUSTimeIn  etensor.Float64

	NormContextTotalAct   bool    `view:"-" `                                                       // TODO UNUSED if true, clamp ContextIn units as 1/n_context_units - reflecting mutual competition
	NormStimTotalAct      bool    `view:"-" `                                                       // TODO UNUSED if true, clamp StimIn units as 1/n_context_units - reflecting mutual competition
	NormUSTimeTotalAct    bool    `view:"-" `                                                       // TODO UNUSED if true, clamp USTimeIn units as 1/n_context_units - reflecting mutual competition
	PctNormTotalActStim   float64 `desc:"amount to add to denominator for StimIn normalization"`    // used in InstantiateEpochTrials and SetRowStdInputDataAlphTrial
	PctNormTotalActCtx    float64 `desc:"amount to add to denominator for ContextIn normalization"` // used in InstantiateEpochTrials and SetRowStdInputDataAlphTrial
	PctNormTotalActUSTime float64 `desc:"amount to add to denominator for USTimeIn normalization"`  // used in InstantiateEpochTrials and SetRowStdInputDataAlphTrial

	InputShapes *map[string][]int
}

// A set of trial groups, sourced from an EnvParams list, instantiated according to the
// PercentOfTotal field in the source list.
// this is what we get after calling SetEpochTrialList
// Still not fully instantiated, US is still a probability
type TrialInstance struct {
	TrialName            string
	ValenceContext       pvlv.Valence
	USFlag               bool
	TestFlag             bool
	MixedUS              bool
	USProb               float64
	USMagnitude          float64
	AlphaTicksPerTrialGp int
	CS                   string
	CSTimeStart          int
	CSTimeEnd            int
	CS2TimeStart         int
	CS2TimeEnd           int
	USTimeStart          int
	USTimeEnd            int
	Context              string
	USType               string
}

type TrialInstanceList struct {
	Trials []TrialInstance
	Index  []int
	INext  int
	NRead  int
	Order  rlist.DataLoopOrder
}

func (tgl *TrialInstanceList) New(tgs []TrialInstance) *TrialInstanceList {
	tgl.Trials = tgs
	tgl.Index = data.IntSequence(0, len(tgs), 1)
	tgl.INext = -1
	tgl.NRead = 0
	return tgl
}
func (tgl *TrialInstanceList) Init() *TrialInstanceList {
	tgl.INext = -1
	tgl.NRead = 0
	tgl.Index = []int{}
	tgl.Trials = []TrialInstance{}
	return tgl
}
func (tgl *TrialInstanceList) SetIndex(ix []int) error {
	if len(ix) > len(tgl.Trials) {
		return errors.New("supplied index is longer than data")
	}
	tgl.Index = ix
	tgl.Reset()
	return nil
}
func (tgl *TrialInstanceList) SetOrder(order rlist.DataLoopOrder) {
	tgl.Order = order
	if order == rlist.SEQUENTIAL {
		tgl.Sequential()
	} else if order == rlist.PERMUTED {
		tgl.Permute()
	}
	tgl.Reset()
}
func (tgl *TrialInstanceList) Permute() {
	tgl.Index = rand.Perm(len(tgl.Trials))
}
func (tgl *TrialInstanceList) Sequential() {
	tgl.Index = data.IntSequence(0, len(tgl.Trials), 1)
}
func (tgl *TrialInstanceList) GetIndex() []int {
	return tgl.Index
}
func (tgl *TrialInstanceList) WriteNext(tgi TrialInstance) {
	tgl.Trials = append(tgl.Trials, tgi)
	tgl.Index = append(tgl.Index, len(tgl.Trials)-1)
}
func (tgl *TrialInstanceList) ReadNext() *TrialInstance {
	tgl.NRead++
	if tgl.Order == rlist.RANDOM {
		return &tgl.Trials[rand.Intn(len(tgl.Trials))]
	} else {
		tgl.INext++
		return &tgl.Trials[tgl.Index[tgl.INext]]
	}
}
func (tgl *TrialInstanceList) Peek() *TrialInstance {
	if tgl.AtEnd() {
		return nil
	}
	return &tgl.Trials[tgl.Index[tgl.INext+1]]
}
func (tgl *TrialInstanceList) AtEnd() bool {
	return tgl.NRead >= len(tgl.Trials)
}
func (tgl *TrialInstanceList) Reset() {
	tgl.NRead = 0
	tgl.INext = -1
}
func (tgl *TrialInstanceList) SetPos(i int) {
	tgl.INext = i - 1
}
func (tgl *TrialInstanceList) Length() int {
	return len(tgl.Trials)
}
func (tgl *TrialInstanceList) Cur() int {
	return tgl.INext + 1
}

// A single step in a trial, fully instantiated, the last intermediate step before writing to
// StdInputData
//type TrialTickParams struct {
//	AlphTrialName      string
//	TrialGpTimestep    Tick
//	TrialGpTimestepInt int
//	ContextIn          Context
//	ContextIn2         Context
//	ContextIn3         Context
//	NContexts          int
//	StimIn             Stim
//	StimIn2            Stim
//	NStims             int
//	CS1                string
//	CS2                string
//	US                 US
//	PosPV              US
//	NegPV              US
//	USTimeIn           PackedUSTimeState
//	USTimeIn2          PackedUSTimeState
//	NUSTimes           int
//	USTimeInStr        string
//	USTimeIn2Str       string
//	USTimeInWrong      PackedUSTimeState
//	USTimeInWrongStr   string
//	NotUSTimeIn        PackedUSTimeState
//	TrialInstance *TrialInstance
//}

func (ev *PVLVEnv) Name() string { return ev.Nm }
func (ev *PVLVEnv) Desc() string { return ev.Dsc }

func (ev *PVLVEnv) New(ss *Sim) {
	ev.CurRunParams = ss.RunParams
	ev.RunCt.Scale = env.Run
	ev.EpochCt.Scale = env.Epoch
	ev.AlphaCycle.Scale = env.Trial
	ev.InputShapes = &ss.InputShapes
	ev.ContextModel = ELEMENTAL // lives in MiscParams in cemer
	ev.AlphaCycle.Init()
	ev.StdInputData = &etable.Table{}
	ev.ConfigStdInputData(ev.StdInputData)
	ev.AlphaTrialName = "trialname"
	ev.TsrStimIn.SetShape(ss.InputShapes["StimIn"], nil, nil)
	ev.TsrContextIn.SetShape(ss.InputShapes["ContextIn"], nil, nil)
	ev.TsrUSTimeIn.SetShape(ss.InputShapes["USTimeIn"], nil, nil)
	ev.TsrPosPV.SetShape(ss.InputShapes["PosPV"], nil, nil)
	ev.TsrNegPV.SetShape(ss.InputShapes["NegPV"], nil, nil)
}

// From looking at the running cemer model, the chain is as follows:
// RunSeqParams(=pos_acq) -> RunParams(=pos_acq_b50) -> PVLVEnv.vars["env_params_table"](=PosAcq_B50)
// RunSeqParams fields: seq_step_1...5 from MultiRunSequence.vars
// RunParams fields: env_params_table, fixed_prob, ... lrs_bump_step, n_batches, batch_start, load_exp, pain_exp
// Trial fields: trial_gp_name, percent_of_total, ...
func (ev *PVLVEnv) Init(ss *Sim) (ok bool) {
	ev.CurRunParams = ss.RunParams
	ev.EpochParamsList, ok = ss.GetEnvParams(ev.CurRunParams.EnvParamsTable)
	if !ok {
		fmt.Printf("EnvParams lookup failed for %v\n", ev.CurRunParams.EnvParamsTable)
		return ok
	}
	ev.GlobalStep = 0
	ev.RunCt.Init()
	ev.RunCt.Max = ss.MaxRuns
	ev.EpochCt.Init()
	ev.EpochCt.Max = ev.CurRunParams.TrainEpochs
	ev.TrialCt.Init()
	if ev.CurRunParams.UseTrialGp {
		ev.TrialCt.Max = ev.CurRunParams.TrialGpsPerEpoch
	} else {
		ev.TrialCt.Max = ev.CurRunParams.TrialsPerEpoch
	}
	ev.AlphaCycle.Init()
	ev.ContextModel = ELEMENTAL // lives in MiscParams in cemer
	ev.TrialInstances = new(TrialInstanceList).Init()
	return ok
}

func (ev *PVLVEnv) ConfigStdInputData(dt *etable.Table) {
	dt.SetMetaData("name", "StdInputData")
	dt.SetMetaData("desc", "input data")
	dt.SetMetaData("precision", "6")
	shapes := *ev.InputShapes
	sch := etable.Schema{
		{"AlphTrialName", etensor.STRING, nil, nil},
		{"Stimulus", etensor.STRING, nil, nil},
		{"Time", etensor.STRING, nil, nil},
		{"Context", etensor.STRING, nil, nil},
		{"USTimeInStr", etensor.STRING, nil, nil},
		{"PosPV", etensor.FLOAT64, shapes["PosPV"], nil},
		{"NegPV", etensor.FLOAT64, shapes["NegPV"], nil},
		{"StimIn", etensor.FLOAT64, shapes["StimIn"], nil},
		{"ContextIn", etensor.FLOAT64, shapes["ContextIn"], nil},
		{"USTimeIn", etensor.FLOAT64, shapes["USTimeIn"], nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ev *PVLVEnv) Defaults() {
	// how much to decrease clamped input activity values when multiple stimuli present (due to presumed mutual competition), e.g., for AX trials: 0 = not at all; 1 = full activity normalization
	//ev.NormProb = 0.5

}

func (ev *PVLVEnv) Validate() error {
	// TODO implement this
	//if ev.N <= 0 {
	//	return fmt.Errorf("BanditEnv: %v N == 0 -- must set with SetN call", ev.TrialName)
	//}
	return nil
}

// EpochStart
func (ev *PVLVEnv) EpochStart(ss *Sim) {
	ev.AlphaCycle.Init()
	ev.TrialInstances = new(TrialInstanceList).Init()
	ev.TrialCt.Init()
	ss.Net.ThrTimerReset()
	// TODO implement ev.ResetTrialMonData()
	// TODO implement ev.ResetTrainTime()
}

// end EpochStart (no internal functions)

// EpochEnd
func (ev *PVLVEnv) EpochEnd(ss *Sim) {
	ss.TrialAnalysis()
	ss.EpochMonitor(ev)
	if ev.EpochCt.Cur%ev.CurRunParams.SaveWtsInterval == 0 && ev.EpochCt.Cur > 0 {
		ev.SaveWeights(ss)
	}
}

// end EpochEnd

// SaveWeights
func (ev *PVLVEnv) SaveWeights(_ *Sim) {
	// TODO implement SaveWeights
}

// end SaveWeights

// SaveOutputData
func (ev *PVLVEnv) SaveOutputData(_ *Sim) {
	// TODO implement SaveOutputData (BIG HAIRY THING)
}

// end SaveOutputData

// ContextModel
type ContextModel int

const (
	ELEMENTAL ContextModel = iota
	CONJUNCTIVE
	BOTH
	ContextModelN
)

var KiT_ContextModel = kit.Enums.AddEnum(ContextModelN, kit.NotBitFlag, nil)

func ReadParamsYaml(fn string) *params.Sets {
	var result params.Sets
	b, err := ioutil.ReadFile(fn)
	if err != nil {
		panic(err)
	}
	err = yaml.Unmarshal(b, &result)
	return &result
}

func (ev *PVLVEnv) Counters() []env.TimeScales {
	return []env.TimeScales{env.Run, env.Epoch, env.Trial}
}

func (ev *PVLVEnv) Actions() env.Elements {
	return nil
}

func (ev *PVLVEnv) States() env.Elements {
	// one-hot representations for each component
	shapes := *ev.InputShapes
	els := env.Elements{
		{"StimIn", shapes["StimIn"], nil}, //[]string{"N"}},
		{"ContextIn", shapes["ContextIn"], []string{"Ctx", "Time"}},
		{"USTimeIn", shapes["USTimeIn"],
			[]string{"CS", "Valence", "Time", "US"}},
		{"PosPV", shapes["PosPV"], []string{"PV"}},
		{"NegPV", shapes["NegPV"], []string{"PV"}},
	}
	return els
}

// SetState sets the input states from ev.StdInputData
func (ev *PVLVEnv) SetState() {
	ev.TsrStimIn.SetZeros()
	ev.TsrStimIn.CopyFrom(ev.StdInputData.CellTensor("StimIn", ev.AlphaCycle.Cur))
	ev.TsrContextIn.SetZeros()
	ev.TsrContextIn.CopyFrom(ev.StdInputData.CellTensor("ContextIn", ev.AlphaCycle.Cur))
	ev.TsrUSTimeIn.SetZeros()
	ev.TsrUSTimeIn.CopyFrom(ev.StdInputData.CellTensor("USTimeIn", ev.AlphaCycle.Cur))
	ev.TsrPosPV.SetZeros()
	ev.TsrPosPV.CopyFrom(ev.StdInputData.CellTensor("PosPV", ev.AlphaCycle.Cur))
	ev.TsrNegPV.SetZeros()
	ev.TsrNegPV.CopyFrom(ev.StdInputData.CellTensor("NegPV", ev.AlphaCycle.Cur))
	ev.AlphaTrialName = ev.StdInputData.CellString("AlphTrialName", ev.AlphaCycle.Cur)
	ev.USTimeInStr = ev.StdInputData.CellString("USTimeInStr", ev.AlphaCycle.Cur)
}

func (ev *PVLVEnv) State(Nm string) etensor.Tensor {
	switch Nm {
	case "StimIn":
		return &ev.TsrStimIn
	case "ContextIn":
		return &ev.TsrContextIn
	case "USTimeIn":
		return &ev.TsrUSTimeIn
	case "PosPV":
		return &ev.TsrPosPV
	case "NegPV":
		return &ev.TsrNegPV
	default:
		return nil
	}
}

func (ev *PVLVEnv) Step() bool {
	ev.EpochCt.Same() // good idea to just reset all non-inner-most counters at start
	if ev.AlphaCycle.Incr() {
		ev.EpochCt.Incr()
	}
	return true
}

func (ev *PVLVEnv) Action(_ string, _ etensor.Tensor) {
	// nop
}

func (ev *PVLVEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.RunCt.Query()
	case env.Epoch:
		return ev.EpochCt.Query()
	case env.Trial:
		return ev.AlphaCycle.Query()
	}
	return -1, -1, false
}

// normalizes PercentOfTotal numbers for trial types within a set
// sets X labels in the TrialTypeData plot from instantiated trial types
// ev.SetTableEpochTrialGpListFmDefnTable to set up a full epoch's worth of trials
// permutes trial order, if specified
func (ev *PVLVEnv) SetEpochTrialList(ss *Sim) {
	pctTotalSum := 0.0

	ev.EpochParamsList.Reset()
	for !ev.EpochParamsList.AtEnd() {
		pctTotalSum += ev.EpochParamsList.ReadNext().PercentOfTotal
	}

	ev.EpochParamsList.Reset()
	for !ev.EpochParamsList.AtEnd() {
		tg := ev.EpochParamsList.ReadNext()
		baseNumber := tg.PercentOfTotal
		normProb := baseNumber / pctTotalSum
		tg.PercentOfTotal = normProb
	}

	ev.SetTableEpochTrialGpListFmDefnTable()
	ss.SetTrialTypeDataXLabels(ev)

	if ev.CurRunParams.PermuteTrialGps {
		ev.TrialInstances.Permute()
	}
}

// turn US probabilities into concrete examples
// reads from ev.EpochParamsList, writes to ev.TrialInstances
func (ev *PVLVEnv) SetTableEpochTrialGpListFmDefnTable() {
	nRepeatsF := 0.0
	nRepeats := 0
	usFlag := false
	exactOmitProportion := false
	exactNOmits := 0
	nOmitCount := 0

	ev.EpochParamsList.SetOrder(ev.DataLoopOrder)
	for !ev.EpochParamsList.AtEnd() {
		curEpochParams := ev.EpochParamsList.ReadNext()

		// using number of eco_trials_per epoch (in lieu of trials_per_epoch)
		if ev.CurRunParams.UseTrialGp {
			nRepeatsF = curEpochParams.PercentOfTotal * float64(ev.CurRunParams.TrialGpsPerEpoch)
		} else {
			nRepeatsF = curEpochParams.PercentOfTotal * float64(ev.CurRunParams.TrialsPerEpoch)
		}
		// fix rounding error from int arithmetic
		nRepeats = int(nRepeatsF + 0.001)
		if nRepeats < 1 && curEpochParams.PercentOfTotal > 0.0 {
			nRepeats = 1
		}
		// should do each at least once (unless user intended 0.0f)
		if ev.CurRunParams.FixedProb || strings.Contains(curEpochParams.TrialGpName, "AutoTstEnv") {
			if curEpochParams.USProb != 0.0 && curEpochParams.USProb != 1.0 {
				exactOmitProportion = true
				exactNOmits = int(math.Round(float64(ev.CurRunParams.TrialGpsPerEpoch) * curEpochParams.PercentOfTotal * (1.0 - curEpochParams.USProb)))
				nOmitCount = 0
			} else {
				exactOmitProportion = false
			}
		}
		for i := 0; i < nRepeats; i++ {
			// TODO:Should prevent making too many rows, but could still make (one?) too few due to rounding errors
			if ev.TrialInstances != nil &&
				(ev.CurRunParams.UseTrialGp && ev.TrialInstances.Length() < ev.CurRunParams.TrialGpsPerEpoch) ||
				(!ev.CurRunParams.UseTrialGp && ev.TrialInstances.Length() < ev.CurRunParams.TrialsPerEpoch) {

				// was SetRow_CurEpoch
				rfFlgTemp := false
				probUSOmit := 0.0
				trialGpName := curEpochParams.TrialGpName + "_" + curEpochParams.ValenceContext.String()
				if !strings.Contains(trialGpName, "NR") { // nonreinforced (NR) trials NEVER get reinforcement
					rfFlgTemp = true // actual Rf can be different each eco_trial
					if !exactOmitProportion {
						probUSOmit = float64(rand.Intn(2))
						if probUSOmit < 1-curEpochParams.USProb {
							rfFlgTemp = false
						}
					} else {
						if nOmitCount < exactNOmits {
							rfFlgTemp = false
							nOmitCount++
						} else {
							rfFlgTemp = true
						}
					}
					trialGpName = strings.TrimSuffix(trialGpName, "_omit")
					// could be repeat of eco trial type - but with different Rf flag
					if rfFlgTemp == false {
						trialGpName += "_omit"
					}
					usFlag = rfFlgTemp
				} else {
					usFlag = false
				}
				trialGpName = strings.TrimSuffix(trialGpName, "_test")
				// could be repeat of eco trial type - but with different test flag
				testFlag := false
				if strings.Contains(strings.ToLower(ev.CurRunParams.EnvParamsTable), "test") {
					trialGpName += "_test"
					testFlag = true // just in case - should be redundant
				} else {
					testFlag = false
				}
				curTrial := TrialInstance{}
				curTrial.TrialName = trialGpName
				curTrial.ValenceContext = curEpochParams.ValenceContext
				curTrial.TestFlag = testFlag
				curTrial.USFlag = usFlag
				curTrial.MixedUS = curEpochParams.MixedUS
				curTrial.USProb = curEpochParams.USProb
				curTrial.USMagnitude = curEpochParams.USMagnitude
				curTrial.USType = curEpochParams.USType.String()
				curTrial.AlphaTicksPerTrialGp = curEpochParams.AlphTicksPerTrialGp
				curTrial.CS = curEpochParams.CS
				curTrial.CSTimeStart = int(curEpochParams.CSTimeStart)
				curTrial.CSTimeEnd = int(curEpochParams.CSTimeEnd)
				curTrial.CS2TimeStart = int(curEpochParams.CS2TimeStart)
				curTrial.CS2TimeEnd = int(curEpochParams.CS2TimeEnd)
				curTrial.USTimeStart = int(curEpochParams.USTimeStart)
				curTrial.USTimeEnd = int(curEpochParams.USTimeEnd)
				curTrial.Context = curEpochParams.Context
				ev.TrialInstances.WriteNext(curTrial)
				// end SetRow_CurEpoch
			}
		}
	}
	ev.EpochParamsList.Sequential() // avoid confusion?
}

func (ev *PVLVEnv) SetupOneAlphaTrial(curTrial *TrialInstance, stimNum int) {
	prefixUSTimeIn := ""

	// CAUTION! - using percent normalization assumes the multiple CSs (e.g., AX) are always on together,
	// i.e., the same timesteps; thus, doesn't work for second-order conditioning
	stimInBase := data.StmNone
	stimIn2Base := data.StmNone
	nStims := 1
	nUSTimes := 1

	// CAUTION! below string-pruning requires particular convention for naming trial_gps in terms of CSs used;
	// e.g., "AX_*", etc.

	// CAUTION! For either multiple CSs (e.g., AX) or mixed_US case
	// (e.g., sometimes reward, sometimes punishment) only two (2) simultaneous representations
	// currently supported; AND, multiple CSs and mixed_US cases can only be used separately, not together;
	// code will need re-write if more complicated cases are desired (e.g., more than two (2) representations
	// or using multiple CSs/mixed_US together).

	cs := curTrial.TrialName[0:2]
	cs1 := ""
	cs2 := ""
	if strings.Contains(cs, "_") {
		cs1 = cs[0:1]
		cs = data.StmNone.String()
		cs2 = data.StmNone.String()
		nStims = 1
		// need one for each predictive CS; also, one for each PREDICTED US if same CS (e.g., Z')
		// predicts two different USs probalistically (i.e., mixed_US == true condition)
		nUSTimes = 1
		stimInBase = data.StimMap[cs1]
	} else {
		cs1 = cs[0:1]
		cs2 = cs[1:2]
		nStims = 2
		// need one for each predictive CS; also, one for each PREDICTED US if same CS (e.g., Z')
		// predicts two different USs probalistically (i.e., mixed_US == true condition)
		nUSTimes = 2
		stimInBase = data.StimMap[cs1]
	}

	// Set up Context_In reps

	// initialize to use the basic context_in var to rep the basic case in which CS and Context are isomorphic
	ctxParts := data.CtxRe.FindStringSubmatch(curTrial.Context)
	ctx1 := ctxParts[1]
	ctx2 := ctxParts[2]
	preContext := ctx1 + ctx2
	postContext := ctxParts[3]
	contextIn := data.CtxMap[curTrial.Context]
	contextIn2 := data.CtxNone
	contextIn3 := data.CtxNone
	nContexts := len(preContext)
	// gets complicated if more than one CS...
	if len(preContext) > 1 {
		switch ev.ContextModel {
		case ELEMENTAL:
			// first element, e.g., A
			contextIn = data.CtxMap[ctx1]
			// second element, e.g., X
			contextIn2 = data.CtxMap[ctx2]
			// only handles two for now...
		case CONJUNCTIVE:
			// use "as is"...
			contextIn = data.CtxMap[curTrial.Context]
			nContexts = 1
		case BOTH:
			// first element, e.g., A
			contextIn = data.CtxMap[ctx1]
			// second element, e.g., X
			contextIn2 = data.CtxMap[ctx2]
			// conjunctive case, e.g., AX
			contextIn3 = data.CtxMap[preContext]
			nContexts = len(preContext) + 1
		}
	}
	// anything after the "_" indicates different context for extinction, renewal, etc.
	if len(postContext) > 0 {
		contextIn = data.CtxMap[ctx1+"_"+postContext]
		if len(ctx2) > 0 {
			contextIn2 = data.CtxMap[ctx2+"_"+postContext]
		}
		contextIn3 = data.CtxNone
	}

	if ev.StdInputData.Rows != 0 {
		ev.StdInputData.SetNumRows(0)
	}

	// configure and write all the leabra trials for one eco trial
	for i := 0; i < curTrial.AlphaTicksPerTrialGp; i++ {
		i := ev.AlphaCycle.Cur
		alphaTrialName := curTrial.TrialName + "_t" + strconv.Itoa(i)
		trialGpTimestep := data.Tick(i)
		trialGpTimestepInt := i
		stimIn := data.StmNone
		stimIn2 := data.StmNone
		posPV := data.PosUSNone
		negPV := data.NegUSNone
		usTimeInStr := ""
		usTimeInWrongStr := ""
		usTimeIn := data.USTimeNone
		usTimeInWrong := data.USTimeNone
		notUSTimeIn := data.USTimeNone
		prefixUSTimeIn = cs1 + "_"
		// set CS input activation values on or off according to timesteps
		// set first CS - may be the only one
		if i >= curTrial.CSTimeStart && i <= curTrial.CSTimeEnd {
			stimIn = stimInBase
			// TODO: Theoretically, USTime reps shouldn't come on at CS-onset until BAacq and/or
			// gets active first - for time being, using a priori inputs as a temporary proof-of-concept
		} else {
			stimIn = data.StmNone
		}
		// set CS2 input activation values on or off according to timesteps, if a second CS exists
		if i >= curTrial.CS2TimeStart && i <= curTrial.CS2TimeEnd {
			stimIn2 = stimIn2Base
		} else {
			stimIn2 = data.StmNone
		}
		// set US and USTime input activation values on or off according to timesteps
		var us int
		if i > curTrial.CSTimeStart && (!(i > curTrial.USTimeStart) || !curTrial.USFlag) {
			if curTrial.ValenceContext == pvlv.POS {
				us = int(data.PosSMap[curTrial.USType])
				posPV = data.PosUS(us)
				usTimeInStr = prefixUSTimeIn + "PosUS" + strconv.Itoa(us) + "_t" +
					strconv.Itoa(i-curTrial.CSTimeStart-1)
				usTimeIn = data.PUSTFromString(usTimeInStr)
				usTimeInWrongStr = data.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "NegUS" + strconv.Itoa(us) + "_t" +
						strconv.Itoa(i-curTrial.CSTimeStart-1)
					usTimeInWrong = data.PUSTFromString(usTimeInWrongStr)
				}
			} else if curTrial.ValenceContext == pvlv.NEG {
				us = int(data.NegSMap[curTrial.USType])
				negPV = data.NegUS(us)
				usTimeInStr = prefixUSTimeIn + "NegUS" + strconv.Itoa(us) + "_t" +
					strconv.Itoa(i-curTrial.CSTimeStart-1)
				usTimeIn = data.PUSTFromString(usTimeInStr)
				usTimeInWrongStr = data.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "PosUS" + strconv.Itoa(us) + "_t" +
						strconv.Itoa(i-curTrial.CSTimeStart-1)
					usTimeInWrong = data.PUSTFromString(usTimeInWrongStr)
				}
			}
		} else {
			usTimeIn = data.USTimeNone
			notUSTimeIn = data.USTimeNone
			usTimeInStr = data.USTimeNone.String()
		}

		if i > curTrial.CS2TimeStart && i <= (curTrial.CS2TimeEnd+1) && (!(i > curTrial.USTimeStart) || !curTrial.USFlag) {
			if curTrial.ValenceContext == pvlv.POS {
				us = int(data.PosSMap[curTrial.USType])
				posPV = data.PosUS(us)
				usTimeInWrongStr = data.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "NegUS" + strconv.Itoa(us) + "_t" +
						strconv.Itoa(i-curTrial.CSTimeStart-1)
					usTimeInWrong = data.USTimeNone.FromString(usTimeInWrongStr)
				}
			} else if curTrial.ValenceContext == pvlv.NEG {
				negPV = data.NegSMap[curTrial.USType]
				us = int(negPV)
				usTimeInWrongStr = data.USTimeNone.String()
				if curTrial.MixedUS {
					usTimeInWrongStr = prefixUSTimeIn + "PosUS" + strconv.Itoa(us) + "_t" +
						strconv.Itoa(i-curTrial.CSTimeStart-1)
					usTimeInWrong = data.USTimeNone.FromString(usTimeInWrongStr)
				}
			}
		} else {
			notUSTimeIn = data.USTimeNone
		}

		if (i >= curTrial.USTimeStart) && (i <= curTrial.USTimeEnd) && curTrial.USFlag {
		} else {
			posPV = data.PosUSNone
			negPV = data.NegUSNone
		}
		if (i > curTrial.USTimeStart) && curTrial.USFlag {
			if curTrial.ValenceContext == pvlv.POS {
				us = int(data.PosSMap[curTrial.USType])
				usTimeInStr = "PosUS" + strconv.Itoa(us) + "_t" + strconv.Itoa(i-curTrial.USTimeStart-1)
				usTimeIn = data.USTimeNone.FromString(usTimeInStr)
				usTimeInWrongStr = data.USTimeNone.String()
				usTimeInWrong = data.USTimeNone
			} else if curTrial.ValenceContext == pvlv.NEG {
				us = int(data.NegSMap[curTrial.USType])
				usTimeInStr = "NegUS" + strconv.Itoa(us) + "_t" + strconv.Itoa(i-curTrial.USTimeStart-1)
				usTimeIn = data.USTimeNone.FromString(usTimeInStr)
				usTimeInWrongStr = data.USTimeNone.String()
				usTimeInWrong = data.USTimeNone
			}
		}
		pvEmpty := data.PosUSNone.Tensor()
		curTimestepStr := ""
		curTimeStepInt := 0
		stimulus := ""
		stimDenom := 1.0
		ctxtDenom := 1.0
		usTimeDenom := 1.0
		curTimeStepInt = trialGpTimestepInt
		curTimestepStr = trialGpTimestep.String()
		if stimNum == 0 {
			ev.StdInputData.AddRows(1)
		}
		if nStims == 1 {
			stimulus = stimIn.String()
		} else {
			stimulus = cs1 + cs2
		} // // i.e., there is a 2nd stimulus, e.g., 'AX', 'BY'

		ev.StdInputData.SetCellString("AlphTrialName", curTimeStepInt, alphaTrialName)
		ev.StdInputData.SetCellString("Time", curTimeStepInt, curTimestepStr)
		ev.StdInputData.SetCellString("Stimulus", curTimeStepInt, stimulus)
		ev.StdInputData.SetCellString("Context", curTimeStepInt, curTrial.Context)

		tsrStim := etensor.NewFloat64(data.StimInShape, nil, nil)
		tsrCtx := etensor.NewFloat64(data.ContextInShape, nil, nil)
		if curTimeStepInt >= curTrial.CSTimeStart && curTimeStepInt <= curTrial.CSTimeEnd {
			stimDenom = 1.0 + ev.PctNormTotalActStim*float64(nStims-1)
			if stimIn != data.StmNone {
				tsrStim.SetFloat([]int{int(stimIn)}, 1.0/stimDenom)
			}
			if stimIn2 != data.StmNone {
				tsrStim.SetFloat([]int{int(stimIn2)}, 1.0/stimDenom)
			}
			ev.StdInputData.SetCellTensor("StimIn", curTimeStepInt, tsrStim)

			ctxtDenom = 1.0 + ev.PctNormTotalActCtx*float64(nContexts-1)
			if contextIn != data.CtxNone {
				tsrCtx.SetFloat(contextIn.Parts(), 1.0/ctxtDenom)
			}
			if contextIn3 != data.CtxNone {
				tsrCtx.SetFloat(contextIn3.Parts(), 1.0/ctxtDenom)
			}
			ev.StdInputData.SetCellTensor("ContextIn", curTimeStepInt, tsrCtx)
		}
		if curTimeStepInt >= curTrial.CS2TimeStart && curTimeStepInt <= curTrial.CS2TimeEnd {
			stimDenom = 1.0 + ev.PctNormTotalActStim*float64(nStims-1)
			if stimIn2 != data.StmNone {
				tsrStim.SetFloat([]int{int(stimIn2)}, 1.0/stimDenom)
			}
			ev.StdInputData.SetCellTensor("StimIn", curTimeStepInt, tsrStim)

			ctxtDenom = 1.0 + ev.PctNormTotalActCtx*float64(nContexts-1)
			if contextIn2 != data.CtxNone {
				tsrCtx.SetFloat(contextIn2.Parts(), 1.0/ctxtDenom)
			}
			if contextIn3 != data.CtxNone {
				tsrCtx.SetFloat(contextIn3.Parts(), 1.0/ctxtDenom)
			}
			ev.StdInputData.SetCellTensor("ContextIn", curTimeStepInt, tsrCtx)
		}

		if curTrial.USFlag && (curTimeStepInt >= curTrial.USTimeStart && curTimeStepInt <= curTrial.USTimeEnd) {
			if curTrial.USFlag && curTrial.ValenceContext == pvlv.POS {
				if posPV != data.PosUSNone {
					ev.StdInputData.SetCellTensor("PosPV", curTimeStepInt, posPV.Tensor())
				} else {
					ev.StdInputData.SetCellTensor("PosPV", curTimeStepInt, pvEmpty)
				}
			} else if curTrial.USFlag && curTrial.ValenceContext == pvlv.NEG {
				if negPV != data.NegUSNone {
					ev.StdInputData.SetCellTensor("NegPV", curTimeStepInt, negPV.Tensor())
				} else {
					ev.StdInputData.SetCellTensor("NegPV", curTimeStepInt, pvEmpty)
				}
			}
		} else {
			ev.StdInputData.SetCellTensor("PosPV", curTimeStepInt, pvEmpty)
			ev.StdInputData.SetCellTensor("NegPV", curTimeStepInt, pvEmpty)
		}

		usTimeDenom = 1.0 + ev.PctNormTotalActUSTime*float64(nUSTimes-1)
		tsrUSTime := etensor.NewFloat64(data.USTimeInShape, nil, nil)
		if usTimeIn != data.USTimeNone {
			setVal := usTimeIn.Unpack().Coords()
			tsrUSTime.SetFloat(setVal, 1.0/usTimeDenom)
		}
		if usTimeInWrong != data.USTimeNone {
			tsrUSTime.SetFloat(usTimeInWrong.Shape(), 1.0/usTimeDenom)
		}
		if notUSTimeIn != data.USTimeNone {
			tsrUSTime.SetFloat(notUSTimeIn.Shape(), 1.0/usTimeDenom)
		}
		ev.StdInputData.SetCellTensor("USTimeIn", curTimeStepInt, tsrUSTime)
		ev.StdInputData.SetCellString("USTimeInStr", curTimeStepInt, usTimeInStr+usTimeIn.Unpack().CoordsString())
	}
}

func (ev *PVLVEnv) IsTestTrial() bool {
	return false
	//testFlag := ev.CurAlphaTrial.TrialInstance.TestFlag
	//eTrlNm := ev.CurAlphaTrial.TrialInstance.TrialName
	//// testing both is an extra safety measure for congruence
	//if testFlag && strings.Contains(strings.ToLower(eTrlNm), "_test") {
	//	return true
	//} else {
	//	if testFlag || strings.Contains(strings.ToLower(eTrlNm), "_test") {
	//		fmt.Println("ERROR: TrialName and TestFlag seem to be incongruent!")
	//	}
	//	return false
	//}
}
