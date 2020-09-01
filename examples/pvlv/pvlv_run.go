// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

//func (ss *Sim) TrainTrial(ev *PVLVEnv) {
//	ss.StopNow = false
//	if ss.NeedsNewRun {
//		ss.InitRun(ev)
//	}
//	ev.Step() // the Env encapsulates and manages all counter state
//
//	// Key to query counters FIRST because current state is in NEXT epoch
//	// if epoch counter has changed
//	epc, _, chg := ev.Counter(env.EpochCt)
//	if chg {
//		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
//			ss.UpdateView(true)
//		}
//		ss.LogTrnEpc(ss.TrnEpcLog)
//		if epc >= ss.MaxEpcs {
//			// done with training..
//			ss.RunEnd()
//			if ev.Run.Incr() { // we are done!
//				ss.StopNow = true
//				return
//			} else {
//				ss.NeedsNewRun = true
//				return
//			}
//		}
//	}
//	ss.ApplyInputs(ev)
//	ss.SettleMinus(true)
//	ss.ApplyInputs(ev)
//	ss.ApplyPVInputs(ev)
//	ss.SettlePlus(true)
//	ss.Net.DWt()
//	ss.TrialStats(true) // accumulate
//}

type StepGrain int

const (
	Cycle StepGrain = iota
	Quarter
	SettleMinus
	SettlePlus
	AlphaCycle
	SGTrial // Trial
	Epoch
	MultiRunSequence
	StepGrainN
)

////go:generate stringer -type=StepGrain -linecomment // moved to stringers.go
var KiT_StepGrain = kit.Enums.AddEnum(StepGrainN, kit.NotBitFlag, nil)

func (ss *Sim) SettleMinus(ev *PVLVEnv, train bool) {
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	for qtr := 0; qtr < 3; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.Net.Cycle(&ss.Time)
			if ss.CycleUpdateNetView {
				ev.GlobalStep++
				ss.LogCycleData(ev)
			}
			ss.Time.CycleInc()
			if ss.Stepper.StepPoint(int(Cycle)) {
				return
			}
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike: // every 10 cycles
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
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
		ss.Time.QuarterInc()
		if !ss.CycleUpdateNetView {
			ev.GlobalStep++
			ss.LogCycleData(ev)
		}
		if ss.Stepper.StepPoint(int(Quarter)) {
			return
		}
	}
}

func (ss *Sim) SettlePlus(ev *PVLVEnv, train bool) {
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
		ss.Net.Cycle(&ss.Time)
		if ss.CycleUpdateNetView {
			ev.GlobalStep++
			ss.LogCycleData(ev)
		}
		ss.Time.CycleInc()
		if ss.Stepper.StepPoint(int(Cycle)) {
			return
		}
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
	ss.Net.QuarterFinal(&ss.Time)
	if ss.ViewOn {
		switch viewUpdt {
		case leabra.Quarter, leabra.Phase:
			ss.UpdateView(train)
		}
	}
	ss.Time.QuarterInc()
	if !ss.CycleUpdateNetView {
		ev.GlobalStep++
		ss.LogCycleData(ev)
	}
	if ss.Stepper.StepPoint(int(Quarter)) {
		return
	}
}

func (ss *Sim) TrialStart(_ *PVLVEnv, train bool) {
	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	//lhb, ok := ss.Net.LayMap["LHbRMTg"].(*LHbRMTgLayer)
	//if !ok {
	//	fmt.Println("LHB type assertion failed")
	//} else {}
	//	fmt.Printf("lhb bef: avgm=%v\n", lhb.Pools[0].ActAvg.ActMAvg)
	if train {
		ss.Net.WtFmDWt()
	}
	ss.Net.AlphaCycInit()
	//fmt.Printf("lhb aft: avgm=%v\n", lhb.Pools[0].ActAvg.ActMAvg)
	ss.Time.AlphaCycStart()
}

func (ss *Sim) TrialEnd(_ *PVLVEnv, train bool) {
	//if train {
	//	ss.Net.DWt()
	//}
	//for li := range ss.Net.Layers {
	//	ly, ok := ss.Net.Layers[li].(*ModLayer)
	//	if ok {
	//		ly.ClearModLevels()
	//	}
	//}
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	if ss.ViewOn && viewUpdt == leabra.Trial {
		ss.UpdateView(train)
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(ev *PVLVEnv) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"StimIn", "ContextIn", "USTimeIn"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := ev.State(ly.Nm)
		//fmt.Println(ly.Nm, pats)
		if pats == nil {
			continue
		}
		ly.ApplyExt(pats)
	}
}

func (ss *Sim) ApplyPVInputs(ev *PVLVEnv) {
	lays := []string{"PosPV", "NegPV"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		pats := ev.State(ly.Nm)
		//fmt.Println(ly.Nm, pats.T())
		if pats == nil {
			continue
		}
		ly.ApplyExt(pats)
	}
}

// SingleTrial and functions -- SingleTrial has been consolidated into this
// An epoch is a set of trials, whose length is set by the current RunParams record
func (ev *PVLVEnv) RunOneEpoch(ss *Sim) {
	epochDone := false
	var curTG *TrialInstance
	ev.SetEpochTrialList(ss) // sets up one epoch's worth of data
	ev.TrialCt.Init()
	//}
	epochDone = ev.TrialCt.Cur >= ev.TrialCt.Max
	for !epochDone {
		//ev.AlphaCycle.Init()
		if ev.TrialInstances.AtEnd() {
			panic(fmt.Sprintf("ran off end of TrialInstances list"))
		}
		curTG = ev.TrialInstances.ReadNext()
		ev.AlphaCycle.Max = curTG.AlphaTicksPerTrialGp
		epochDone = ev.RunOneTrial(ss, curTG) // run one instantiated trial type (aka "trial group")
		if ss.Stepper.StepPoint(int(SGTrial)) {
			return
		}
		if ss.TrainUpdt == leabra.Trial {
			ss.UpdateView(ev == &ss.TrainEnv)
		}
	}
	//if ev.IsTrialGpEnd() {
	//	if ss.TrainUpdt == leabra.Block { ss.UpdateView(ev == &ss.TrainEnv) }
	//	// TODO CLEAR DEEP ACTS
	//	if ev.IsEpochEnd() {
	//		if ev.EpochCt.Cur >= ev.CurRunParams.TrainEpochs || ss.StopStepTest(ev, SingleTrial) {
	//			ev.TrainEnd(ss)
	//			return true
	//		}
	//		ev.EpochStart(ss)
	//	}
	//}
	//ev.TrialCt.Init()
	ev.EpochCt.Incr()
	ev.EpochEnded = true
	ev.EpochEnd(ss) // run monitoring and analysis, maybe save weights
	if ss.Stepper.StepPoint(int(Epoch)) {
		return
	}
	if ss.ViewOn && ss.TrainUpdt >= leabra.Epoch {
		ss.UpdateView(true)
	}
}

// run through a complete trial, consisting of a number of ticks as specified in the Trial spec
func (ev *PVLVEnv) RunOneTrial(ss *Sim, curTrial *TrialInstance) (epochDone bool) {
	var train bool
	trialDone := false
	ss.Net.ClearModActs(&ss.Time)
	for !trialDone {
		ev.SetupOneAlphaTrial(curTrial, 0)
		train = !ev.IsTestTrial()
		ev.RunOneAlphaCycle(ss)
		trialDone = ev.AlphaCycle.Incr()
		if ss.Stepper.StepPoint(int(AlphaCycle)) {
			return
		}
		if ss.TrainUpdt <= leabra.Quarter {
			ss.UpdateView(true)
		}
	}
	ss.Net.ClearMSNTraces(&ss.Time)
	epochDone = ev.TrialCt.Incr()
	ss.TrialEnd(ev, train)
	ss.TrialStats(train) // accumulate
	if ss.TrainUpdt == leabra.Trial {
		ss.UpdateView(true)
	}
	//if ss.Stepper.StepPoint(int(SGTrial)) { return }
	return epochDone
}

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ev *PVLVEnv) RunOneAlphaCycle(ss *Sim) {
	train := !ev.IsTestTrial()
	ss.TrialStart(ev, train)
	ev.SetState()
	//fmt.Println("ev.CurAlphaTrial.AlphTrialName:", ev.CurAlphaTrial.AlphTrialName, ev.AlphaTick)
	ss.ApplyInputs(ev)
	ss.SettleMinus(ev, train)
	if ss.Stepper.StepPoint(int(SettleMinus)) {
		return
	}
	ss.ApplyInputs(ev)
	ss.ApplyPVInputs(ev)
	ss.SettlePlus(ev, train)
	if train {
		ss.Net.DWt()
	}
	ss.LogTrialTypeData(ev)
	_ = ss.Stepper.StepPoint(int(SettlePlus))
}

func (ev *PVLVEnv) IsEpochEnd() bool {
	if ev.CurRunParams.UseTrialGp {
		if ev.TrialCt.Cur >= ev.CurRunParams.TrialGpsPerEpoch {
			return true
		} else {
			return false
		}
	} else {
		if ev.AlphaCycle.Cur >= ev.CurRunParams.TrialsPerEpoch {
			return true
		} else {
			return false
		}
	}
}

// brought over from cemer. This was named StepStopTest in cemer
func (ev *PVLVEnv) TrialNameStopTest(_ *Sim) bool {
	return false
	//if ss.StopStepTrialNameString == "" {
	//	return ev.IsEpochEnd()
	//}
	//return strings.Contains(ev.CurAlphaTrial.AlphTrialName, ss.StopStepTrialNameString)
}

// end SingleTrial and functions

// TrainEnd
func (ev *PVLVEnv) TrainEnd(ss *Sim) {
	if ev.CurRunParams.SaveFinalWts {
		ev.SaveWeights(ss)
	}
	if ev.CurRunParams.LoadExp {
		ev.SaveOutputData(ss)
	}
	ss.Stop()
}

// end TrainEnd
