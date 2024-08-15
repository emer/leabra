// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
)

// LooperStdPhases adds the minus and plus phases of the alpha cycle,
// along with embedded beta phases which just record St1 and St2 activity in this case.
// plusStart is start of plus phase, typically 75,
// and plusEnd is end of plus phase, typically 99
// resets the state at start of trial.
// Can pass a trial-level time scale to use instead of the default etime.Trial
func LooperStdPhases(man *looper.Manager, ltime *Time, net *Network, plusStart, plusEnd int, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	minusPhase := &looper.Event{Name: "MinusPhase", AtCounter: 0}
	minusPhase.OnEvent.Add("MinusPhase:Start", func() {
		ltime.PlusPhase = false
	})
	quarter1 := looper.NewEvent("Quarter1", 25, func() {
		net.QuarterFinal(ltime)
		ltime.QuarterInc()
	})
	quarter2 := looper.NewEvent("Quarter2", 50, func() {
		net.QuarterFinal(ltime)
		ltime.QuarterInc()
	})
	plusPhase := &looper.Event{Name: "PlusPhase", AtCounter: plusStart}
	plusPhase.OnEvent.Add("MinusPhase:End", func() {
		net.MinusPhase(ltime)
		ltime.QuarterInc()
	})
	plusPhase.OnEvent.Add("PlusPhase:Start", func() {
		ltime.PlusPhase = true
	})

	man.AddEventAllModes(etime.Cycle, minusPhase, quarter1, quarter2, plusPhase)

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[trl].OnStart.Add("AlphaCycInit", func() {
			net.AlphaCycInit(mode == etime.Train)
			ltime.AlphaCycStart()
		})
		stack.Loops[trl].OnEnd.Add("PlusPhase:End", func() {
			net.PlusPhase(ltime)
		})
	}
}

// LooperSimCycleAndLearn adds Cycle and DWt, WtFromDWt functions to looper
// for given network, ltime, and netview update manager
// Can pass a trial-level time scale to use instead of the default etime.Trial
func LooperSimCycleAndLearn(man *looper.Manager, net *Network, ltime *Time, viewupdt *netview.ViewUpdate, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	for m, _ := range man.Stacks {
		cycLoop := man.Stacks[m].Loops[etime.Cycle]
		cycLoop.Main.Add("Cycle", func() {
			net.Cycle(ltime)
			ltime.CycleInc()
		})
	}
	man.GetLoop(etime.Train, trl).OnEnd.Add("UpdateWeights", func() {
		net.DWt()
		if viewupdt.IsViewingSynapse() {
			viewupdt.RecordSyns() // note: critical to update weights here so DWt is visible
		}
		net.WtFromDWt()
	})

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range man.Stacks {
		curMode := m // For closures.
		for _, loop := range loops.Loops {
			loop.OnStart.Add("SetTimeMode", func() {
				ltime.Mode = curMode
			})
		}
	}
}

// LooperResetLogBelow adds a function in OnStart to all stacks and loops
// to reset the log at the level below each loop -- this is good default behavior.
// Exceptions can be passed to exclude specific levels -- e.g., if except is Epoch
// then Epoch does not reset the log below it
func LooperResetLogBelow(man *looper.Manager, logs *elog.Logs, except ...etime.Times) {
	for m, stack := range man.Stacks {
		curMode := m // For closures.
		for t, loop := range stack.Loops {
			curTime := t
			isExcept := false
			for _, ex := range except {
				if curTime == ex {
					isExcept = true
					break
				}
			}
			if below := stack.TimeBelow(curTime); !isExcept && below != etime.NoTime {
				loop.OnStart.Add("ResetLog"+below.String(), func() {
					logs.ResetLog(curMode, below)
				})
			}
		}
	}
}

// LooperUpdateNetView adds netview update calls at each time level
func LooperUpdateNetView(man *looper.Manager, viewupdt *netview.ViewUpdate, net *Network, ctrUpdateFunc func(tm etime.Times)) {
	for m, stack := range man.Stacks {
		curMode := m // For closures.
		for t, loop := range stack.Loops {
			curTime := t
			if curTime != etime.Cycle {
				loop.OnEnd.Add("GUI:UpdateNetView", func() {
					ctrUpdateFunc(curTime)
					viewupdt.UpdateTime(curTime)
				})
			}
		}
		cycLoop := man.GetLoop(curMode, etime.Cycle)
		cycLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			cyc := cycLoop.Counter.Cur
			ctrUpdateFunc(etime.Cycle)
			viewupdt.UpdateCycle(cyc)
		})
	}
}

// LooperUpdatePlots adds plot update calls at each time level
func LooperUpdatePlots(man *looper.Manager, gui *egui.GUI) {
	for m, stack := range man.Stacks {
		curMode := m // For closures.
		for t, loop := range stack.Loops {
			curTime := t
			curLoop := loop
			if curTime == etime.Cycle {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					cyc := curLoop.Counter.Cur
					gui.UpdateCyclePlot(curMode, cyc)
				})
			} else {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					gui.UpdatePlot(curMode, curTime)
				})
			}
		}
	}
}
