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
func LooperStdPhases(ls *looper.Stacks, ctx *Context, net *Network, plusStart, plusEnd int, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	ls.AddEventAllModes(etime.Cycle, "MinusPhase:Start", 0, func() {
		ctx.PlusPhase = false
	})
	ls.AddEventAllModes(etime.Cycle, "Quarter1", 25, func() {
		net.QuarterFinal(ctx)
		ctx.QuarterInc()
	})
	ls.AddEventAllModes(etime.Cycle, "Quarter2", 50, func() {
		net.QuarterFinal(ctx)
		ctx.QuarterInc()
	})
	ls.AddEventAllModes(etime.Cycle, "MinusPhase:End", plusStart, func() {
		net.QuarterFinal(ctx)
		ctx.QuarterInc()
	})
	ls.AddEventAllModes(etime.Cycle, "PlusPhase:Start", plusStart, func() {
		ctx.PlusPhase = true
	})

	for m, stack := range ls.Stacks {
		stack.Loops[trl].OnStart.Add("AlphaCycInit", func() {
			net.AlphaCycInit(m == etime.Train)
			ctx.AlphaCycStart()
		})
		stack.Loops[trl].OnEnd.Add("PlusPhase:End", func() {
			net.QuarterFinal(ctx)
		})
	}
}

// LooperSimCycleAndLearn adds Cycle and DWt, WtFromDWt functions to looper
// for given network, ctx, and netview update manager
// Can pass a trial-level time scale to use instead of the default etime.Trial
func LooperSimCycleAndLearn(ls *looper.Stacks, net *Network, ctx *Context, viewupdt *netview.ViewUpdate, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	for m := range ls.Stacks {
		ls.Stacks[m].Loops[etime.Cycle].OnStart.Add("Cycle", func() {
			net.Cycle(ctx)
			ctx.CycleInc()
		})
	}
	ttrl := ls.Loop(etime.Train, trl)
	if ttrl != nil {
		ttrl.OnEnd.Add("UpdateWeights", func() {
			net.DWt()
			if viewupdt.IsViewingSynapse() {
				viewupdt.RecordSyns() // note: critical to update weights here so DWt is visible
			}
			net.WtFromDWt()
		})
	}

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range ls.Stacks {
		for _, loop := range loops.Loops {
			loop.OnStart.Add("SetCtxMode", func() {
				ctx.Mode = m.(etime.Modes)
			})
		}
	}
}

// LooperResetLogBelow adds a function in OnStart to all stacks and loops
// to reset the log at the level below each loop -- this is good default behavior.
// Exceptions can be passed to exclude specific levels -- e.g., if except is Epoch
// then Epoch does not reset the log below it
func LooperResetLogBelow(ls *looper.Stacks, logs *elog.Logs, except ...etime.Times) {
	for m, stack := range ls.Stacks {
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
					logs.ResetLog(m.(etime.Modes), below.(etime.Times))
				})
			}
		}
	}
}

// LooperUpdateNetView adds netview update calls at each time level
func LooperUpdateNetView(ls *looper.Stacks, viewupdt *netview.ViewUpdate, net *Network, ctrUpdateFunc func(tm etime.Times)) {
	for m, stack := range ls.Stacks {
		for t, loop := range stack.Loops {
			curTime := t.(etime.Times)
			if curTime != etime.Cycle {
				loop.OnEnd.Add("GUI:UpdateNetView", func() {
					ctrUpdateFunc(curTime)
					viewupdt.Testing = m == etime.Test
					viewupdt.UpdateTime(curTime)
				})
			}
		}
		cycLoop := ls.Loop(m, etime.Cycle)
		cycLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			cyc := cycLoop.Counter.Cur
			ctrUpdateFunc(etime.Cycle)
			viewupdt.Testing = m == etime.Test
			viewupdt.UpdateCycle(cyc)
		})
	}
}

// LooperUpdatePlots adds plot update calls at each time level
func LooperUpdatePlots(ls *looper.Stacks, gui *egui.GUI) {
	for m, stack := range ls.Stacks {
		for t, loop := range stack.Loops {
			curTime := t.(etime.Times)
			curLoop := loop
			if curTime == etime.Cycle {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					cyc := curLoop.Counter.Cur
					gui.GoUpdateCyclePlot(m.(etime.Modes), cyc)
				})
			} else {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					gui.GoUpdatePlot(m.(etime.Modes), curTime)
				})
			}
		}
	}
}
