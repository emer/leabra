// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stepper allows you to set StepPoints in simulation code that will pause if some condition is satisfied.
// While paused, the simulation waits for the top-level process (the user interface) to tell it to continue.
// Once a continue notification is received, the simulation continues on its way, with all internal state
// exactly as it was when the StopPoint was hit, without having to explicitly save anything.

// There are two "running" states, Stepping and Running. The difference is that in the Running state, unless
// there is a Stop request, the application will forego the possibly-complex checking for a pause (see StepPoint,
// at the bottom of this file). StepPoint is written to make checking as quick as possible, so in general, Stepping
// is preferred, since it allows you to continue without having to rerun any initialization code.

package stepper

import (
	"github.com/goki/ki/kit"
	"sync"
)

type RunState int

const (
	Created  RunState = iota // this is the initial state, when a stepper is first created
	Stopped                  // execution is stopped. The sim is NOT waiting, so running again is basically a restart
	Paused                   // execution is paused. The sim is waiting for further instructions, and can continue, or stop
	Stepping                 // the application is running, but will pause if it hits a StepPoint that matches the current StepGrain
	Running                  // the application is running, and will NOT pause at StepPoints. It will stop if a stop has been requested.
	RunStateN
)

var KiT_RunState = kit.Enums.AddEnum(RunStateN, kit.NotBitFlag, nil)

//go:generate stringer -type=RunState

type StopConditionChecker func(sv interface{}, grain int) (matched bool)

type PauseNotifier func(sv interface{})

type Stepper struct {
	StateMut       sync.Mutex           `view:"-" desc:"mutex for RunState"`
	StateChange    *sync.Cond           `view:"-" desc:"state change condition variable"`
	CurState       RunState             `desc:"current run state"`
	RequestedState RunState             `desc:"requested run state"`
	StepGrain      int                  `desc:"granularity of one step. No enum type here so clients can define their own"`
	CondChecker    StopConditionChecker `view:"-" desc:"function to test for special stopping conditions"`
	CheckerState   interface{}          `view:"-" desc:"arbitrary state information for condition checker"`
	PauseNotifier  PauseNotifier        `view:"-" desc:"function to deal with any changes on client side when paused after stepping"`
	PNState        interface{}          `view:"-" desc:"arbitrary state information for pause notifier"`
	Stepping       bool                 `desc:"flag for stepping (as opposed to running"`
	StepsPerClick  int                  `desc:"number of steps to execute before returning"`
	StepsRemaining int                  `desc:"number of steps yet to execute before returning"`
}

// Make a new Stepper. Always call this to create a Stepper, so that initialization will be run correctly.
func New() *Stepper { return new(Stepper).Init() }

var oneTimeInit sync.Once // this ensures that global initialization only happens once

// Put everything into a good state before starting a run
// Called automatically by New, and should be called before running again after calling Stop (not Pause)
func (st *Stepper) Init() *Stepper {
	oneTimeInit.Do(func() {
		st.StateMut = sync.Mutex{}
		st.StateChange = sync.NewCond(&st.StateMut)
		st.CurState = Created
		st.RequestedState = Created
		st.StepGrain = 0
		st.StepsRemaining = 0
		st.StepsPerClick = 1
		st.Stepping = false
	})
	st.Enter(Stopped)
	return st
}

// Set the internal value of StepGrain to an uninterpreted int. Semantics are up to the client
func (st *Stepper) SetStepGrain(grain int) {
	st.StepGrain = grain
}

// Get the current StepGrain
func (st *Stepper) Grain() int {
	return st.StepGrain
}

// Request that the running application enter the requested state
// After calling this, the caller should check to see if the application has changed CurState
func (st *Stepper) PleaseEnter(state RunState) {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	st.RequestedState = state
	if state == Stepping {
		st.Stepping = true
	} else if state == Running {
		st.Stepping = false
	}
	st.StateChange.Broadcast()
}

// Unconditionally enter the specified RunState, without checking or waiting
func (st *Stepper) Enter(state RunState) {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	st.CurState = state
	st.RequestedState = state
	st.StateChange.Broadcast()
}

// Register a StopConditionChecked callback. This is completely optional.
func (st *Stepper) RegisterStopChecker(checker StopConditionChecker, cs interface{}) {
	st.CondChecker = checker
	st.CheckerState = cs
}

// Register a PauseNotifier callback
func (st *Stepper) RegisterPauseNotifier(notifier PauseNotifier, pnState interface{}) {
	st.PauseNotifier = notifier
	st.PNState = pnState
}

// Request to enter the Stopped state. Doesn't actually change state, and does not wait.
func (st *Stepper) Stop() {
	st.PleaseEnter(Stopped)
}

// Request to enter the Stopped state, and wait for it to happen.
// This will wait forever if nothing responds to the request.
func (st *Stepper) RequestStop(wait bool) {
	st.PleaseEnter(Stopped)
	if wait {
		st.StateMut.Lock()
		defer st.StateMut.Unlock()
		for {
			state := st.CurState
			switch state {
			case Stopped:
				return
			default:
				st.StateChange.Wait()
			}
		}
	}
}

// Check for a request to enter the Stopped state
func (st *Stepper) StopRequested() bool {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	return st.RequestedState == Stopped
}

// Pause and wait for stop or go signal
func (st *Stepper) Pause() {
	st.PleaseEnter(Paused)
}

// Check for the application either Running or Stepping (neither Paused nor Stopped)
func (st *Stepper) Active() bool {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	return st.CurState == Running || st.CurState == Stepping
}

// Enter the Stepping run state
func (st *Stepper) StartStepping(grain int, nSteps int) {
	if nSteps > 0 {
		st.StepsRemaining = nSteps
	}
	st.Stepping = true
	st.StepGrain = grain
	st.Enter(Stepping)
}

// Set the number of times to go through a StepPoint of the current granularity before actually pausing
func (st *Stepper) SetNSteps(toTake int) {
	st.StepsPerClick = toTake
	st.StepsRemaining = toTake
}

// Wait for RunState to become the passed-in state, or Stopped
// May return Stopped, in which case the caller should clean up and return to its caller.
func (st *Stepper) WaitUntil(state RunState) RunState {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	for {
		if st.CurState == state {
			return state
		} else if st.CurState == Stopped {
			return Stopped
		} else {
			st.StateChange.Wait()
		}
	}
}

// Wait for the application to enter Running, Stepping, or Stopped.
// Return the entered state
func (st *Stepper) WaitToGo() RunState {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	for {
		state := st.CurState
		if state == Running || state == Stepping {
			return state
		} else if st.CurState == Stopped {
			return Stopped
		} else {
			st.StateChange.Wait()
		}
	}
}

// Get current RunState
func (st *Stepper) CheckStates() (cur, req RunState) {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	return st.CurState, st.RequestedState
}

// Check for possible pause or stop
// If the application is:
// Running: keep going with no further examination of state
// Stopped: return true, and the application should return (i.e., stop completely)
// Stepping: check to see if we should pause (if StepGrain matches, decrement StepsRemaining, stop if <= 0)
// Paused: wait for state change
func (st *Stepper) StepPoint(grain int) (stop bool) {
	state, reqState := st.CheckStates()
	if reqState == Stopped && state != Stopped {
		st.Enter(Stopped)
	}
	if state == Stopped || reqState == Stopped {
		return true
	}
	if !st.Stepping { // quick check. If state is Running, just keep going. If Paused, Stepping should be true.
		return false
	}
	if grain == st.StepGrain { // exact equality is the only test that really works well
		st.StepsRemaining--
		if st.StepsRemaining <= 0 {
			st.Enter(Paused)
			st.PauseNotifier(st.PNState)
			st.StepsRemaining = st.StepsPerClick
		}
	}
	stopMatched := st.CondChecker(st.CheckerState, grain)
	if stopMatched {
		st.Enter(Stopped)
		return true
	}
	for {
		_, reqState := st.CheckStates()
		st.StateMut.Lock()
		switch reqState {
		case Stopped:
			st.CurState = Stopped
			st.StateMut.Unlock()
			return true
		case Running:
			st.Stepping = false
			st.CurState = Running
			st.StateMut.Unlock()
			return false
		case Stepping:
			st.Stepping = true
			st.CurState = Stepping
			st.StateMut.Unlock()
			return false
		case Paused:
			st.StateChange.Wait()
			st.StateMut.Unlock()
		}
	}
}
