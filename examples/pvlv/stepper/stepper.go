package stepper

import (
	"github.com/goki/ki/kit"
	"sync"
)

// Adds the ability to suspend execution at any point without losing any context, and without having to save any
// state.
// To use:
// - Define a StepGrain enumerated type, with whatever values you'd like--values are just ints within the stepper.
// - Add calls to stepper.StepPoint in your code. StepPoint takes an int, which should be one of the StepGrain
//   values defined above. StepPoint return true if the simulation should bail out of the current run context.
//   Otherwise, simply continue.
// - Optionally, you can define a StopConditionChecker function, and call RegisterStopChecker in order to check
//   whatever specialized state you need to in a callback to your StopConditionChecker function. Your function will
//   be called with the StepGrain used in the StepPoint call, and an arbitrary value (usually a pointer to the top-level
//   Sim struct) of your choosing, which it can use in any way to determine if execution should pause.
// - At any point, the main UI process, which runs in a different GoRoutine from the sim execution process, can call
//   stepper.Pause or stepper.Stop. A call to Pause will cause the sim to enter the Paused state at the next StepPoint
//   call, while a call to Stop will cause the sim to exit (i.e., return) at the next StepPoint call.

type RunState int

const (
	Created RunState = iota
	Stopped
	Paused
	Stepping
	Running
	RunStateN
)

var KiT_RunState = kit.Enums.AddEnum(RunStateN, kit.NotBitFlag, nil)

//go:generate stringer -type=RunState

//type IRunner interface {
//	Init()
//	Run()
//	Step(n int)
//	CheckStates() (cur, req RunState)
//	SetState(state RunState)
//}

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

func New() *Stepper { return new(Stepper).Init() }

var oneTimeInit sync.Once

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

func (st *Stepper) SetStepGrain(grain int) {
	st.StepGrain = grain
}

func (st *Stepper) Grain() int {
	return st.StepGrain
}

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

func (st *Stepper) Enter(state RunState) {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	st.CurState = state
	st.RequestedState = state
	st.StateChange.Broadcast()
}

func (st *Stepper) RegisterStopChecker(checker StopConditionChecker, cs interface{}) {
	st.CondChecker = checker
	st.CheckerState = cs
}

func (st *Stepper) RegisterPauseNotifier(notifier PauseNotifier, pnState interface{}) {
	st.PauseNotifier = notifier
	st.PNState = pnState
}

func (st *Stepper) Stop() {
	st.PleaseEnter(Stopped)
}

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

func (st *Stepper) StopRequested() bool {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	return st.RequestedState == Stopped
}

// Pause and wait for stop or go signal
func (st *Stepper) Pause() {
	st.PleaseEnter(Paused)
}

func (st *Stepper) Active() bool {
	st.StateMut.Lock()
	defer st.StateMut.Unlock()
	return st.CurState == Running || st.CurState == Stepping
}

func (st *Stepper) StartStepping(grain int, nSteps int) {
	if nSteps > 0 {
		st.StepsRemaining = nSteps
	}
	st.Stepping = true
	st.StepGrain = grain
	st.Enter(Stepping)
}

func (st *Stepper) SetNSteps(toTake int) {
	st.StepsPerClick = toTake
	st.StepsRemaining = toTake
}

// Wait for RunState to become the passed-in state
// May return Stopped, in which case the caller
// should clean up and return to its caller
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
func (st *Stepper) StepPoint(grain int) (stop bool) {
	state, reqState := st.CheckStates()
	if reqState == Stopped && state != Stopped {
		st.Enter(Stopped)
	}
	if state == Stopped || reqState == Stopped {
		return true
	}
	if !st.Stepping {
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
