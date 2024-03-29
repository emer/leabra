// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/kit"
)

// Actions are SIR actions
type Actions int

//go:generate stringer -type=Actions

var KiT_Actions = kit.Enums.AddEnum(ActionsN, kit.NotBitFlag, nil)

func (ev Actions) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Actions) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

const (
	Store1 Actions = iota
	Store2
	Ignore
	Recall1
	Recall2
	ActionsN
)

// SIREnv implements the store-ignore-recall task
type SIREnv struct {

	// name of this environment
	Nm string `desc:"name of this environment"`

	// description of this environment
	Dsc string `desc:"description of this environment"`

	// number of different stimuli that can be maintained
	NStim int `desc:"number of different stimuli that can be maintained"`

	// value for reward, based on whether model output = target
	RewVal float32 `desc:"value for reward, based on whether model output = target"`

	// value for non-reward
	NoRewVal float32 `desc:"value for non-reward"`

	// current action
	Act Actions `desc:"current action"`

	// current stimulus
	Stim int `desc:"current stimulus"`

	// current stimulus being maintained
	Maint1 int `desc:"current stimulus being maintained"`

	// current stimulus being maintained
	Maint2 int `desc:"current stimulus being maintained"`

	// stimulus input pattern
	Input etensor.Float64 `desc:"stimulus input pattern"`

	// input pattern with action
	CtrlInput etensor.Float64 `desc:"input pattern with action"`

	// output pattern of what to respond
	Output etensor.Float64 `desc:"output pattern of what to respond"`

	// reward value
	Reward etensor.Float64 `desc:"reward value"`

	// [view: inline] current run of model as provided during Init
	Run env.Ctr `view:"inline" desc:"current run of model as provided during Init"`

	// [view: inline] number of times through Seq.Max number of sequences
	Epoch env.Ctr `view:"inline" desc:"number of times through Seq.Max number of sequences"`

	// [view: inline] trial is the step counter within epoch
	Trial env.Ctr `view:"inline" desc:"trial is the step counter within epoch"`
}

func (ev *SIREnv) Name() string { return ev.Nm }
func (ev *SIREnv) Desc() string { return ev.Dsc }

// SetNStim initializes env for given number of stimuli, init states
func (ev *SIREnv) SetNStim(n int) {
	ev.NStim = n
	ev.Input.SetShape([]int{n}, nil, []string{"N"})
	ev.CtrlInput.SetShape([]int{int(ActionsN)}, nil, []string{"N"})
	ev.Output.SetShape([]int{n}, nil, []string{"N"})
	ev.Reward.SetShape([]int{1}, nil, []string{"1"})
	if ev.RewVal == 0 {
		ev.RewVal = 1
	}
}

func (ev *SIREnv) Validate() error {
	if ev.NStim <= 0 {
		return fmt.Errorf("SIREnv: %v NStim == 0 -- must set with SetNStim call", ev.Nm)
	}
	return nil
}

func (ev *SIREnv) State(element string) etensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "CtrlInput":
		return &ev.CtrlInput
	case "Output":
		return &ev.Output
	case "Reward":
		return &ev.Reward
	}
	return nil
}

func (ev *SIREnv) Actions() env.Elements {
	return nil
}

// StimStr returns a letter string rep of stim (A, B...)
func (ev *SIREnv) StimStr(stim int) string {
	return string([]byte{byte('A' + stim)})
}

// String returns the current state as a string
func (ev *SIREnv) String() string {
	return fmt.Sprintf("%s_%s_mnt1_%s_mnt2_%s_rew_%g", ev.Act, ev.StimStr(ev.Stim), ev.StimStr(ev.Maint1), ev.StimStr(ev.Maint2), ev.Reward.Values[0])
}

func (ev *SIREnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.Maint1 = -1
	ev.Maint2 = -1
}

// SetState sets the input, output states
func (ev *SIREnv) SetState() {
	ev.CtrlInput.SetZeros()
	ev.CtrlInput.Values[ev.Act] = 1
	ev.Input.SetZeros()
	if ev.Act != Recall1 && ev.Act != Recall2 {
		ev.Input.Values[ev.Stim] = 1
	}
	ev.Output.SetZeros()
	ev.Output.Values[ev.Stim] = 1
}

// SetReward sets reward based on network's output
func (ev *SIREnv) SetReward(netout int) bool {
	cor := ev.Stim // already correct
	rw := netout == cor
	if rw {
		ev.Reward.Values[0] = float64(ev.RewVal)
	} else {
		ev.Reward.Values[0] = float64(ev.NoRewVal)
	}
	return rw
}

// Step the SIR task
func (ev *SIREnv) StepSIR() {
	for {
		ev.Act = Actions(rand.Intn(int(ActionsN)))
		if ev.Act == Store1 && ev.Maint1 >= 0 { // already full
			continue
		}
		if ev.Act == Recall1 && ev.Maint1 < 0 { // nothing
			continue
		}
		if ev.Act == Store2 && ev.Maint2 >= 0 { // already full
			continue
		}
		if ev.Act == Recall2 && ev.Maint2 < 0 { // nothing
			continue
		}
		break
	}
	ev.Stim = rand.Intn(ev.NStim)
	switch ev.Act {
	case Store1:
		ev.Maint1 = ev.Stim
	case Store2:
		ev.Maint2 = ev.Stim
	case Ignore:
	case Recall1:
		ev.Stim = ev.Maint1
		ev.Maint1 = -1
	case Recall2:
		ev.Stim = ev.Maint2
		ev.Maint2 = -1
	}
	ev.SetState()
}

func (ev *SIREnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start

	ev.StepSIR()

	if ev.Trial.Incr() {
		ev.Epoch.Incr()
	}
	return true
}

func (ev *SIREnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (ev *SIREnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case env.Run:
		return ev.Run.Query()
	case env.Epoch:
		return ev.Epoch.Query()
	case env.Trial:
		return ev.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*SIREnv)(nil)
