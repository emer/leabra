// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
)

// Actions are SIR actions
type Actions int32 //enums:enum

const (
	Store1 Actions = iota
	Store2
	Ignore
	Recall1
	Recall2
)

// SIREnv implements the store-ignore-recall task
type SIREnv struct {
	// name of this environment
	Name string

	// number of different stimuli that can be maintained
	NStim int

	// value for reward, based on whether model output = target
	RewVal float32

	// value for non-reward
	NoRewVal float32

	// current action
	Act Actions

	// current stimulus
	Stim int

	// current stimulus being maintained
	Maint1 int

	// current stimulus being maintained
	Maint2 int

	// stimulus input pattern
	Input tensor.Float64

	// input pattern with action
	CtrlInput tensor.Float64

	// output pattern of what to respond
	Output tensor.Float64

	// reward value
	Reward tensor.Float64

	// trial is the step counter within epoch
	Trial env.Counter `display:"inline"`
}

func (ev *SIREnv) Label() string { return ev.Name }

// SetNStim initializes env for given number of stimuli, init states
func (ev *SIREnv) SetNStim(n int) {
	ev.NStim = n
	ev.Input.SetShape([]int{n})
	ev.CtrlInput.SetShape([]int{int(ActionsN)})
	ev.Output.SetShape([]int{n})
	ev.Reward.SetShape([]int{1})
	if ev.RewVal == 0 {
		ev.RewVal = 1
	}
}

func (ev *SIREnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "CtrlInput":
		return &ev.CtrlInput
	case "Output":
		return &ev.Output
	case "Rew":
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
	ev.Trial.Scale = etime.Trial
	ev.Trial.Init()
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
	ev.StepSIR()
	ev.Trial.Incr()
	return true
}

func (ev *SIREnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*SIREnv)(nil)
