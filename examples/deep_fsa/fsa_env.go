// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/etime"
)

// FSAEnv generates states in a finite state automaton (FSA) which is a
// simple form of grammar for creating non-deterministic but still
// overall structured sequences.
type FSAEnv struct {

	// name of this environment
	Name string

	// transition matrix, which is a square NxN tensor with outer dim being current state and inner dim having probability of transitioning to that state
	TMat tensor.Float64 `display:"no-inline"`

	// transition labels, one for each transition cell in TMat matrix
	Labels tensor.String

	// automaton state within FSA that we're in
	AState env.CurPrvInt

	// number of next states in current state output (scalar)
	NNext tensor.Int

	// next states that have non-zero probability, with actual randomly chosen next state at start
	NextStates tensor.Int

	// transition labels for next states that have non-zero probability, with actual randomly chosen one for next state at start
	NextLabels tensor.String

	// sequence counter within epoch
	Seq env.Counter `display:"inline"`

	// tick counter within sequence
	Tick env.Counter `display:"inline"`

	// trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence
	Trial env.Counter `display:"inline"`
}

func (ev *FSAEnv) Label() string { return ev.Name }

// InitTMat initializes matrix and labels to given size
func (ev *FSAEnv) InitTMat(nst int) {
	ev.TMat.SetShape([]int{nst, nst})
	ev.Labels.SetShape([]int{nst, nst})
	ev.TMat.SetZeros()
	ev.Labels.SetZeros()
	ev.NNext.SetShape([]int{1})
	ev.NextStates.SetShape([]int{nst})
	ev.NextLabels.SetShape([]int{nst})
}

// SetTMat sets given transition matrix probability and label
func (ev *FSAEnv) SetTMat(fm, to int, p float64, lbl string) {
	ev.TMat.Set([]int{fm, to}, p)
	ev.Labels.Set([]int{fm, to}, lbl)
}

// TMatReber sets the transition matrix to the standard Reber grammar FSA
func (ev *FSAEnv) TMatReber() {
	ev.InitTMat(8)
	ev.SetTMat(0, 1, 1, "B")   // 0 = start
	ev.SetTMat(1, 2, 0.5, "T") // 1 = state 0 in usu diagram (+1 for all states)
	ev.SetTMat(1, 3, 0.5, "P")
	ev.SetTMat(2, 2, 0.5, "S")
	ev.SetTMat(2, 4, 0.5, "X")
	ev.SetTMat(3, 3, 0.5, "T")
	ev.SetTMat(3, 5, 0.5, "V")
	ev.SetTMat(4, 6, 0.5, "S")
	ev.SetTMat(4, 3, 0.5, "X")
	ev.SetTMat(5, 6, 0.5, "V")
	ev.SetTMat(5, 4, 0.5, "P")
	ev.SetTMat(6, 7, 1, "E") // 7 = end
	ev.Init(0)
}

func (ev *FSAEnv) Validate() error {
	if ev.TMat.Len() == 0 {
		return fmt.Errorf("FSAEnv: %v has no transition matrix TMat set", ev.Name)
	}
	return nil
}

func (ev *FSAEnv) State(element string) tensor.Tensor {
	switch element {
	case "NNext":
		return &ev.NNext
	case "NextStates":
		return &ev.NextStates
	case "NextLabels":
		return &ev.NextLabels
	}
	return nil
}

// String returns the current state as a string
func (ev *FSAEnv) String() string {
	nn := ev.NNext.Values[0]
	lbls := ev.NextLabels.Values[0:nn]
	return fmt.Sprintf("S_%d_%v", ev.AState.Cur, lbls)
}

func (ev *FSAEnv) Init(run int) {
	ev.Tick.Scale = etime.Tick
	ev.Trial.Scale = etime.Trial
	ev.Seq.Init()
	ev.Tick.Init()
	ev.Trial.Init()
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
	ev.AState.Cur = 0
	ev.AState.Prv = -1
}

// NextState sets NextStates including randomly chosen one at start
func (ev *FSAEnv) NextState() {
	nst := ev.TMat.DimSize(0)
	if ev.AState.Cur < 0 || ev.AState.Cur >= nst-1 {
		ev.AState.Cur = 0
	}
	ri := ev.AState.Cur * nst
	ps := ev.TMat.Values[ri : ri+nst]
	ls := ev.Labels.Values[ri : ri+nst]
	nxt := randx.PChoose64(ps) // next state chosen at random
	ev.NextStates.Set1D(0, nxt)
	ev.NextLabels.Set1D(0, ls[nxt])
	idx := 1
	for i, p := range ps {
		if i != nxt && p > 0 {
			ev.NextStates.Set1D(idx, i)
			ev.NextLabels.Set1D(idx, ls[i])
			idx++
		}
	}
	ev.NNext.Set1D(0, idx)
	ev.AState.Set(nxt)
}

func (ev *FSAEnv) Step() bool {
	ev.NextState()
	ev.Trial.Incr()
	ev.Tick.Incr()
	if ev.AState.Prv == 0 {
		ev.Tick.Init()
		ev.Seq.Incr()
	}
	return true
}

func (ev *FSAEnv) Action(element string, input tensor.Tensor) {
	// nop
}

// Compile-time check that implements Env interface
var _ env.Env = (*FSAEnv)(nil)
