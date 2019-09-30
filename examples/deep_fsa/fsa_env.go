// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/etable/etensor"
)

// FSAEnv generates states in a finite state automaton (FSA) which is a
// simple form of grammar for creating non-deterministic but still
// overall structured sequences.
type FSAEnv struct {
	Nm         string          `desc:"name of this environment"`
	Dsc        string          `desc:"description of this environment"`
	EpochNSeq  int             `desc:"number of sequences to iterate per epoch"`
	TMat       etensor.Float64 `view:"no-inline" desc:"transition matrix, which is a square NxN tensor with outer dim being current state and inner dim having probability of transitioning to that state"`
	Labels     etensor.String  `desc:"transition labels, one for each transition cell in TMat matrix"`
	CurState   int             `desc:"current state within FSA that we're in"`
	PrvState   int             `desc:"previous state within FSA"`
	NextStates []int           `desc:"next states that have non-zero probability, with randomly chosen one at start"`
	NextLabels []string        `desc:"transition labels for next states that have non-zero probability, with randomly chosen one at start"`
	Run        env.Ctr         `view:"inline" desc:"current run of model as provided during Init"`
	Epoch      env.Ctr         `view:"inline" desc:"number of times through EpochNSeq number of sequences"`
	Seq        env.Qtr         `view:"inline" desc:"sequence counter within epoch"`
	Trial      env.Ctr         `view:"inline" desc:"trial is the step counter within sequence - how many steps taken within current sequence -- it resets to 0 at start of each sequence"`
}

func (fe *FSAEnv) Name() string { return fe.Nm }
func (fe *FSAEnv) Desc() string { return fe.Dsc }

// TMatReber sets the transition matrix to the standard Reber grammar FSA
func (fe *FSAEnv) TMatReber() {
	fe.TMat.SetShape([]int{6, 6}, nil, []string{"cur", "next"})
	fe.TMat.SetZeros()
	fe.TMat.Set([]int{0, 1}, 0.5)
	fe.Init()
}

func (fe *FSAEnv) Validate() error {
	fe.Run.Scale = Run
	fe.Epoch.Scale = Epoch
	fe.Trial.Scale = Trial
	if fe.TMat.Len() == 0 {
		return fmt.Errorf("FSAEnv: %v has no transition matrix TMat set", fe.Nm)
	}
	return nil
}

func (fe *FSAEnv) Counters() []TimeScales {
	return []TimeScales{Run, Epoch, Sequence, Trial}
}

func (fe *FSAEnv) States() Elements {
	els := Elements{}
	els.FromSchema(fe.Table.Table.Schema())
	return els
}

func (fe *FSAEnv) Actions() Elements {
	return nil
}

func (fe *FSAEnv) Init(run int) {
	fe.Run.Init()
	fe.Epoch.Init()
	fe.Seq.Init()
	fe.Trial.Init()
	fe.Run.Cur = run
	fe.Trial.Cur = -1 // init state -- key so that first Step() = 0
	fe.CurState = 0
	fe.PrvState = -1
	nstates := fe.TMat.Dim(0)
	if cap(fe.NextStates) < nstates {
		fe.NextStates = make([]int, 0, nstates)
	}
}

// NextState sets NextStates including randomly chosen one at start
func (fe *FSAEnv) NextState() {
	nstates := fe.TMat.Dim(0)
	if fe.CurState < 0 || fe.CurState >= nstates {
		fe.CurState = 0
	}
	ri := fe.CurState * nstate
	ps := fe.TMat.Values[ri : ri+nstate]
	ls := fe.Labels.Values[ri : ri+nstate]
	ns := erand.PChoose64(ps) // next state chosen at random
	fe.NextStates[0] = ns
	fe.NextLabels[0] = ls[ns]
	idx := 1
	for i, p := range ps {
		if i != ns && p > 0 {
			fe.NextStates[idx] = i
			fe.NextLabels[idx] = ls[i]
			idx++
		}
	}
	fe.PrvState = fe.CurState
	fe.CurState = ns
}

func (fe *FSAEnv) Step() bool {
	fe.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	fe.NextState()
	if fe.Trial.Incr() { // if true, hit max, reset to 0
		erand.PermuteInts(fe.Order)
		fe.Epoch.Incr()
	}
	fe.PrvTrialName = fe.TrialName
	fe.SetTrialName()
	return true
}

func (fe *FSAEnv) State(element string) etensor.Tensor {
	et, err := fe.Table.Table.CellTensorTry(element, fe.Row())
	if err != nil {
		log.Println(err)
	}
	return et
}

func (fe *FSAEnv) Action(element string, input etensor.Tensor) {
	// nop
}

func (fe *FSAEnv) Counter(scale TimeScales) (cur, prv int, chg bool) {
	switch scale {
	case Run:
		return fe.Run.Query()
	case Epoch:
		return fe.Epoch.Query()
	case Trial:
		return fe.Trial.Query()
	}
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ Env = (*FSAEnv)(nil)
