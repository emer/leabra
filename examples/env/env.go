// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image"
	"math/rand"

	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/env"
)

// ExEnv is an example environment, that sets a single input point in a 2D
// input state and two output states as the X and Y coordinates of point.
// It can be used as a starting point for writing your own Env, without
// having much existing code to rewrite.
type ExEnv struct {

	// name of this environment
	Name string

	// size of each dimension in 2D input
	Size int

	// X,Y coordinates of point
	Point image.Point

	// input state, 2D Size x Size
	Input tensor.Float32

	// X as a one-hot state 1D Size
	X tensor.Float32

	// Y  as a one-hot state 1D Size
	Y tensor.Float32

	// current run of model as provided during Init
	Run env.Counter `display:"inline"`

	// number of times through Seq.Max number of sequences
	Epoch env.Counter `display:"inline"`

	// trial increments over input states -- could add Event as a lower level
	Trial env.Counter `display:"inline"`
}

func (ev *ExEnv) Label() string { return ev.Name }

// Config sets the size, number of trials to run per epoch, and configures the states
func (ev *ExEnv) Config(sz int, ntrls int) {
	ev.Size = sz
	ev.Trial.Max = ntrls
	ev.Input.SetShape([]int{sz, sz}, nil, []string{"Y", "X"})
	ev.X.SetShape([]int{sz}, nil, []string{"X"})
	ev.Y.SetShape([]int{sz}, nil, []string{"Y"})
}

func (ev *ExEnv) State(element string) tensor.Tensor {
	switch element {
	case "Input":
		return &ev.Input
	case "X":
		return &ev.X
	case "Y":
		return &ev.Y
	}
	return nil
}

// String returns the current state as a string
func (ev *ExEnv) String() string {
	return fmt.Sprintf("Pt_%d_%d", ev.Point.X, ev.Point.Y)
}

// Init is called to restart environment
func (ev *ExEnv) Init(run int) {
	ev.Run.Scale = env.Run
	ev.Epoch.Scale = env.Epoch
	ev.Trial.Scale = env.Trial
	ev.Run.Init()
	ev.Epoch.Init()
	ev.Trial.Init()
	ev.Run.Cur = run
	ev.Trial.Cur = -1 // init state -- key so that first Step() = 0
}

// NewPoint generates a new point and sets state accordingly
func (ev *ExEnv) NewPoint() {
	ev.Point.X = rand.Intn(ev.Size)
	ev.Point.Y = rand.Intn(ev.Size)
	ev.Input.SetZeros()
	ev.Input.SetFloat([]int{ev.Point.Y, ev.Point.X}, 1)
	ev.X.SetZeros()
	ev.X.SetFloat([]int{ev.Point.X}, 1)
	ev.Y.SetZeros()
	ev.Y.SetFloat([]int{ev.Point.Y}, 1)
}

// Step is called to advance the environment state
func (ev *ExEnv) Step() bool {
	ev.Epoch.Same() // good idea to just reset all non-inner-most counters at start
	ev.NewPoint()
	if ev.Trial.Incr() { // true if wraps around Max back to 0
		ev.Epoch.Incr()
	}
	return true
}

func (ev *ExEnv) Action(element string, input tensor.Tensor) {
	// nop
}

func (ev *ExEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
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
var _ env.Env = (*ExEnv)(nil)
