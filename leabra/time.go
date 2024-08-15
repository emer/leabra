// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import "github.com/emer/emergent/v2/etime"

// leabra.Time contains all the timing state and parameter information for running a model
type Time struct {

	// accumulated amount of time the network has been running,
	// in simulation-time (not real world time), in seconds.
	Time float32

	// cycle counter: number of iterations of activation updating
	// (settling) on the current alpha-cycle (100 msec / 10 Hz) trial.
	// This counts time sequentially through the entire trial,
	// typically from 0 to 99 cycles.
	Cycle int

	// total cycle count. this increments continuously from whenever
	// it was last reset, typically this is number of milliseconds
	// in simulation time.
	CycleTot int

	// current gamma-frequency (25 msec / 40 Hz) quarter of alpha-cycle
	// (100 msec / 10 Hz) trial being processed.
	// Due to 0-based indexing, the first quarter is 0, second is 1, etc.
	// The plus phase final quarter is 3.
	Quarter Quarters

	// true if this is the plus phase (final quarter = 3), else minus phase.
	PlusPhase bool

	// amount of time to increment per cycle.
	TimePerCyc float32 `def:"0.001"`

	// number of cycles per quarter to run: 25 = standard 100 msec alpha-cycle.
	CycPerQtr int `def:"25"`

	// current evaluation mode, e.g., Train, Test, etc
	Mode etime.Modes
}

// NewTime returns a new Time struct with default parameters
func NewTime() *Time {
	tm := &Time{}
	tm.Defaults()
	return tm
}

// Defaults sets default values
func (tm *Time) Defaults() {
	tm.TimePerCyc = 0.001
	tm.CycPerQtr = 25
}

// Reset resets the counters all back to zero
func (tm *Time) Reset() {
	tm.Time = 0
	tm.Cycle = 0
	tm.CycleTot = 0
	tm.Quarter = 0
	tm.PlusPhase = false
	if tm.CycPerQtr == 0 {
		tm.Defaults()
	}
}

// AlphaCycStart starts a new alpha-cycle (set of 4 quarters)
func (tm *Time) AlphaCycStart() {
	tm.Cycle = 0
	tm.Quarter = 0
	tm.PlusPhase = false
}

// CycleInc increments at the cycle level
func (tm *Time) CycleInc() {
	tm.Cycle++
	tm.CycleTot++
	tm.Time += tm.TimePerCyc
}

// QuarterInc increments at the quarter level, updating Quarter and PlusPhase
func (tm *Time) QuarterInc() {
	tm.Quarter++
}

// QuarterCycle returns the number of cycles into current quarter
func (tm *Time) QuarterCycle() int {
	qmin := int(tm.Quarter) * tm.CycPerQtr
	return tm.Cycle - qmin
}

//////////////////////////////////////////////////////////////////////////////////////
//  Quarters

// Quarters are the different alpha trial quarters, as a bitflag,
// for use in relevant timing parameters where quarters need to be specified.
// The Q1..4 defined values are integer *bit positions* -- use Set, Has etc methods
// to set bits from these bit positions.
type Quarters int64 //enums:bitflag

// The quarters
const (
	// Q1 is the first quarter, which, due to 0-based indexing, shows up as Quarter = 0 in timer
	Q1 Quarters = iota
	Q2
	Q3
	Q4
)

// HasNext returns true if the quarter after given quarter is set.
// This wraps around from Q4 to Q1.  (qtr = 0..3 = same as Quarters)
func (qt Quarters) HasNext(qtr Quarters) bool {
	nqt := (qtr + 1) % 4
	return qt.HasFlag(nqt)
}

// HasPrev returns true if the quarter before given quarter is set.
// This wraps around from Q1 to Q4.  (qtr = 0..3 = same as Quarters)
func (qt Quarters) HasPrev(qtr Quarters) bool {
	pqt := (qtr - 1)
	if pqt < 0 {
		pqt += 4
	}
	return qt.HasFlag(pqt)
}
