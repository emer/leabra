// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import "github.com/goki/ki/kit"

// leabra.Time contains all the timing state and parameter information for running a model
type Time struct {
	Time      float32 `desc:"accumulated amount of time the network has been running, in simulation-time (not real world time), in seconds"`
	Cycle     int     `desc:"cycle counter: number of iterations of activation updating (settling) on the current alpha-cycle (100 msec / 10 Hz) trial -- this counts time sequentially through the entire trial, typically from 0 to 99 cycles"`
	CycleTot  int     `desc:"total cycle count -- this increments continuously from whenever it was last reset -- typically this is number of milliseconds in simulation time"`
	Quarter   int     `desc:"[0-3] current gamma-frequency (25 msec / 40 Hz) quarter of alpha-cycle (100 msec / 10 Hz) trial being processed.  Due to 0-based indexing, the first quarter is 0, second is 1, etc -- the plus phase final quarter is 3."`
	PlusPhase bool    `desc:"true if this is the plus phase (final quarter = 3) -- else minus phase"`

	TimePerCyc float32 `def:"0.001" desc:"amount of time to increment per cycle"`
	CycPerQtr  int     `def:"25" desc:"number of cycles per quarter to run -- 25 = standard 100 msec alpha-cycle"`
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
	if tm.Quarter == 3 {
		tm.PlusPhase = true
	} else {
		tm.PlusPhase = false
	}
}

// QuarterCycle returns the number of cycles into current quarter
func (tm *Time) QuarterCycle() int {
	qmin := tm.Quarter * tm.CycPerQtr
	return tm.Cycle - qmin
}

//////////////////////////////////////////////////////////////////////////////////////
//  Quarters

// Quarters are the different alpha trial quarters, as a bitflag, for use in relevant timing
// parameters where quarters need to be specified
type Quarters int32

//go:generate stringer -type=Quarters

var KiT_Quarters = kit.Enums.AddEnum(QuartersN, kit.BitFlag, nil)

func (ev Quarters) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Quarters) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The quarters
const (
	// Q1 is the first quarter, which, due to 0-based indexing, shows up as Quarter = 0 in timer
	Q1 Quarters = iota
	Q2
	Q3
	Q4
	QuartersN
)

//////////////////////////////////////////////////////////////////////////////////////
//  TimeScales

// TimeScales are the different time scales associated with overall simulation running, and
// can be used to parameterize the updating and control flow of simulations at different scales.
// The definitions become increasingly subjective imprecise as the time scales increase.
// This is not used directly in the algorithm code -- all control is responsibility of the
// end simulation.  This list is designed to standardize terminology across simulations and
// establish a common conceptual framework for time -- it can easily be extended in specific
// simulations to add needed additional levels, although using one of the existing standard
// values is recommended wherever possible.
type TimeScales int32

//go:generate stringer -type=TimeScales

var KiT_TimeScales = kit.Enums.AddEnum(TimeScalesN, kit.NotBitFlag, nil)

func (ev TimeScales) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *TimeScales) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The time scales
const (
	// Cycle is the finest time scale -- typically 1 msec -- a single activation update.
	Cycle TimeScales = iota

	// FastSpike is typically 10 cycles = 10 msec (100hz) = the fastest spiking time
	// generally observed in the brain.  This can be useful for visualizing updates
	// at a granularity in between Cycle and Quarter.
	FastSpike

	// Quarter is typically 25 cycles = 25 msec (40hz) = 1/4 of the 100 msec alpha trial
	// This is also the GammaCycle (gamma = 40hz), but we use Quarter functionally
	// by virtue of there being 4 per AlphaCycle.
	Quarter

	// Phase is either Minus or Plus phase -- Minus = first 3 quarters, Plus = last quarter
	Phase

	// BetaCycle is typically 50 cycles = 50 msec (20 hz) = one beta-frequency cycle.
	// Gating in the basal ganglia and associated updating in prefrontal cortex
	// occurs at this frequency.
	BetaCycle

	// AlphaCycle is typically 100 cycles = 100 msec (10 hz) = one alpha-frequency cycle,
	// which is the fundamental unit of learning in posterior cortex.
	AlphaCycle

	// ThetaCycle is typically 200 cycles = 200 msec (5 hz) = two alpha-frequency cycles.
	// This is the modal duration of a saccade, the update frequency of medial temporal lobe
	// episodic memory, and the minimal predictive learning cycle (perceive an Alpha 1, predict on 2).
	ThetaCycle

	// Event is the smallest unit of naturalistic experience that coheres unto itself
	// (e.g., something that could be described in a sentence).
	// Typically this is on the time scale of a few seconds: e.g., reaching for
	// something, catching a ball.
	Event

	// Trial is one unit of behavior in an experiment -- it is typically environmentally
	// defined instead of endogenously defined in terms of basic brain rhythms.
	// In the minimal case it could be one AlphaCycle, but could be multiple, and
	// could encompass multiple Events (e.g., one event is fixation, next is stimulus,
	// last is response)
	Trial

	// Tick is one step in a sequence -- often it is useful to have Trial count
	// up throughout the entire Epoch but also include a Tick to count trials
	// within a Sequence
	Tick

	// Sequence is a sequential group of Trials (not always needed).
	Sequence

	// Block is a collection of Trials, Sequences or Events, often used in experiments
	// when conditions are varied across blocks.
	Block

	// Epoch is used in two different contexts.  In machine learning, it represents a
	// collection of Trials, Sequences or Events that constitute a "representative sample"
	// of the environment.  In the simplest case, it is the entire collection of Trials
	// used for training.  In electrophysiology, it is a timing window used for organizing
	// the analysis of electrode data.
	Epoch

	// Run is a complete run of a model / subject, from training to testing, etc.
	// Often multiple runs are done in an Expt to obtain statistics over initial
	// random weights etc.
	Run

	// Expt is an entire experiment -- multiple Runs through a given protocol / set of
	// parameters.
	Expt

	// Scene is a sequence of events that constitutes the next larger-scale coherent unit
	// of naturalistic experience corresponding e.g., to a scene in a movie.
	// Typically consists of events that all take place in one location over
	// e.g., a minute or so. This could be a paragraph or a page or so in a book.
	Scene

	// Episode is a sequence of scenes that constitutes the next larger-scale unit
	// of naturalistic experience e.g., going to the grocery store or eating at a
	// restaurant, attending a wedding or other "event".
	// This could be a chapter in a book.
	Episode

	TimeScalesN
)
