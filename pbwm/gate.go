// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/etable/minmax"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/ints"
)

// GateShape defines the shape of the outer pool dimensions of gating layers,
// organized into Maint and Out subsets which are arrayed along the X axis
// with Maint first (to the left) then Out.  Individual layers may only
// represent Maint or Out subsets of this overall shape, but all need
// to have this coordinated shape information to be able to share gating
// state information.  Indexes into GateStates for MaintOut combined layers
// are organized so that all of the states for Maint are first and arranged
// as they would be in a pure Maint layer, and likewise for the subsequent
// Out states.
type GateShape struct {
	Y      int `desc:"overall shape dimensions for the full set of gating pools, e.g., as present in the Matrix and GPiThal levels"`
	MaintX int `desc:"how many pools in the X dimension are Maint gating pools -- rest are Out"`
	OutX   int `desc:"how many pools in the X dimension are Out gating pools -- comes after Maint"`
}

// Set sets the shape parameters: number of Y dimension pools, and
// numbers of maint and out pools along X axis
func (gs *GateShape) Set(nY, maintX, outX int) {
	gs.Y = nY
	gs.MaintX = maintX
	gs.OutX = outX
}

// TotX returns the total number of X-axis pools (Maint + Out)
func (gs *GateShape) TotX() int {
	return gs.MaintX + gs.OutX
}

// Index returns the index into GateStates for given 2D pool coords
// for given GateType.  For Maint and Out, it is just the basic pool
// index (0-based) for a layer of that shape, but for MaintOut layers
// that have both types, it obeys the separation between Maint (first)
// and Out (second).
func (gs *GateShape) Index(pY, pX int, typ GateTypes) int {
	switch typ {
	case Maint:
		if gs.MaintX == 0 {
			return 0
		}
		return pY*gs.MaintX + pX
	case Out:
		if gs.OutX == 0 {
			return 0
		}
		return pY*gs.OutX + pX
	case MaintOut:
		if pX < gs.MaintX {
			return pY*gs.MaintX + pX
		} else {
			return pY*gs.OutX + (pX - gs.MaintX)
		}
	}
	return 0
}

// Index1D returns the index into GateStates for given 1D pool idx (0-based)
// appropriate for given GateType. For Maint and Out, it is just the
// input index, but for MaintOut layers that have both types,
// it obeys the separation between Maint (first) and Out (second).
func (gs *GateShape) Index1D(idx int, typ GateTypes) int {
	if typ != MaintOut {
		return idx
	}
	// convert to 2D and use that
	tX := gs.TotX()
	pY := idx / tX
	pX := idx % tX
	return gs.Index(pY, pX, typ)
}

// FullIndex1D returns the index into full MaintOut GateStates
// for given 1D pool idx (0-based) *from given GateType*.
func (gs *GateShape) FullIndex1D(idx int, fmTyp GateTypes) int {
	switch fmTyp {
	case Maint:
		return idx
	case Out:
		if gs.OutX == 0 {
			return 0
		}
		// convert to 2D and use that
		pY := idx / gs.OutX
		pX := idx%gs.OutX + gs.MaintX
		return gs.Index(pY, pX, MaintOut)
	case MaintOut:
		return gs.Index1D(idx, MaintOut)
	}
	return 0
}

//////////////////////////////////////////////////////////////////////////////
// GateState

// GateState is gating state values stored in layers that receive thalamic gating signals
// including MatrixLayer, PFCLayer, GPiThal layer, etc -- use GateLayer as base layer to include.
type GateState struct {
	Act   float32         `desc:"gating activation value, reflecting thalamic gating layer activation at time of gating (when Now = true) -- will be 0 if gating below threshold for this pool, and prior to first Now for AlphaCycle"`
	Now   bool            `desc:"gating timing signal -- true if this is the moment when gating takes place"`
	Cnt   int             `copy:"-" desc:"unique to each layer -- not copied.  Generally is a counter for interval between gating signals -- starts at -1, goes to 0 at first gating, counts up from there for subsequent gating.  Can be reset back to -1 when gate is reset (e.g., output gating) and counts down from -1 while not gating."`
	GeRaw minmax.AvgMax32 `copy:"-" desc:"not copies: average and max Ge Raw excitatory conductance values -- before being influenced by gating signals"`
}

// Init initializes the values -- call during InitActs()
func (gs *GateState) Init() {
	gs.Act = 0
	gs.Now = false
	gs.Cnt = -1
	gs.GeRaw.Init()
}

// CopyFrom copies from another GateState -- only the Act and Now signals are copied
func (gs *GateState) CopyFrom(fm *GateState) {
	gs.Act = fm.Act
	gs.Now = fm.Now
}

//////////////////////////////////////////////////////////////////////////////
// GateLayer

// GateLayer is a layer that cares about thalamic (BG) gating signals, and has
// slice of GateState fields that a gating layer will update.
type GateLayer struct {
	ModLayer
	GateShp    GateShape   `desc:"shape of overall Maint + Out gating system that this layer is part of"`
	GateStates []GateState `desc:"slice of gating state values for this layer, one for each separate gating pool, according to its GateType.  For MaintOut, it is ordered such that 0:MaintN are Maint and MaintN:n are Out"`
}

func (ly *GateLayer) AsGate() *GateLayer {
	return ly
}

func (ly *GateLayer) GateShape() *GateShape {
	return &ly.GateShp
}

// note: each layer must define its own GateType() method!

// GateState returns the GateState for given pool index (0 based) on this layer
func (ly *GateLayer) GateState(poolIdx int) *GateState {
	myt := ly.LeabraLay.(GateLayerer).GateType()
	return &ly.GateStates[ly.GateShp.Index1D(poolIdx, myt)]
}

// SetGateState sets the GateState for given pool index (individual pools start at 1) on this layer
func (ly *GateLayer) SetGateState(poolIdx int, state *GateState) {
	myt := ly.LeabraLay.(GateLayerer).GateType()
	gs := &ly.GateStates[ly.GateShp.Index1D(poolIdx, myt)]
	gs.CopyFrom(state)
}

// SetGateStates sets the GateStates from given source states, of given gating type
func (ly *GateLayer) SetGateStates(states []GateState, typ GateTypes) {
	myt := ly.LeabraLay.(GateLayerer).GateType()
	if myt < MaintOut && typ < MaintOut && myt != typ { // mismatch
		return
	}
	switch {
	case myt == typ:
		mx := ints.MinInt(len(states), len(ly.GateStates))
		for i := 0; i < mx; i++ {
			ly.SetGateState(i, &states[i])
		}
	case myt == Maint: // typ must == MaintOut at this point
		mx := len(ly.GateStates)
		for i := 0; i < mx; i++ {
			gs := &ly.GateStates[i]
			src := &states[ly.GateShp.FullIndex1D(i, myt)]
			gs.CopyFrom(src)
		}
	}
}

// UnitValByIdx returns value of given variable by variable index
// and flat neuron index (from layer or neuron-specific one).
// First indexes are ModNeuronVars
func (ly *GateLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	nrn := &ly.Neurons[idx]
	gs := ly.GateState(int(nrn.SubPool) - 1) // 0-based
	switch vidx {
	case DA:
		return ly.DA
	case ACh:
		return ly.ACh
	case SE:
		return ly.SE
	case GateAct:
		return gs.Act
	case GateNow:
		if gs.Now {
			return 1
		}
		return 0
	case GateCnt:
		return float32(gs.Cnt)
	}
	return 0
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *GateLayer) Build() error {
	err := ly.ModLayer.Build()
	if err != nil {
		return err
	}
	ly.GateStates = make([]GateState, len(ly.Pools))
	return err
}

func (ly *GateLayer) InitActs() {
	ly.ModLayer.InitActs()
	for si := range ly.GateStates {
		gs := &ly.GateStates[si]
		gs.Init()
	}
}

// AvgMaxGeRaw computes the average and max GeRaw stats
func (ly *GateLayer) AvgMaxGeRaw(ltime *leabra.Time) {
	for si := range ly.GateStates {
		gs := &ly.GateStates[si]
		pl := &ly.Pools[si+1]
		gs.GeRaw.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			gs.GeRaw.UpdateVal(nrn.GeRaw, ni)
		}
		gs.GeRaw.CalcAvg()
	}
}

//////////////////////////////////////////////////////////////////////////////
// GateLayerer interface

// GateLayerer is an interface for GateLayer layers
type GateLayerer interface {
	// AsGate returns the layer as a GateLayer layer, for direct access to fields
	AsGate() *GateLayer

	// GateType returns the type of gating supported by this layer
	GateType() GateTypes

	// GateShape returns the shape of gating system that this layer is part of
	GateShape() *GateShape

	// GateState returns the GateState for given pool index (0-based) on this layer
	GateState(poolIdx int) *GateState

	// SetGateState sets the GateState for given pool index (0-based) on this layer
	SetGateState(poolIdx int, state *GateState)

	// SetGateStates sets the GateStates from given source states, of given gating type
	SetGateStates(states []GateState, typ GateTypes)
}
