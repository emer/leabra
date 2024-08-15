// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"strconv"

	"github.com/emer/leabra/v2/leabra"
	//"github.com/emer/leabra/v2/pbwm"
)

// The PPTg passes on a positively rectified version of its input signal.
type PPTgLayer struct {
	leabra.Layer
	Ge      float32
	GePrev  float32
	SendAct float32
	DA      float32

	// gain on input activation
	DNetGain float32

	// activation threshold for passing through
	ActThreshold float32

	// clamp activation directly, after applying gain
	ClampActivation bool
}

func (ly *PPTgLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	return nil
}

func (ly *PPTgLayer) Defaults() {
	ly.Layer.Defaults()
}

// Add a Pedunculopontine Gyrus layer. Acts as a positive rectifier for its inputs.
func AddPPTgLayer(nt *Network, name string, nY, nX int) *PPTgLayer {
	rl := &PPTgLayer{}
	nt.AddLayerInit(rl, name, []int{nY, nX, 1, 1}, leabra.SuperLayer)
	return rl
}

func (ly *PPTgLayer) InitActs() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Act = 0
		nrn.ActSent = 0
	}
}

func (ly *PPTgLayer) GetDA() float32 {
	return ly.DA
}

func (ly *PPTgLayer) SetDA(da float32) {
	ly.DA = da
}

func (ly *PPTgLayer) QuarterFinal(ltime *leabra.Time) {
	ly.Layer.QuarterFinal(ltime)
	ly.Ge = ly.Neurons[0].Ge
	if ltime.PlusPhase {
		ly.GePrev = ly.Ge
	}
}

// GetMonitorVal retrieves a value for a trace of some quantity, possibly more than just a variable
func (ly *PPTgLayer) GetMonitorValue(data []string) float64 {
	var val float32
	idx, _ := strconv.Atoi(data[1])
	switch data[0] {
	case "Act":
		val = ly.Neurons[idx].Act
	case "Ge":
		val = ly.Neurons[idx].Ge
	case "GePrev":
		val = ly.GePrev
	case "TotalAct":
		val = TotalAct(ly)
	}
	return float64(val)
}

func (ly *PPTgLayer) ActFmG(_ *leabra.Time) {
	nrn := &ly.Neurons[0]
	geSave := nrn.Ge
	nrn.Ge = ly.DNetGain * (nrn.Ge - ly.GePrev)
	if nrn.Ge < ly.ActThreshold {
		nrn.Ge = 0.0
	}
	ly.Ge = nrn.Ge
	ly.SendAct = nrn.Act // mainly for debugging
	nrn.Act = nrn.Ge
	nrn.ActLrn = nrn.Act
	nrn.ActDel = 0.0
	nrn.Ge = geSave
	ly.Learn.AvgsFmAct(nrn)
}
