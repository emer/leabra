// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"fmt"
	"log"

	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

////////////////////////////////////////////////////////////////////
// GPiThalPath

// GPiThalPath accumulates per-path raw conductance that is needed for separately weighting
// NoGo vs. Go inputs
type GPiThalPath struct {
	leabra.Path // access as .Path

	// per-recv, per-path raw excitatory input
	GeRaw []float32
}

func (pj *GPiThalPath) Build() error {
	err := pj.Path.Build()
	if err != nil {
		return err
	}
	rsh := pj.Recv.Shape()
	rlen := rsh.Len()
	pj.GeRaw = make([]float32, rlen)
	return nil
}

func (pj *GPiThalPath) InitGInc() {
	pj.Path.InitGInc()
	for ri := range pj.GeRaw {
		pj.GeRaw[ri] = 0
	}
}

// RecvGInc increments the receiver's GeInc or GiInc from that of all the pathways.
func (pj *GPiThalPath) RecvGInc() {
	rlay := pj.Recv.(leabra.LeabraLayer).AsLeabra()
	if pj.Typ == emer.Inhib {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GiRaw += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	} else {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			ginc := pj.GInc[ri]
			pj.GeRaw[ri] += ginc
			rn.GeRaw += ginc
			pj.GInc[ri] = 0
		}
	}
}

////////////////////////////////////////////////////////////////////
// GPiThalLayer

// GPiTimingParams has timing parameters for gating in the GPiThal layer
type GPiTimingParams struct {

	// Quarter(s) when gating takes place, typically Q1 and Q3, which is the quarter prior to the PFC GateQtr when deep layer updating takes place. Note: this is a bitflag and must be accessed using bitflag.Set / Has etc routines, 32 bit versions.
	GateQtr leabra.Quarters

	// cycle within Qtr to determine if activation over threshold for gating.  We send GateState updates on this cycle either way.
	Cycle int `def:"18"`
}

func (gt *GPiTimingParams) Defaults() {
	gt.GateQtr.SetFlag(true, leabra.Q1)
	gt.GateQtr.SetFlag(true, leabra.Q3)
	gt.Cycle = 18
}

// GPiGateParams has gating parameters for gating in GPiThal layer, including threshold
type GPiGateParams struct {

	// extra netinput gain factor to compensate for reduction in Ge from subtracting away NoGo -- this is *IN ADDITION* to adding the NoGo factor as an extra gain: Ge = (GeGain + NoGo) * (GoIn - NoGo * NoGoIn)
	GeGain float32 `def:"3"`

	// how much to weight NoGo inputs relative to Go inputs (which have an implied weight of 1 -- this also up-scales overall Ge to compensate for subtraction
	NoGo float32 `min:"0" def:"1,0.1"`

	// threshold for gating, applied to activation -- when any GPiThal unit activation gets above this threshold, it counts as having gated, driving updating of GateState which is broadcast to other layers that use the gating signal
	Thr float32 `def:"0.2"`

	// Act value of GPiThal unit reflects gating threshold: if below threshold, it is zeroed -- see ActLrn for underlying non-thresholded activation
	ThrAct bool `def:"true"`
}

func (gp *GPiGateParams) Defaults() {
	gp.GeGain = 3
	gp.NoGo = 1
	gp.Thr = 0.2
	gp.ThrAct = true
}

// GeRaw returns the net GeRaw from go, nogo specific values
func (gp *GPiGateParams) GeRaw(goRaw, nogoRaw float32) float32 {
	return (gp.GeGain + gp.NoGo) * (goRaw - gp.NoGo*nogoRaw)
}

// GPiNeuron contains extra variables for GPiThalLayer neurons -- stored separately
type GPiNeuron struct {

	// gating activation -- the activity value when gating occurred in this pool
	ActG float32
}

// GPiThalLayer represents the combined Winner-Take-All dynamic of GPi (SNr) and Thalamus.
// It is the final arbiter of gating in the BG, weighing Go (direct) and NoGo (indirect)
// inputs from MatrixLayers (indirectly via GPe layer in case of NoGo).
// Use 4D structure for this so it matches 4D structure in Matrix layers
type GPiThalLayer struct {
	GateLayer

	// timing parameters determining when gating happens
	Timing GPiTimingParams `view:"inline"`

	// gating parameters determining threshold for gating etc
	Gate GPiGateParams `view:"inline"`

	// list of layers to send GateState to
	SendTo []string

	// slice of GPiNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values.
	GPiNeurs []GPiNeuron
}

// Sel: "GPiThalLayer", Desc: "defaults ",
// 	Params: params.Params{
// 		"Layer.Inhib.Layer.Gi":     "1.8",
// 		"Layer.Inhib.Layer.FB":     "0.2",
// 		"Layer.Inhib.Pool.On":      "false",
// 		"Layer.Inhib.ActAvg.Init":  "1",
// 		"Layer.Inhib.ActAvg.Fixed": "true",
// 	}}

func (ly *GPiThalLayer) Defaults() {
	ly.GateLayer.Defaults()
	ly.Timing.Defaults()
	ly.Gate.Defaults()
	ly.Inhib.Layer.Gi = 1.8
	ly.Inhib.Layer.FB = 0.2
	ly.Inhib.Pool.On = false
	ly.Inhib.ActAvg.Fixed = true
	ly.Inhib.ActAvg.Init = 1
}

func (ly *GPiThalLayer) GateType() GateTypes {
	return MaintOut // always both
}

// UnitValueByIndex returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *GPiThalLayer) UnitValueByIndex(vidx NeurVars, idx int) float32 {
	if vidx != ActG {
		return ly.GateLayer.UnitValueByIndex(vidx, idx)
	}
	gnrn := &ly.GPiNeurs[idx]
	return gnrn.ActG
}

// SendToMatrixPFC adds standard SendTo layers for PBWM: MatrixGo, NoGo, PFCmntD, PFCoutD
// with optional prefix -- excludes mnt, out cases if corresp shape = 0
func (ly *GPiThalLayer) SendToMatrixPFC(prefix string) {
	pfcprefix := "PFC"
	if prefix != "" {
		pfcprefix = prefix
	}
	std := []string{prefix + "MatrixGo", prefix + "MatrixNoGo", pfcprefix + "mntD", pfcprefix + "outD"}
	ly.SendTo = make([]string, 2)
	for i, s := range std {
		nm := s
		switch {
		case i < 2:
			ly.SendTo[i] = nm
		case i == 2:
			if ly.GateShp.MaintX > 0 {
				ly.SendTo = append(ly.SendTo, nm)
			}
		case i == 3:
			if ly.GateShp.OutX > 0 {
				ly.SendTo = append(ly.SendTo, nm)
			}
		}
	}
}

// SendGateShape send GateShape info to all SendTo layers -- convenient config-time
// way to ensure all are consistent -- also checks validity of SendTo's
func (ly *GPiThalLayer) SendGateShape() error {
	var lasterr error
	for _, lnm := range ly.SendTo {
		tly, err := ly.Network.LayerByNameTry(lnm)
		if err != nil {
			log.Printf("GPiThalLayer %s SendGateShape: %v\n", ly.Name(), err)
			lasterr = err
		}
		gl, ok := tly.(GateLayerer)
		if !ok {
			err = fmt.Errorf("GPiThalLayer %s SendGateShape: can only send to layers that implement the GateLayerer interface (i.e., are based on GateLayer)", ly.Name())
			log.Println(err)
			lasterr = err
			continue
		}
		gll := gl.AsGate()
		gll.GateShp = ly.GateShp
	}
	return lasterr
}

// MatrixPaths returns the recv paths from Go and NoGo MatrixLayer pathways -- error if not
// found or if paths are not of the GPiThalPath type
func (ly *GPiThalLayer) MatrixPaths() (goPath, nogoPath *GPiThalPath, err error) {
	for _, p := range ly.RecvPaths {
		if p.IsOff() {
			continue
		}
		gp, ok := p.(*GPiThalPath)
		if !ok {
			err = fmt.Errorf("GPiThalLayer must have RecvPath's of type GPiThalPath")
			return
		}
		slay := p.SendLay()
		mlay, ok := slay.(*MatrixLayer)
		if ok {
			if mlay.DaR == D1R {
				goPath = gp
			} else {
				nogoPath = gp
			}
		} else {
			nogoPath = gp
		}
	}
	if goPath == nil || nogoPath == nil {
		err = fmt.Errorf("GPiThalLayer must have RecvPath's from a MatrixLayer D1R (Go) and another NoGo layer")
	}
	return
}

// SendToCheck is called during Build to ensure that SendTo layers are valid
func (ly *GPiThalLayer) SendToCheck() error {
	var lasterr error
	for _, lnm := range ly.SendTo {
		tly, err := ly.Network.LayerByNameTry(lnm)
		if err != nil {
			log.Printf("GPiThalLayer %s SendToCheck: %v\n", ly.Name(), err)
			lasterr = err
		}
		_, ok := tly.(GateLayerer)
		if !ok {
			err = fmt.Errorf("GPiThalLayer %s SendToCheck: can only send to layers that implement the GateLayerer interface (i.e., are based on GateLayer)", ly.Name())
			log.Println(err)
			lasterr = err
		}
	}
	return lasterr
}

// AddSendTo adds given layer name to list of those to send DA to
func (ly *GPiThalLayer) AddSendTo(laynm string) {
	ly.SendTo = append(ly.SendTo, laynm)
}

// Build constructs the layer state, including calling Build on the pathways.
func (ly *GPiThalLayer) Build() error {
	err := ly.GateLayer.Build()
	if err != nil {
		return err
	}
	ly.GPiNeurs = make([]GPiNeuron, len(ly.Neurons))
	_, _, err = ly.MatrixPaths()
	if err != nil {
		log.Println(err)
	}
	err = ly.SendToCheck()
	if err != nil {
		log.Println(err)
	}
	return err
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *GPiThalLayer) InitActs() {
	ly.GateLayer.InitActs()
	for ni := range ly.GPiNeurs {
		gnr := &ly.GPiNeurs[ni]
		gnr.ActG = 0
	}
}

func (ly *GPiThalLayer) AlphaCycInit(updtActAvg bool) {
	ly.GateLayer.AlphaCycInit(updtActAvg)
	ly.LeabraLay.InitGInc()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *GPiThalLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	goPath, nogoPath, _ := ly.MatrixPaths()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		goRaw := goPath.GeRaw[ni]
		nogoRaw := nogoPath.GeRaw[ni]
		nrn.GeRaw = ly.Gate.GeRaw(goRaw, nogoRaw)
		ly.Act.GeFmRaw(nrn, nrn.GeRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
}

// GateSend updates gating state and sends it along to other layers
func (ly *GPiThalLayer) GateSend(ltime *leabra.Time) {
	ly.GateFmAct(ltime)
	ly.SendGateStates()
}

// GateFmAct updates GateState from current activations, at time of gating
func (ly *GPiThalLayer) GateFmAct(ltime *leabra.Time) {
	gateQtr := ly.Timing.GateQtr.HasFlag(ltime.Quarter)
	qtrCyc := ltime.QuarterCycle()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		gs := ly.GateState(int(nrn.SubPool) - 1)
		if ltime.Quarter == 0 && qtrCyc == 0 {
			gs.Act = 0 // reset at start
		}
		if gateQtr && qtrCyc == ly.Timing.Cycle { // gating
			gs.Now = true
			if nrn.Act < ly.Gate.Thr { // didn't gate
				gs.Act = 0 // not over thr
				if ly.Gate.ThrAct {
					gs.Act = 0
				}
				if gs.Cnt >= 0 {
					gs.Cnt++
				} else if gs.Cnt < 0 {
					gs.Cnt--
				}
			} else { // did gate
				gs.Cnt = 0
				gs.Act = nrn.Act
			}
		} else {
			gs.Now = false
		}
	}
}

// SendGateStates sends GateStates to other layers
func (ly *GPiThalLayer) SendGateStates() {
	myt := ly.GateType()
	for _, lnm := range ly.SendTo {
		gl := ly.Network.LayerByName(lnm).(GateLayerer)
		gl.SetGateStates(ly.GateStates, myt)
	}
}

// RecGateAct records the gating activation from current activation, when gating occcurs
// based on GateState.Now
func (ly *GPiThalLayer) RecGateAct(ltime *leabra.Time) {
	for gi := range ly.GateStates {
		gs := &ly.GateStates[gi]
		if !gs.Now { // not gating now
			continue
		}
		pl := &ly.Pools[1+gi]
		for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			gnr := &ly.GPiNeurs[ni]
			gnr.ActG = nrn.Act
		}
	}
}
