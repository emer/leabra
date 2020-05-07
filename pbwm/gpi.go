// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"fmt"
	"log"

	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/bitflag"
	"github.com/goki/ki/kit"
)

////////////////////////////////////////////////////////////////////
// GPiThalPrjn

// GPiThalPrjn accumulates per-prjn raw conductance that is needed for separately weighting
// NoGo vs. Go inputs
type GPiThalPrjn struct {
	deep.Prjn           // access as .Prjn
	GeRaw     []float32 `desc:"per-recv, per-prjn raw excitatory input"`
}

var KiT_GPiThalPrjn = kit.Types.AddType(&GPiThalPrjn{}, deep.PrjnProps)

func (pj *GPiThalPrjn) Build() error {
	err := pj.Prjn.Build()
	if err != nil {
		return err
	}
	rsh := pj.Recv.Shape()
	rlen := rsh.Len()
	pj.GeRaw = make([]float32, rlen)
	return nil
}

func (pj *GPiThalPrjn) InitGInc() {
	pj.Prjn.InitGInc()
	for ri := range pj.GeRaw {
		pj.GeRaw[ri] = 0
	}
}

// RecvGInc increments the receiver's GeInc or GiInc from that of all the projections.
func (pj *GPiThalPrjn) RecvGInc() {
	rlay := pj.Recv.(leabra.LeabraLayer).AsLeabra()
	if pj.Typ == emer.Inhib {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GiInc += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	} else {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			pj.GeRaw[ri] += pj.GInc[ri]
			rn.GeInc += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	}
}

////////////////////////////////////////////////////////////////////
// GPiThalLayer

// GPiTimingParams has timing parameters for gating in the GPiThal layer
type GPiTimingParams struct {
	GateQtr leabra.Quarters `desc:"Quarter(s) when gating takes place, typically Q1 and Q3, which is the quarter prior to the deep BurstQtr when deep layer updating takes place. Note: this is a bitflag and must be accessed using bitflag.Set / Has etc routines, 32 bit versions."`
	Cycle   int             `def:"18" desc:"cycle within Qtr to determine if activation over threshold for gating.  We send GateState updates on this cycle either way."`
}

func (gt *GPiTimingParams) Defaults() {
	gt.SetGateQtr(leabra.Q1)
	gt.SetGateQtr(leabra.Q3)
	gt.Cycle = 18
}

// SetGateQtr sets given gating quarter (adds to any existing) -- Q1, 3 by default
func (gt *GPiTimingParams) SetGateQtr(qtr leabra.Quarters) {
	bitflag.Set32((*int32)(&gt.GateQtr), int(qtr))
}

// IsGateQtr returns true if the given quarter (0-3) is set as a Gating quarter
func (gt *GPiTimingParams) IsGateQtr(qtr int) bool {
	qmsk := leabra.Quarters(1 << uint(qtr))
	if gt.GateQtr&qmsk != 0 {
		return true
	}
	return false
}

// GPiGateParams has gating parameters for gating in GPiThal layer, including threshold
type GPiGateParams struct {
	GeGain float32 `def:"3" desc:"extra netinput gain factor to compensate for reduction in Ge from subtracting away NoGo -- this is *IN ADDITION* to adding the NoGo factor as an extra gain: Ge = (GeGain + NoGo) * (GoIn - NoGo * NoGoIn)"`
	NoGo   float32 `min:"0" def:"1,0.1" desc:"how much to weight NoGo inputs relative to Go inputs (which have an implied weight of 1 -- this also up-scales overall Ge to compensate for subtraction"`
	Thr    float32 `def:"0.2" desc:"threshold for gating, applied to activation -- when any GPiThal unit activation gets above this threshold, it counts as having gated, driving updating of GateState which is broadcast to other layers that use the gating signal"`
	ThrAct bool    `def:"true" desc:"Act value of GPiThal unit reflects gating threshold: if below threshold, it is zeroed -- see ActLrn for underlying non-thresholded activation"`
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
	ActG float32 `desc:"gating activation -- the activity value when gating occurred in this pool"`
}

// GPiThalLayer represents the combined Winner-Take-All dynamic of GPi (SNr) and Thalamus.
// It is the final arbiter of gating in the BG, weighing Go (direct) and NoGo (indirect)
// inputs from MatrixLayers (indirectly via GPe layer in case of NoGo).
// Use 4D structure for this so it matches 4D structure in Matrix layers
type GPiThalLayer struct {
	GateLayer
	Timing   GPiTimingParams `view:"inline" desc:"timing parameters determining when gating happens"`
	Gate     GPiGateParams   `view:"inline" desc:"gating parameters determining threshold for gating etc"`
	SendTo   []string        `desc:"list of layers to send GateState to"`
	GPiNeurs []GPiNeuron     `desc:"slice of GPiNeuron state for this layer -- flat list of len = Shape.Len().  You must iterate over index and use pointer to modify values."`
}

var KiT_GPiThalLayer = kit.Types.AddType(&GPiThalLayer{}, deep.LayerProps)

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

// UnitValByIdx returns value of given PBWM-specific variable by variable index
// and flat neuron index (from layer or neuron-specific one).
func (ly *GPiThalLayer) UnitValByIdx(vidx NeuronVars, idx int) float32 {
	if vidx != ActG {
		return ly.GateLayer.UnitValByIdx(vidx, idx)
	}
	gnrn := &ly.GPiNeurs[idx]
	return gnrn.ActG
}

// SendToMatrixPFC adds standard SendTo layers for PBWM: MatrixGo, NoGo, PFCmnt, PFCout
// with optional prefix -- excludes mnt, out cases if corresp shape = 0
func (ly *GPiThalLayer) SendToMatrixPFC(prefix string) {
	pfcprefix := "PFC"
	if prefix != "" {
		pfcprefix = prefix
	}
	std := []string{prefix + "MatrixGo", prefix + "MatrixNoGo", pfcprefix + "mnt", pfcprefix + "out"}
	ly.SendTo = make([]string, 2)
	for i, s := range std {
		nm := s
		switch {
		case i < 2:
			ly.SendTo[i] = nm
		case i == 2:
			if ly.GateShp.MaintX > 0 {
				ly.SendTo = append(ly.SendTo, nm)
				ly.SendTo = append(ly.SendTo, nm+"D")
			}
		case i == 3:
			if ly.GateShp.OutX > 0 {
				ly.SendTo = append(ly.SendTo, nm)
				ly.SendTo = append(ly.SendTo, nm+"D")
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

// MatrixPrjns returns the recv prjns from Go and NoGo MatrixLayer pathways -- error if not
// found or if prjns are not of the GPiThalPrjn type
func (ly *GPiThalLayer) MatrixPrjns() (goPrjn, nogoPrjn *GPiThalPrjn, err error) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		gp, ok := p.(*GPiThalPrjn)
		if !ok {
			err = fmt.Errorf("GPiThalLayer must have RecvPrjn's of type GPiThalPrjn")
			return
		}
		slay := p.SendLay()
		mlay, ok := slay.(*MatrixLayer)
		if ok {
			if mlay.DaR == D1R {
				goPrjn = gp
			} else {
				nogoPrjn = gp
			}
		} else {
			nogoPrjn = gp
		}
	}
	if goPrjn == nil || nogoPrjn == nil {
		err = fmt.Errorf("GPiThalLayer must have RecvPrjn's from a MatrixLayer D1R (Go) and another NoGo layer")
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

// Build constructs the layer state, including calling Build on the projections.
func (ly *GPiThalLayer) Build() error {
	err := ly.GateLayer.Build()
	if err != nil {
		return err
	}
	ly.GPiNeurs = make([]GPiNeuron, len(ly.Neurons))
	_, _, err = ly.MatrixPrjns()
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

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
// should already have presented the external input to the network at this point.
// need to clear incrementing GeRaw from prjns
func (ly *GPiThalLayer) AlphaCycInit() {
	ly.GateLayer.AlphaCycInit()
	ly.LeabraLay.InitGInc()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *GPiThalLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	goPrjn, nogoPrjn, _ := ly.MatrixPrjns()
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.GRawFmInc(nrn) // for inhib, just in case
		goRaw := goPrjn.GeRaw[ni]
		nogoRaw := nogoPrjn.GeRaw[ni]
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
	gateQtr := ly.Timing.IsGateQtr(ltime.Quarter)
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
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			gnr := &ly.GPiNeurs[ni]
			gnr.ActG = nrn.Act
		}
	}
}
