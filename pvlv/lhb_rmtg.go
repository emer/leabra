// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	_ "github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// Gain constants for LHbRMTg inputs
type LHbRMTgGains struct {
	All                float32 `desc:"final overall gain on everything"`
	VSPatchPosD1       float32 `desc:"patch D1 APPETITIVE pathway - versus pos PV outcomes"`
	VSPatchPosD2       float32 `desc:"patch D2 APPETITIVE pathway versus vspatch_pos_D1"`
	VSPatchPosDisinhib float32 `desc:"proportion of positive reward prediction error (RPE) to use if RPE results from a predicted omission of positive"`
	VSMatrixPosD1      float32 `desc:"gain on VS matrix D1 APPETITIVE guys"`
	VSMatrixPosD2      float32 `desc:"VS matrix D2 APPETITIVE"`
	VSPatchNegD1       float32 `desc:"VS patch D1 pathway versus neg PV outcomes"`
	VSPatchNegD2       float32 `desc:"VS patch D2 pathway versus vspatch_neg_D1"`
	VSMatrixNegD1      float32 `desc:"VS matrix D1 AVERSIVE"`
	VSMatrixNegD2      float32 `desc:"VS matrix D2 AVERSIVE"`
}

type LHbRMTgLayer struct {
	leabra.Layer
	RcvFrom       emer.LayNames
	Gains         LHbRMTgGains         `view:"inline"`
	PVNegDiscount float32              `desc:"reduction in effective PVNeg net value (when positive) so that negative outcomes can never be completely predicted away -- still allows for positive da for less-bad outcomes"`
	InternalState LHBRMTgInternalState // for debugging
}

var KiT_LHbRMTgLayer = kit.Types.AddType(&LHbRMTgLayer{}, leabra.LayerProps)

type LHBRMTgInternalState struct {
	VSPatchPosD1   float32
	VSPatchPosD2   float32
	VSPatchNegD1   float32
	VSPatchNegD2   float32
	VSMatrixPosD1  float32
	VSMatrixPosD2  float32
	VSMatrixNegD1  float32
	VSMatrixNegD2  float32
	PosPV          float32
	NegPV          float32
	VSPatchPosNet  float32
	VSPatchNegNet  float32
	VSMatrixPosNet float32
	VSMatrixNegNet float32
	NetPos         float32
	NetNeg         float32
}

func AddLHbRMTgLayer(nt *Network, name string) *LHbRMTgLayer {
	ly := LHbRMTgLayer{}
	nt.AddLayerInit(&ly, name, []int{1, 1, 1, 1}, emer.Hidden)
	ly.SetClass("LHbRMTg")
	ly.PVNegDiscount = 0.8 // from cemer
	return &ly
}

func (ly *LHbRMTgLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Gains.All = 1.0
	ly.Gains.VSPatchPosD1 = 1.0
	ly.Gains.VSPatchPosD2 = 1.0
	ly.PVNegDiscount = 0.8
	ly.Act.Clamp.Range.Min = -2.0
	ly.Act.Clamp.Range.Max = 2.0
	ly.Gains.VSPatchPosDisinhib = 0.2
	ly.Gains.VSMatrixPosD1 = 1.0
	ly.Gains.VSMatrixPosD2 = 1.0
	ly.Gains.VSPatchNegD1 = 1.0
	ly.Gains.VSPatchNegD2 = 1.0
	ly.Gains.VSMatrixNegD1 = 1.0
	ly.Gains.VSMatrixNegD2 = 1.0
}

func (ly *LHbRMTgLayer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.Neurons = make([]leabra.Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	if err != nil {
		return err
	}
	return err
}

func (ly *LHbRMTgLayer) ActFmG(ltime *leabra.Time) {
	if ltime.Quarter != 3 {
		return
	}
	var vsPatchPosD1, vsPatchPosD2, vsPatchNegD1, vsPatchNegD2, vsMatrixPosD1, vsMatrixPosD2,
		vsMatrixNegD1, vsMatrixNegD2, pvPos, pvNeg float32
	for _, lNm := range ly.RcvFrom {
		sLy := ly.Network.LayerByName(lNm).(leabra.LeabraLayer).AsLeabra()
		lyAct := TotalAct(sLy)
		switch lNm {
		case "VSPatchPosD1":
			vsPatchPosD1 = lyAct
		case "VSPatchPosD2":
			vsPatchPosD2 = lyAct
		case "VSPatchNegD1":
			vsPatchNegD1 = lyAct
		case "VSPatchNegD2":
			vsPatchNegD2 = lyAct
		case "VSMatrixPosD1":
			vsMatrixPosD1 = lyAct
		case "VSMatrixPosD2":
			vsMatrixPosD2 = lyAct
		case "VSMatrixNegD1":
			vsMatrixNegD1 = lyAct
		case "VSMatrixNegD2":
			vsMatrixNegD2 = lyAct
		case "PosPV":
			pvPos = lyAct
		case "NegPV":
			pvNeg = lyAct
		}
	}

	vsPatchPosNet := ly.Gains.VSPatchPosD1*vsPatchPosD1 - ly.Gains.VSPatchPosD2*vsPatchPosD2 // positive number net excitatory in LHb, i.e., the "dipper"
	if vsPatchPosNet < 0 {
		vsPatchPosNet *= ly.Gains.VSPatchPosDisinhib
	}

	vsPatchNegNet := ly.Gains.VSPatchNegD2*vsPatchNegD2 - ly.Gains.VSPatchNegD1*vsPatchNegD1 // positive number is net inhibitory in LHb - disinhibitory "burster"

	// pvneg_discount - should not fully predict away an expected punishment
	if vsPatchNegNet > 0 {
		//vspatch_neg_net = fminf(vspatch_neg_net,pv_neg); // helps mag .05, but
		// prevents burst after mag 1.0 training, then test 0.5
		vsPatchNegNet *= ly.PVNegDiscount
	}

	// net out the VS matrix D1 versus D2 pairs...WATCH the signs - double negatives!
	vsMatrixPosNet := ly.Gains.VSMatrixPosD1*vsMatrixPosD1 - ly.Gains.VSMatrixPosD2*vsMatrixPosD2 // positive number net inhibitory!
	//vsMatrixPosNet = math32.Max(0.0, vsMatrixPosNet); // restrict to positive net values
	vsMatrixNegNet := ly.Gains.VSMatrixNegD2*vsMatrixNegD2 - ly.Gains.VSMatrixNegD1*vsMatrixNegD1 // positive number net excitatory!
	//vsMatrixNegNet = math32.Max(0.0, vsMatrixNegNet); // restrict to positive net values

	// don't double count pv going through the matrix guys
	netPos := vsMatrixPosNet
	if pvPos != 0 {
		netPos = math32.Max(pvPos, vsMatrixPosNet)
	}

	netNeg := vsMatrixNegNet

	if pvNeg != 0 {
		// below can arise when same CS can predict either pos_pv or neg_pv probalistically
		if vsMatrixPosNet < 0 {
			netNeg = math32.Max(netNeg, math32.Abs(vsMatrixPosNet))
			netPos = 0 // don't double-count since transferred to net_neg in this case only
		}
		netNeg = math32.Max(pvNeg, netNeg)
	}

	netLHb := netNeg - netPos + vsPatchPosNet - vsPatchNegNet
	netLHb *= ly.Gains.All

	ly.InternalState.VSPatchPosD1 = vsPatchPosD1
	ly.InternalState.VSPatchPosD2 = vsPatchPosD2
	ly.InternalState.VSPatchNegD1 = vsPatchNegD1
	ly.InternalState.VSPatchNegD2 = vsPatchNegD2
	ly.InternalState.VSMatrixPosD1 = vsMatrixPosD1
	ly.InternalState.VSMatrixPosD2 = vsMatrixPosD2
	ly.InternalState.VSMatrixNegD1 = vsMatrixNegD1
	ly.InternalState.VSMatrixNegD2 = vsMatrixNegD2
	ly.InternalState.PosPV = pvPos
	ly.InternalState.NegPV = pvNeg
	ly.InternalState.VSPatchPosNet = vsPatchPosNet
	ly.InternalState.VSPatchNegNet = vsPatchNegNet
	ly.InternalState.VSMatrixPosNet = vsMatrixPosNet
	ly.InternalState.VSMatrixNegNet = vsMatrixNegNet
	ly.InternalState.NetPos = netPos
	ly.InternalState.NetNeg = netNeg

	for i := range ly.Neurons {
		ly.Neurons[i].Act = netLHb
		ly.Neurons[i].ActLrn = netLHb
		ly.Neurons[i].ActAvg = netLHb
		ly.Neurons[i].Ext = netLHb
		ly.Neurons[i].Ge = netLHb
	}
}

// GetMonitorVal retrieves a value for a trace of some quantity, possibly more than just a variable
func (ly *LHbRMTgLayer) GetMonitorVal(data []string) float64 {
	var val float32
	valType := data[0]
	switch valType {
	case "TotalAct":
		val = TotalAct(ly)

	case "VSPatchPosD1":
		val = ly.InternalState.VSPatchPosD1
	case "VSPatchPosD2":
		val = ly.InternalState.VSPatchPosD2
	case "VSPatchNegD1":
		val = ly.InternalState.VSPatchNegD1
	case "VSPatchNegD2":
		val = ly.InternalState.VSPatchNegD2
	case "VSMatrixPosD1":
		val = ly.InternalState.VSMatrixPosD1
	case "VSMatrixPosD2":
		val = ly.InternalState.VSMatrixPosD2
	case "VSMatrixNegD1":
		val = ly.InternalState.VSMatrixNegD1
	case "VSMatrixNegD2":
		val = ly.InternalState.VSMatrixNegD2
	case "PosPV":
		val = ly.InternalState.PosPV
	case "NegPV":
		val = ly.InternalState.NegPV
	case "VSPatchPosNet":
		val = ly.InternalState.VSPatchPosNet
	case "VSPatchNegNet":
		val = ly.InternalState.VSPatchNegNet
	case "VSMatrixPosNet":
		val = ly.InternalState.VSMatrixPosNet
	case "VSMatrixNegNet":
		val = ly.InternalState.VSMatrixNegNet
	case "NetPos":
		val = ly.InternalState.NetPos
	case "NetNeg":
		val = ly.InternalState.NetNeg

	default:
		val = ly.Neurons[0].Act
	}
	return float64(val)
}
