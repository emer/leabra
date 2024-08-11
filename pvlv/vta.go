// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"strconv"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
	"github.com/emer/leabra/v2/rl"
)

// Gain constants for inputs to the VTA
type VTADAGains struct {

	// overall multiplier for dopamine values
	DA float32

	// gain on bursts from PPTg
	PPTg float32

	// gain on dips/bursts from LHbRMTg
	LHb float32

	// gain on positive PV component of total phasic DA signal (net after subtracting VSPatchIndir (PVi) shunt signal)
	PV float32

	// gain on VSPatch pathway that shunts bursting in VTA (for VTAp = VSPatchPosD1, for VTAn = VSPatchNegD2)
	PVIBurstShunt float32

	// gain on VSPatch pathway that opposes shunting of bursting in VTA (for VTAp = VSPatchPosD2, for VTAn = VSPatchNegD1)
	PVIAntiBurstShunt float32

	// gain on VSPatch pathway that shunts dipping of VTA (currently only VTAp supported = VSPatchNegD2) -- optional and somewhat controversial
	PVIDipShunt float32

	// gain on VSPatch pathway that opposes the shunting of dipping in VTA (currently only VTAp supported = VSPatchNegD1)
	PVIAntiDipShunt float32
}

// VTA internal state
type VTALayer struct {
	rl.ClampDaLayer
	SendVal float32

	// VTA layer DA valence, positive or negative
	Valence Valence

	// set a tonic 'dopamine' (DA) level (offset to add to da values)
	TonicDA float32

	// gains for various VTA inputs
	DAGains  VTADAGains `view:"inline"`
	RecvFrom map[string]emer.Layer

	// input values--for debugging only
	InternalState VTAState
}

// monitoring and debugging only. Received values from all inputs
type VTAState struct {
	PPTgDAp    float32
	LHbDA      float32
	PosPVAct   float32
	VSPosPVI   float32
	VSNegPVI   float32
	BurstLHbDA float32
	DipLHbDA   float32
	TotBurstDA float32
	TotDipDA   float32
	NetDipDA   float32
	NetDA      float32
	SendVal    float32
}

func AddVTALayer(nt *Network, name string, val Valence) *VTALayer {
	ly := &VTALayer{Valence: val}
	nt.AddLayerInit(ly, name, []int{1, 1}, emer.Hidden)
	return ly
}

func (ly *VTALayer) Build() error {
	net := ly.Network
	ly.RecvFrom = map[string]emer.Layer{}
	for _, lyNm := range []string{
		"PPTg", "LHbRMTg", "PosPV", "NegPV", "VSPatchPosD1", "VSPatchPosD2",
		"VSPatchNegD1", "VSPatchNegD2"} {
		ly.RecvFrom[lyNm] = net.LayerByName(lyNm).(leabra.LeabraLayer).AsLeabra()
	}
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("build Layer %v: no units specified in Shape", ly.Name)
	}
	ly.Neurons = make([]leabra.Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPaths()
	if err != nil {
		return err
	}
	return err
}

func (ly *VTALayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.VmRange.Min = -2.0
	ly.Act.VmRange.Max = 2.0
	ly.Act.Clamp.Range.Min = -2.0
	ly.Act.Clamp.Range.Max = 2.0

	ly.TonicDA = 0

	ly.DAGains.Defaults()
}

func (dag *VTADAGains) Defaults() {
	dag.DA = 1.0
	dag.PPTg = 1.1
	dag.LHb = 1.0
	dag.PV = 1.0
	dag.PVIBurstShunt = 1.9
	dag.PVIAntiBurstShunt = 2
	dag.PVIDipShunt = 0.0
	dag.PVIAntiDipShunt = 0.0

}

// GetMonitorVal is for monitoring during run. Includes values beyond the scope of neuron fields.
func (ly *VTALayer) GetMonitorValue(data []string) float64 {
	var val float32
	valType := data[0]
	unitIndex, _ := strconv.Atoi(data[1])
	switch valType {
	case "PPTgDAp":
		val = ly.InternalState.PPTgDAp
	case "LHbDA":
		val = ly.InternalState.LHbDA
	case "PosPVAct":
		val = ly.InternalState.PosPVAct
	case "VSPosPVI":
		val = ly.InternalState.VSPosPVI
	case "VSNegPVI":
		val = ly.InternalState.VSNegPVI
	case "BurstLHbDA":
		val = ly.InternalState.BurstLHbDA
	case "DipLHbDA":
		val = ly.InternalState.DipLHbDA
	case "TotBurstDA":
		val = ly.InternalState.TotBurstDA
	case "TotDipDA":
		val = ly.InternalState.TotDipDA
	case "NetDipDA":
		val = ly.InternalState.NetDipDA
	case "NetDA":
		val = ly.InternalState.NetDA
	case "SendVal":
		val = ly.InternalState.SendVal
	case "TotalAct":
		val = TotalAct(ly)
	case "PoolActAvg":
		val = ly.Pools[unitIndex].Inhib.Act.Avg
	case "PoolActMax":
		val = ly.Pools[unitIndex].Inhib.Act.Max
	case "Act":
		val = ly.Neurons[unitIndex].Act
	case "DA":
		val = ly.DA
	default:
		val = ly.Neurons[0].Act
	}
	return float64(val)
}

func (ly *VTALayer) ActFmG(ltime *leabra.Time) {
	if ltime.Quarter == leabra.Q4 {
		ly.VTAAct(ltime)
	} else {
		nrn := &ly.Neurons[0]
		nrn.ActLrn = 0
		nrn.Act = 0
		nrn.Ge = 0
		ly.SendVal = 0
	}
	ly.DA = 0
}

func (ly *VTALayer) CyclePost(_ *leabra.Time) {
	ly.SendDA.SendDA(ly.Network, ly.SendVal)
}

func (ly *VTALayer) VTAAct(ltime *leabra.Time) {
	if ly.Valence == POS {
		ly.VTAActP(ltime)
	} else {
		ly.VTAActN(ltime)
	}
}

// VTAp activation
func (ly *VTALayer) VTAActP(_ *leabra.Time) {
	pptGLy := ly.RecvFrom["PPTg"]
	lhbLy := ly.RecvFrom["LHbRMTg"]
	posPVLy := ly.RecvFrom["PosPV"]
	vsPatchPosD1Ly := ly.RecvFrom["VSPatchPosD1"]
	vsPatchPosD2Ly := ly.RecvFrom["VSPatchPosD2"]
	vsPatchNegD1Ly := ly.RecvFrom["VSPatchNegD1"]
	vsPatchNegD2Ly := ly.RecvFrom["VSPatchNegD2"]

	g := ly.DAGains
	nrn := &ly.Neurons[0]

	pptgDAp := TotalAct(pptGLy)
	lhbDA := TotalAct(lhbLy)

	posPVAct := TotalAct(posPVLy)
	var vsPosPVI float32 = 0
	if g.PVIAntiBurstShunt > 0 {
		vsPosPVI = g.PVIBurstShunt*TotalAct(vsPatchPosD1Ly) -
			g.PVIAntiBurstShunt*TotalAct(vsPatchPosD2Ly)
	} else {
		vsPosPVI = g.PVIBurstShunt * TotalAct(vsPatchPosD1Ly)
	}

	// vspospvi must be >= 0.0f
	vsPosPVI = math32.Max(vsPosPVI, 0)

	var vsNegPVI float32 = 0

	if g.PVIDipShunt > 0 && g.PVIAntiDipShunt > 0 {
		vsNegPVI = g.PVIDipShunt*TotalAct(vsPatchNegD2Ly) -
			g.PVIAntiDipShunt*TotalAct(vsPatchNegD1Ly)
	} else if g.PVIDipShunt > 0 {
		vsNegPVI = g.PVIDipShunt * TotalAct(vsPatchNegD2Ly)
	}

	burstLHbDA := math32.Min(lhbDA, 0) // if neg, promotes bursting
	dipLHbDA := math32.Max(lhbDA, 0)   // else, promotes dipping

	// absorbing PosPV value - prevents double counting
	totBurstDA := math32.Max(g.PV*posPVAct, g.PPTg*pptgDAp)
	// likewise for lhb contribution to bursting (burst_lhb_da non-positive)
	totBurstDA = math32.Max(totBurstDA, -g.LHb*burstLHbDA)

	// pos PVi shunting
	netBurstDA := totBurstDA - vsPosPVI
	netBurstDA = math32.Max(netBurstDA, 0)

	totDipDA := g.LHb * dipLHbDA

	// neg PVi shunting
	netDipDA := totDipDA - vsNegPVI
	netDipDA = math32.Max(netDipDA, 0)

	if math32.IsNaN(netBurstDA) || math32.IsNaN(netDipDA) || math32.IsNaN(lhbDA) {
		fmt.Println("NaN in VTA")
	}

	netDA := netBurstDA - netDipDA
	netDA *= g.DA

	//netDA -= da.SEGain * ly.SE // subtract 5HT serotonin -- has its own gain

	ly.DA = netDA
	nrn.Ext = ly.TonicDA + ly.DA
	nrn.ActLrn = nrn.Ext
	nrn.Act = nrn.Ext
	nrn.Ge = nrn.Ext
	nrn.ActDel = 0
	ly.SendVal = nrn.Ext

	ly.InternalState.PPTgDAp = pptgDAp
	ly.InternalState.LHbDA = lhbDA
	ly.InternalState.PosPVAct = posPVAct
	ly.InternalState.VSPosPVI = vsPosPVI
	ly.InternalState.VSNegPVI = vsNegPVI
	ly.InternalState.BurstLHbDA = burstLHbDA
	ly.InternalState.DipLHbDA = dipLHbDA
	ly.InternalState.TotBurstDA = totBurstDA
	ly.InternalState.TotDipDA = totDipDA
	ly.InternalState.NetDipDA = netDipDA
	ly.InternalState.NetDA = netDA
	ly.InternalState.SendVal = ly.SendVal

}

// VTAn activation
func (ly *VTALayer) VTAActN(_ *leabra.Time) {
	negPVLy := ly.RecvFrom["NegPV"]
	lhbLy := ly.RecvFrom["LHbRMTg"]
	vsPatchNegD1Ly := ly.RecvFrom["VSPatchNegD1"]
	vsPatchNegD2Ly := ly.RecvFrom["VSPatchNegD2"]
	g := ly.DAGains
	nrn := &ly.Neurons[0]

	negPVAct := TotalAct(negPVLy)
	lhbDAn := TotalAct(lhbLy)

	var vsPVIn float32

	if g.PVIAntiBurstShunt > 0 {
		vsPVIn = g.PVIBurstShunt*TotalAct(vsPatchNegD2Ly) -
			g.PVIAntiBurstShunt*TotalAct(vsPatchNegD1Ly)
	} else {
		vsPVIn = g.PVIBurstShunt * TotalAct(vsPatchNegD2Ly)
	}

	burstLHbDAn := math32.Max(lhbDAn, 0)
	dipLHbDAn := math32.Min(lhbDAn, 0)

	// absorbing NegPV value - prevents double counting
	negPVDA := negPVAct
	negPVDA = math32.Max(negPVDA, 0)

	totBurstDA := math32.Max(g.PV*negPVDA, g.LHb*burstLHbDAn)

	// PVi shunting
	netBurstDA := totBurstDA - vsPVIn
	netBurstDA = math32.Max(netBurstDA, 0)

	totDipDA := g.LHb * dipLHbDAn

	netDA := netBurstDA + totDipDA
	netDA *= g.DA

	ly.DA = netDA
	nrn.Ext = ly.TonicDA + ly.DA
	nrn.ActLrn = nrn.Ext
	nrn.Act = nrn.Ext
	nrn.Ge = nrn.Ext
	nrn.ActDel = 0
	ly.SendVal = nrn.Ge
}
