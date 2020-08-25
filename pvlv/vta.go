package pvlv

import (
	"fmt"
	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/rl"
	"strconv"
)

type VTADASpec struct {
	TonicDA  float32 `desc:"set a tonic 'dopamine' (DA) level (offset to add to da values)"`
	PatchCur bool    `desc:"use current trial patch activations -- otherwise use previous trial -- current trial is appropriate for simple discrete trial environments (e.g., with some PBWM models), whereas previous is more approprate for trials with more realistic temporal structure"`
	SEGain   float32 `desc:"strength of the serotonin 5HT inputs on the dopamine outputs -- sev values when present subtract from the dopamine value otherwise computed"`
	RecData  bool    `desc:"record all the internal computations in user data on the VTA layer"`
}

type VTADAGains struct {
	DAGain                float32 `desc:"overall multiplier for dopamine values"`
	PPTgGain              float32 `desc:"gain on bursts from PPTg"`
	LHbGain               float32 `desc:"gain on dips/bursts from LHbRMTg"`
	PVGain                float32 `desc:"gain on positive PV component of total phasic DA signal (net after subtracting VSPatchIndir (PVi) shunt signal)"`
	PVIBurstShuntGain     float32 `desc:"gain on VSPatch projection that shunts bursting in VTA (for VTAp = VSPatchPosD1, for VTAn = VSPatchNegD2)"`
	PVIAntiBurstShuntGain float32 `desc:"gain on VSPatch projection that opposes shunting of bursting in VTA (for VTAp = VSPatchPosD2, for VTAn = VSPatchNegD1)"`
	PVIDipShuntGain       float32 `desc:"gain on VSPatch projection that shunts dipping of VTA (currently only VTAp supported = VSPatchNegD2) -- optional and somewhat controversial"`
	PVIAntiDipShuntGain   float32 `desc:"gain on VSPatch projection that opposes the shunting of dipping in VTA (currently only VTAp supported = VSPatchNegD1)"`
}

type LVBlockSpec struct {
	PosPV  float32 `desc:"down-regulate LV by factor of: (1 - pos_pv * pv) for positive pv signals (e.g., from LHA etc) -- the larger this value, the more LV is blocked -- if it is 0, then there is no LV block at all -- net actual block is 1 - sum over both sources of block"`
	LHbDip float32 `desc:"down-regulate LV by factor of: (1 - dip * lhb_rmtg) for da dip signals coming from the LHbRMTg sytem -- the larger this value, the more LV is blocked -- if it is 0, then there is no LV block at all -- net actual block is 1 - sum over both sources of block"`
}

type VTALayer struct {
	rl.ClampDaLayer
	SendVal       float32
	Valence       Valence     `desc:"VTA layer DA valence, positive or negative"`
	DASpec        VTADASpec   `desc:"specs for PVLV da parameters"`
	DAGains       VTADAGains  `view:"inline" desc:"gains for various VTA inputs"`
	LVBlock       LVBlockSpec `desc:"how LV signals are blocked by PV and LHbRMTg dip signals -- there are good reasons for these signals to block LV, because they reflect a stronger overall signal about outcomes, compared to the more 'speculative' LV signal"`
	RecvFrom      map[string]emer.Layer
	InternalState VTAState
}

// monitoring and debugging only
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
	ly.Defaults()
	return err
}

func (ly *VTALayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.VmRange.Min = -2.0
	ly.Act.VmRange.Max = 2.0
	ly.Act.Clamp.Range.Min = -2.0
	ly.Act.Clamp.Range.Max = 2.0

	ly.DASpec.TonicDA = 0
	ly.DASpec.PatchCur = true
	ly.DASpec.SEGain = 0.1
	ly.DASpec.RecData = true

	ly.DAGains.DAGain = 1.0
	ly.DAGains.PPTgGain = 1.0
	ly.DAGains.LHbGain = 1.0
	ly.DAGains.PVGain = 1.0
	ly.DAGains.PVIBurstShuntGain = 1.05
	ly.DAGains.PVIAntiBurstShuntGain = 1.0
	ly.DAGains.PVIDipShuntGain = 0.0
	ly.DAGains.PVIAntiDipShuntGain = 0.0

	ly.LVBlock.PosPV = 1.0
	ly.LVBlock.LHbDip = 2.0
}

func (ly *VTALayer) GetMonitorVal(data []string) float64 {
	var val float32
	valType := data[0]
	unitIdx, _ := strconv.Atoi(data[1])
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
		val = GlobalTotalActFn(ly)
	case "PoolActAvg":
		val = ly.Pools[unitIdx].Inhib.Act.Avg
	case "PoolActMax":
		val = ly.Pools[unitIdx].Inhib.Act.Max
	case "Act":
		val = ly.Neurons[unitIdx].Act
	case "DA":
		val = ly.DA
	default:
		val = ly.Neurons[0].Act
	}
	return float64(val)
}

func TotalAct(ly emer.Layer) float32 {
	lly := ly.(leabra.LeabraLayer).AsLeabra()
	acts := make([]float32, len(lly.Neurons))
	err := lly.UnitVals(&acts, "Act")
	if err != nil {
		fmt.Println(err)
	}
	sum := float32(0)
	for _, n := range acts {
		sum += n
	}
	//fmt.Printf("TotalAct for %v=%v\n", ly.Name(), sum)
	return sum
}

func TotalActGp(ly emer.Layer) float32 {
	lly := ly.(leabra.LeabraLayer).AsLeabra()
	sum := float32(0)
	nUnits := 0
	var pStart int
	if len(lly.Pools) != len(lly.Neurons) {
		pStart = 1
	} else {
		pStart = 0
	}
	for pi := pStart; pi < len(lly.Pools); pi++ {
		pl := lly.Pools[pi].Inhib.Act
		sum += pl.Avg
		nUnits += pl.N
	}
	if sum == math32.NaN() {
		fmt.Println("NaN in TotalActGp")
	}
	res := sum * float32(nUnits)
	//fmt.Printf("TotalActGp for %v=%v\n", ly.Name(), res)
	return res
}

func TotalActGp0(ly emer.Layer) float32 {
	lly := ly.(leabra.LeabraLayer).AsLeabra()
	pl := lly.Pools[0].Inhib.Act
	res := pl.Avg * float32(pl.N)
	if math32.IsNaN(res) {
		fmt.Println("NaN in TotalActGp0")
	}
	return res
}

func TotalActQ0(ly emer.Layer) float32 {
	lly := ly.(leabra.LeabraLayer).AsLeabra()
	acts := make([]float32, len(lly.Neurons))
	err := ly.UnitVals(&acts, "ActQ0")
	if err != nil {
		fmt.Println(err)
	}
	sum := float32(0)
	for _, n := range acts {
		sum += n
	}
	//fmt.Printf("TotalActQ0 for %v=%v\n", ly.Name(), sum)
	return sum
}

/*
func TotalModAct(ly emer.Layer) float32 {
	var mly *ModLayer
	switch ly.(type) {
	case *ModLayer:
		mly = ly.(*ModLayer)
	case *leabra.Layer:
		return TotalAct(ly)
	default:
		mly = ly.(IModLayer).AsMod()
	}
	acts := make([]float32, len(mly.ModNeurs))
	err := mly.UnitVals(&acts, "ModAct")
	if err != nil {
		fmt.Println(err)
	}
	sum := float32(0)
	for _, n := range acts {
		sum += n
	}
	//fmt.Printf("TotalModAct for %v=%v\n", ly.Name(), sum)
	return sum
}

func TotalModActGp(ly emer.Layer) float32 {
	//sum := ly.Pools[0].ActAvg.ActPAvg
	var mly *ModLayer
	imly, ok := ly.(IModLayer)
	if !ok {
		fmt.Printf("%v is not a ModLayer (%v)\n", ly.Name(), mly)
		return TotalActGp(ly.(leabra.LeabraLayer).AsLeabra())
	} else {
		mly = imly.(IModLayer).AsMod()
	}
	sum := float32(0)
	nUnits := 0
	var pStart int
	if len(mly.ModPools) != len(mly.Neurons) {
		pStart = 1
	} else {
		pStart = 0
	}
	for pi := pStart; pi < len(mly.ModPools); pi++ {
		mpl := &mly.ModPools[pi].ModNetStats
		sum += mpl.Avg
		nUnits += mpl.N
	}
	if sum == math32.NaN() {
		fmt.Println("NaN in TotalModActGp")
	}
	res := sum * float32(nUnits)
	//fmt.Printf("TotalModActGp for %v=%v\n", ly.Name(), res)
	return res
}


func TotalActQ0Gp(ly emer.Layer) float32 {
	lly := ly.(leabra.LeabraLayer).AsLeabra()
	return TotalActQ0(lly)
}


func (ly *VTALayer) ActFmG(ltime *leabra.Time) {
	if ltime.Quarter == int(leabra.Q4) {
		ly.VTAAct()
	} else {
		nrn := &ly.Neurons[0]
		nrn.ActLrn = 0
		nrn.Act = 0
		nrn.Ge = 0
		ly.SendVal = 0
	}
	ly.DA = 0
}
*/

//// GLOBAL VARIABLE FOR DETERMINING WHICH VERSION OF "TotalActEq" WE USE. ALSO USED IN LHbRMTg
var GlobalTotalActFn = TotalActGp0

func (ly *VTALayer) CyclePost(ltime *leabra.Time) {
	ly.SendDA.SendDA(ly.Network, ly.SendVal)
}

func (ly *VTALayer) VTAAct() {
	if ly.Valence == POS {
		ly.VTAActP()
	} else {
		ly.VTAActN()
	}
}

func (ly *VTALayer) VTAActP() {
	totalFn := GlobalTotalActFn
	pptGLy := ly.RecvFrom["PPTg"]
	lhbLy := ly.RecvFrom["LHbRMTg"]
	posPVLy := ly.RecvFrom["PosPV"]
	vsPatchPosD1Ly := ly.RecvFrom["VSPatchPosD1"]
	vsPatchPosD2Ly := ly.RecvFrom["VSPatchPosD2"]
	vsPatchNegD1Ly := ly.RecvFrom["VSPatchNegD1"]
	vsPatchNegD2Ly := ly.RecvFrom["VSPatchNegD2"]

	g := ly.DAGains
	da := ly.DASpec
	nrn := &ly.Neurons[0]

	// actfn covers both cases, since they're identical except for which activation function is used
	// using Q0 has not been tested and may crash
	var actfn func(ly emer.Layer) float32
	if da.PatchCur {
		actfn = totalFn
	} else {
		fmt.Println("da.PatchCur FALSE!")
		actfn = TotalActQ0
	}

	pptgDAp := totalFn(pptGLy)
	lhbDA := totalFn(lhbLy)

	posPVAct := totalFn(posPVLy)
	var vsPosPVI float32 = 0
	if g.PVIAntiBurstShuntGain > 0 {
		vsPosPVI = g.PVIBurstShuntGain*totalFn(vsPatchPosD1Ly) -
			g.PVIAntiBurstShuntGain*actfn(vsPatchPosD2Ly)
	} else {
		vsPosPVI = g.PVIBurstShuntGain * actfn(vsPatchPosD1Ly)
	}

	// vspospvi must be >= 0.0f
	vsPosPVI = math32.Max(vsPosPVI, 0)

	var vsNegPVI float32 = 0

	if g.PVIDipShuntGain > 0 && g.PVIAntiDipShuntGain > 0 {
		vsNegPVI = g.PVIDipShuntGain*actfn(vsPatchNegD2Ly) -
			g.PVIAntiDipShuntGain*actfn(vsPatchNegD1Ly)
	} else if g.PVIDipShuntGain > 0 {
		vsNegPVI = g.PVIDipShuntGain * actfn(vsPatchNegD2Ly)
	}

	burstLHbDA := math32.Min(lhbDA, 0) // if neg, promotes bursting
	dipLHbDA := math32.Max(lhbDA, 0)   // else, promotes dipping

	// absorbing PosPV value - prevents double counting
	totBurstDA := math32.Max(g.PVGain*posPVAct, g.PPTgGain*pptgDAp)
	// likewise for lhb contribution to bursting (burst_lhb_da non-positive)
	totBurstDA = math32.Max(totBurstDA, -g.LHbGain*burstLHbDA)

	// pos PVi shunting
	netBurstDA := totBurstDA - vsPosPVI
	netBurstDA = math32.Max(netBurstDA, 0)
	//if(net_burst_da < 0.1f) { net_burst_da = 0.0f; } // debug...

	totDipDA := g.LHbGain * dipLHbDA

	// neg PVi shunting
	netDipDA := totDipDA - vsNegPVI
	netDipDA = math32.Max(netDipDA, 0)
	//if(net_dip_da < 0.1f) { net_dip_da = 0.0f; } // debug...

	if math32.IsNaN(netBurstDA) || math32.IsNaN(netDipDA) || math32.IsNaN(lhbDA) {
		fmt.Println("NaN in VTA")
	}

	netDA := netBurstDA - netDipDA
	netDA *= g.DAGain

	//netDA -= da.SEGain * ly.SE // subtract 5HT serotonin -- has its own gain

	ly.DA = netDA
	nrn.Ext = da.TonicDA + ly.DA
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

	//net->ext_rew_avail = true;    // always record pv values -- todo: why??
	//net->ext_rew = pospv;

}

func (ly *VTALayer) VTAActN() {
	totalFn := GlobalTotalActFn
	negPVLy := ly.RecvFrom["NegPV"]
	lhbLy := ly.RecvFrom["LHbRMTg"]
	vsPatchNegD1Ly := ly.RecvFrom["VSPatchNegD1"]
	vsPatchNegD2Ly := ly.RecvFrom["VSPatchNegD2"]
	g := ly.DAGains
	daSpec := ly.DASpec
	nrn := &ly.Neurons[0]

	var actfn func(ly emer.Layer) float32
	if daSpec.PatchCur {
		actfn = totalFn
	} else {
		actfn = TotalActQ0
	}

	negPVAct := totalFn(negPVLy)
	lhbDAn := totalFn(lhbLy)

	var vsPVIn float32

	if g.PVIAntiBurstShuntGain > 0 {
		vsPVIn = g.PVIBurstShuntGain*actfn(vsPatchNegD2Ly) -
			g.PVIAntiBurstShuntGain*actfn(vsPatchNegD1Ly)
	} else {
		vsPVIn = g.PVIBurstShuntGain * actfn(vsPatchNegD2Ly)
	}

	burstLHbDAn := math32.Max(lhbDAn, 0)
	dipLHbDAn := math32.Min(lhbDAn, 0)

	// absorbing NegPV value - prevents double counting
	negPVDA := negPVAct
	negPVDA = math32.Max(negPVDA, 0)

	totBurstDA := math32.Max(g.PVGain*negPVDA, g.LHbGain*burstLHbDAn)

	// PVi shunting
	netBurstDA := totBurstDA - vsPVIn
	netBurstDA = math32.Max(netBurstDA, 0)

	totDipDA := g.LHbGain * dipLHbDAn

	netDA := netBurstDA + totDipDA
	netDA *= g.DAGain

	ly.DA = netDA
	nrn.Ext = daSpec.TonicDA + ly.DA
	nrn.ActLrn = nrn.Ext
	nrn.Act = nrn.Ext
	nrn.Ge = nrn.Ext
	nrn.ActDel = 0
	ly.SendVal = nrn.Ge
}
