// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"log"

	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// deep.Layer is the DeepLeabra layer, based on basic rate-coded leabra.Layer
type Layer struct {
	leabra.Layer             // access as .Layer
	DeepBurst    BurstParams `view:"inline" desc:"parameters for computing Burst from act, in Superficial layers (but also needed in Deep layers for deep self connections)"`
	DeepTRC      TRCParams   `view:"inline" desc:"parameters for computing TRC plus-phase (outcome) activations based on TRCBurstGe excitatory input from BurstTRC projections"`
	DeepAttn     AttnParams  `view:"inline" desc:"parameters for computing DeepAttn and DeepLrn attentional modulation signals based on DeepAttn projection inputs integrated into AttnGe excitatory conductances"`
	DeepNeurs    []Neuron    `desc:"slice of extra deep.Neuron state for this layer -- flat list of len = Shape.Len(). You must iterate over index and use pointer to modify values."`
	DeepPools    []Pool      `desc:"extra layer and sub-pool (unit group) statistics used in DeepLeabra -- flat list has at least of 1 for layer, and one for each sub-pool (unit group) if shape supports that (4D).  You must iterate over index and use pointer to modify values."`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

// AsDeep returns this layer as a deep.Layer
func (ly *Layer) AsDeep() *Layer {
	return ly
}

func (ly *Layer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Init.Decay = 0 // deep doesn't decay!
	ly.DeepBurst.Defaults()
	ly.DeepTRC.Defaults()
	ly.DeepAttn.Defaults()
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *Layer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.DeepBurst.Update()
	ly.DeepTRC.Update()
	ly.DeepAttn.Update()
}

func (ly *Layer) Class() string {
	switch ly.Typ {
	case Deep:
		return "Deep " + ly.Cls
	case TRC:
		return "TRC " + ly.Cls
	}
	return ly.Typ.String() + " " + ly.Cls
}

// IsSuper returns true if layer is not a TRC or Deep type -- all others are Super
func (ly *Layer) IsSuper() bool {
	if ly.Typ == TRC || ly.Typ == Deep {
		return false
	}
	return true // everything else is super
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *Layer) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitVals fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *Layer) UnitVals(vals *[]float32, varNm string) error {
	vidx, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitVals(vals, varNm)
	}
	vidx, err = NeuronVarByName(varNm)
	if err != nil {
		return err
	}
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	for i := range ly.DeepNeurs {
		dnr := &ly.DeepNeurs[i]
		(*vals)[i] = dnr.VarByIndex(vidx)
	}
	return nil
}

// UnitValsTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *Layer) UnitValsTensor(tsr etensor.Tensor, varNm string) error {
	if tsr == nil {
		err := fmt.Errorf("leabra.UnitValsTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	vidx, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitValsTensor(tsr, varNm)
	}
	vidx, err = NeuronVarByName(varNm)
	if err != nil {
		return err
	}
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	for i := range ly.DeepNeurs {
		dnr := &ly.DeepNeurs[i]
		tsr.SetFloat1D(i, float64(dnr.VarByIndex(vidx)))
	}
	return nil
}

// UnitValTry returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *Layer) UnitValTry(varNm string, idx []int) (float32, error) {
	_, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitValTry(varNm, idx)
	}
	fidx := ly.Shp.Offset(idx)
	nn := len(ly.DeepNeurs)
	if fidx < 0 || fidx >= nn {
		return 0, fmt.Errorf("Layer UnitVal index: %v out of range, N = %v", fidx, nn)
	}
	dnr := &ly.DeepNeurs[fidx]
	return dnr.VarByName(varNm)
}

// UnitVal1DTry returns value of given variable name on given unit,
// using 1-dimensional index.
func (ly *Layer) UnitVal1DTry(varNm string, idx int) (float32, error) {
	_, err := leabra.NeuronVarByName(varNm)
	if err == nil {
		return ly.Layer.UnitVal1DTry(varNm, idx)
	}
	nn := len(ly.DeepNeurs)
	if idx < 0 || idx >= nn {
		return 0, fmt.Errorf("Layer UnitVal1D index: %v out of range, N = %v", idx, nn)
	}
	dnr := &ly.DeepNeurs[idx]
	return dnr.VarByName(varNm)
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *Layer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.DeepNeurs = make([]Neuron, len(ly.Neurons))
	ly.DeepPools = make([]Pool, len(ly.Pools))
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

func (ly *Layer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.DeepNeurs {
		dnr := &ly.DeepNeurs[ni]
		dnr.ActNoAttn = 0
		dnr.Burst = 0
		dnr.BurstPrv = 0
		dnr.CtxtGe = 0
		dnr.TRCBurstGe = 0
		dnr.BurstSent = 0
		dnr.AttnGe = 0
		dnr.DeepAttn = 0
		dnr.DeepLrn = 0
	}
}

// GScaleFmAvgAct computes the scaling factor for synaptic input conductances G,
// based on sending layer average activation.
// This attempts to automatically adjust for overall differences in raw activity coming into the units
// to achieve a general target of around .5 to 1 for the integrated G values.
// DeepLeabra version separately normalizes the Deep projection types.
func (ly *Layer) GScaleFmAvgAct() {
	totGeRel := float32(0)
	totGiRel := float32(0)
	totTrcRel := float32(0)
	totAttnRel := float32(0)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(leabra.LeabraPrjn).AsLeabra()
		slay := p.SendLay().(leabra.LeabraLayer).AsLeabra()
		slpl := &slay.Pools[0]
		savg := slpl.ActAvg.ActPAvgEff
		snu := len(slay.Neurons)
		ncon := pj.RConNAvgMax.Avg
		pj.GScale = pj.WtScale.FullScale(savg, float32(snu), ncon)
		switch pj.Typ {
		case emer.Inhib:
			totGiRel += pj.WtScale.Rel
		case BurstTRC:
			totTrcRel += pj.WtScale.Rel
		case DeepAttn:
			totAttnRel += pj.WtScale.Rel
		default:
			// note: BurstCtxt is added in here!
			totGeRel += pj.WtScale.Rel
		}
	}

	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(leabra.LeabraPrjn).AsLeabra()
		switch pj.Typ {
		case emer.Inhib:
			if totGiRel > 0 {
				pj.GScale /= totGiRel
			}
		case BurstTRC:
			if totTrcRel > 0 {
				pj.GScale /= totTrcRel
			}
		case DeepAttn:
			if totAttnRel > 0 {
				pj.GScale /= totAttnRel
			}
		default:
			if totGeRel > 0 {
				pj.GScale /= totGeRel
			}
		}
	}
}

func (ly *Layer) DecayState(decay float32) {
	ly.Layer.DecayState(decay)
	for ni := range ly.DeepNeurs {
		dnr := &ly.DeepNeurs[ni]
		dnr.ActNoAttn -= decay * (dnr.ActNoAttn - ly.Act.Init.Act)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// InitGinc initializes the Ge excitatory and Gi inhibitory conductance accumulation states
// including ActSent and G*Raw values.
// called at start of trial always, and can be called optionally
// when delta-based Ge computation needs to be updated (e.g., weights
// might have changed strength)
func (ly *Layer) InitGInc() {
	ly.Layer.InitGInc()
	for ni := range ly.DeepNeurs {
		dnr := &ly.DeepNeurs[ni]
		dnr.BurstSent = 0
		dnr.TRCBurstGe = 0
		dnr.AttnGe = 0
	}
}

// SendGDelta sends change in activation since last sent, if above thresholds.
// Deep version sends either to standard Ge or AttnGe for DeepAttn projections.
func (ly *Layer) SendGDelta(ltime *leabra.Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.Act > ly.Act.OptThresh.Send {
			delta := nrn.Act - nrn.ActSent
			if math32.Abs(delta) > ly.Act.OptThresh.Delta {
				for _, sp := range ly.SndPrjns {
					if sp.IsOff() {
						continue
					}
					ptyp := sp.Type()
					if ptyp == BurstCtxt || ptyp == BurstTRC {
						continue
					}
					if ptyp == DeepAttn {
						if ly.DeepAttn.On {
							sp.(DeepPrjn).SendAttnGeDelta(ni, delta)
						}
					} else {
						sp.(leabra.LeabraPrjn).SendGDelta(ni, delta)
					}
				}
				nrn.ActSent = nrn.Act
			}
		} else if nrn.ActSent > ly.Act.OptThresh.Send {
			delta := -nrn.ActSent // un-send the last above-threshold activation to get back to 0
			for _, sp := range ly.SndPrjns {
				if sp.IsOff() {
					continue
				}
				ptyp := sp.Type()
				if ptyp == BurstCtxt || ptyp == BurstTRC {
					continue
				}
				if ptyp == DeepAttn {
					if ly.DeepAttn.On {
						sp.(DeepPrjn).SendAttnGeDelta(ni, delta)
					}
				} else {
					sp.(leabra.LeabraPrjn).SendGDelta(ni, delta)
				}
			}
			nrn.ActSent = 0
		}
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *Layer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	if ly.Typ == TRC && ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			// note: TRCBurstGe is sent at *end* of previous cycle, after Burst act is computed
			var pl *Pool
			if ly.DeepTRC.InhibPool {
				pl = &ly.DeepPools[nrn.SubPool]
			} else {
				pl = &ly.DeepPools[0]
			}
			dnr := &ly.DeepNeurs[ni]
			ly.Act.GRawFmInc(nrn)
			burstInhib := math32.Min(1, pl.TRCBurstGe.Max/ly.DeepTRC.MaxInhib)
			geRaw := (1-burstInhib)*nrn.GeRaw + ly.DeepTRC.BurstGe(dnr.TRCBurstGe)
			ly.Act.GeFmRaw(nrn, geRaw)
			ly.Act.GiFmRaw(nrn, nrn.GiRaw)
		}
		return
	}
	if ly.Typ == Deep {
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			dnr := &ly.DeepNeurs[ni]
			ly.Act.GRawFmInc(nrn)
			geRaw := nrn.GeRaw + dnr.CtxtGe
			ly.Act.GeFmRaw(nrn, geRaw)
			ly.Act.GiFmRaw(nrn, nrn.GiRaw)
		}
		return
	}
	ly.GFmIncNeur(ltime) // regular
	ly.LeabraLay.(DeepLayer).AttnGeInc(ltime)
}

// AttnGeInc integrates new AttnGe from increments sent during last SendGDelta.
// Very low overhead if no DeepAttn prjns.
func (ly *Layer) AttnGeInc(ltime *leabra.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj, ok := p.(DeepPrjn)
		if !ok {
			continue
		}
		ptyp := pj.Type()
		if ptyp != DeepAttn {
			continue
		}
		pj.RecvAttnGeInc()
	}
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
// Deep version also computes AttnGe stats
func (ly *Layer) AvgMaxGe(ltime *leabra.Time) {
	ly.Layer.AvgMaxGe(ltime)
	ly.LeabraLay.(DeepLayer).AvgMaxAttnGe(ltime)
}

// AvgMaxAttnGe computes the average and max AttnGe stats
func (ly *Layer) AvgMaxAttnGe(ltime *leabra.Time) {
	for pi := range ly.DeepPools {
		pl := &ly.Pools[pi]
		dpl := &ly.DeepPools[pi]
		dpl.AttnGe.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			dnr := &ly.DeepNeurs[ni]
			dpl.AttnGe.UpdateVal(dnr.AttnGe, ni)
		}
		dpl.AttnGe.CalcAvg()
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *Layer) ActFmG(ltime *leabra.Time) {
	ly.Layer.ActFmG(ltime)
	ly.LeabraLay.(DeepLayer).DeepAttnFmG(ltime)
}

// DeepAttnFmG computes DeepAttn and DeepLrn from AttnGe input,
// and then applies the DeepAttn modulation to the Act activation value.
func (ly *Layer) DeepAttnFmG(ltime *leabra.Time) {
	lpl := &ly.DeepPools[0]
	attnMax := lpl.AttnGe.Max
	for ni := range ly.DeepNeurs {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		dnr := &ly.DeepNeurs[ni]
		switch {
		case !ly.DeepAttn.On:
			dnr.DeepAttn = 1
			dnr.DeepLrn = 1
		case ly.Typ == Deep:
			dnr.DeepAttn = nrn.Act // record Deep activation = DeepAttn signal coming from deep layers
			dnr.DeepLrn = 1
		case ly.Typ == TRC:
			dnr.DeepAttn = 1
			dnr.DeepLrn = 1
		default:
			if attnMax < ly.DeepAttn.Thr {
				dnr.DeepAttn = 1
				dnr.DeepLrn = 1
			} else {
				dnr.DeepLrn = dnr.AttnGe / attnMax
				dnr.DeepAttn = ly.DeepAttn.DeepAttnFmG(dnr.DeepLrn)
			}
		}
		dnr.ActNoAttn = nrn.Act
		nrn.Act *= dnr.DeepAttn
	}
}

// AvgMaxAct computes the average and max Act stats, used in inhibition
// Deep version also computes AvgMaxActNoAttn
func (ly *Layer) AvgMaxAct(ltime *leabra.Time) {
	ly.Layer.AvgMaxAct(ltime)
	ly.LeabraLay.(DeepLayer).AvgMaxActNoAttn(ltime)
}

// AvgMaxActNoAttn computes the average and max ActNoAttn stats
func (ly *Layer) AvgMaxActNoAttn(ltime *leabra.Time) {
	for pi := range ly.DeepPools {
		pl := &ly.Pools[pi]
		dpl := &ly.DeepPools[pi]
		dpl.ActNoAttn.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			dnr := &ly.DeepNeurs[ni]
			dpl.ActNoAttn.UpdateVal(dnr.ActNoAttn, ni)
		}
		dpl.ActNoAttn.CalcAvg()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  DeepBurst -- computed every cycle at end of standard Cycle in DeepBurst quarter

// BurstFmAct updates Burst layer 5 IB bursting value from current Act (superficial activation)
// Subject to thresholding.
func (ly *Layer) BurstFmAct(ltime *leabra.Time) {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		return
	}
	lpl := &ly.DeepPools[0]
	actMax := lpl.ActNoAttn.Max
	actAvg := lpl.ActNoAttn.Avg
	thr := actAvg + ly.DeepBurst.ThrRel*(actMax-actAvg)
	thr = math32.Max(thr, ly.DeepBurst.ThrAbs)
	for ni := range ly.DeepNeurs {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		dnr := &ly.DeepNeurs[ni]
		burst := float32(0)
		if dnr.ActNoAttn > thr {
			burst = dnr.ActNoAttn
		}
		dnr.Burst = burst
	}
}

// SendTRCBurstGeDelta sends change in Burst activation since last sent, over BurstTRC
// projections.
func (ly *Layer) SendTRCBurstGeDelta(ltime *leabra.Time) {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		return
	}
	for ni := range ly.DeepNeurs {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		dnr := &ly.DeepNeurs[ni]
		if dnr.Burst > ly.Act.OptThresh.Send {
			delta := dnr.Burst - dnr.BurstSent
			if math32.Abs(delta) > ly.Act.OptThresh.Delta {
				for _, sp := range ly.SndPrjns {
					if sp.IsOff() {
						continue
					}
					ptyp := sp.Type()
					if ptyp != BurstTRC {
						continue
					}
					pj, ok := sp.(DeepPrjn)
					if !ok {
						continue
					}
					pj.SendTRCBurstGeDelta(ni, delta)
				}
				dnr.BurstSent = dnr.Burst
			}
		} else if dnr.BurstSent > ly.Act.OptThresh.Send {
			delta := -dnr.BurstSent // un-send the last above-threshold activation to get back to 0
			for _, sp := range ly.SndPrjns {
				if sp.IsOff() {
					continue
				}
				ptyp := sp.Type()
				if ptyp != BurstTRC {
					continue
				}
				pj, ok := sp.(DeepPrjn)
				if !ok {
					continue
				}
				pj.SendTRCBurstGeDelta(ni, delta)
			}
			dnr.BurstSent = 0
		}
	}
}

// TRCBurstGeFmInc computes the TRCBurstGe input from sent values
func (ly *Layer) TRCBurstGeFmInc(ltime *leabra.Time) {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		return
	}
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		ptyp := p.Type()
		if ptyp != BurstTRC {
			continue
		}
		pj, ok := p.(DeepPrjn)
		if !ok {
			continue
		}
		pj.RecvTRCBurstGeInc()
	}
	// note: full integration of Inc happens next cycle..
}

// AvgMaxTRCBurstGe computes the average and max TRCBurstGe stats
func (ly *Layer) AvgMaxTRCBurstGe(ltime *leabra.Time) {
	for pi := range ly.DeepPools {
		pl := &ly.Pools[pi]
		dpl := &ly.DeepPools[pi]
		dpl.TRCBurstGe.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			dnr := &ly.DeepNeurs[ni]
			dpl.TRCBurstGe.UpdateVal(dnr.TRCBurstGe, ni)
		}
		dpl.TRCBurstGe.CalcAvg()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  DeepCtxt -- once after DeepBurst quarter

// SendCtxtGe sends full Burst activation over BurstCtxt projections to integrate
// CtxtGe excitatory conductance on deep layers.
// This must be called at the end of the DeepBurst quarter for this layer.
func (ly *Layer) SendCtxtGe(ltime *leabra.Time) {
	if !ly.DeepBurst.On || !ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		return
	}
	for ni := range ly.DeepNeurs {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		dnr := &ly.DeepNeurs[ni]
		if dnr.Burst > ly.Act.OptThresh.Send {
			for _, sp := range ly.SndPrjns {
				if sp.IsOff() {
					continue
				}
				ptyp := sp.Type()
				if ptyp != BurstCtxt {
					continue
				}
				pj, ok := sp.(DeepPrjn)
				if !ok {
					continue
				}
				pj.SendCtxtGe(ni, dnr.Burst)
			}
		}
	}
}

// CtxtFmGe integrates new CtxtGe excitatory conductance from projections, and computes
// overall Ctxt value, only on Deep layers.
// This must be called at the end of the DeepBurst quarter for this layer, after SendCtxtGe.
func (ly *Layer) CtxtFmGe(ltime *leabra.Time) {
	if ly.Typ != Deep || !ly.DeepBurst.IsBurstQtr(ltime.Quarter) {
		return
	}
	for ni := range ly.DeepNeurs {
		dnr := &ly.DeepNeurs[ni]
		dnr.CtxtGe = 0
	}
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		ptyp := p.Type()
		if ptyp != BurstCtxt {
			continue
		}
		pj, ok := p.(DeepPrjn)
		if !ok {
			continue
		}
		pj.RecvCtxtGeInc()
	}
}

// QuarterFinal does updating after end of a quarter
func (ly *Layer) QuarterFinal(ltime *leabra.Time) {
	ly.Layer.QuarterFinal(ltime)
	ly.LeabraLay.(DeepLayer).BurstPrv(ltime)
}

// BurstPrv saves Burst as BurstPrv
func (ly *Layer) BurstPrv(ltime *leabra.Time) {
	if !ly.DeepBurst.On || !ly.DeepBurst.NextIsBurstQtr(ltime.Quarter) {
		return
	}
	for ni := range ly.DeepNeurs {
		dnr := &ly.DeepNeurs[ni]
		dnr.BurstPrv = dnr.Burst
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  LayerType

// note: need to define a new type for these extensions for the GUI interface,
// but need to use the *old type* in the code, so we have this unfortunate
// redundancy here.

// LayerType has the DeepLeabra extensions to the emer.LayerType types, for gui
type LayerType emer.LayerType

//go:generate stringer -type=LayerType

var KiT_LayerType = kit.Enums.AddEnumExt(emer.KiT_LayerType, LayerTypeN, kit.NotBitFlag, nil)

const (
	// Deep are deep-layer neurons, reflecting activation of layer 6 regular spiking
	// CT corticothalamic neurons, which drive both attention in Super (via DeepAttn
	// projections) and  predictions in TRC (Pulvinar) via standard projections.
	Deep emer.LayerType = emer.LayerTypeN + iota

	// TRC are thalamic relay cell neurons, typically in the Pulvinar, which alternately reflect
	// predictions driven by Deep layer projections, and actual outcomes driven by BurstTRC
	// projections from corresponding Super layer neurons that provide strong driving inputs to
	// TRC neurons.
	TRC
)

// gui versions
const (
	Deep_ LayerType = LayerType(emer.LayerTypeN) + iota
	TRC_
	LayerTypeN
)

var LayerProps = ki.Props{
	"EnumType:Typ": KiT_LayerType,
	"ToolBar": ki.PropSlice{
		{"Defaults", ki.Props{
			"icon": "reset",
			"desc": "return all parameters to their intial default values",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's weight values according to prjn parameters, for all *sending* projections out of this layer",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"LesionNeurons", ki.Props{
			"icon": "close",
			"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
			"Args": ki.PropSlice{
				{"Proportion", ki.Props{
					"desc": "proportion (0 -- 1) of neurons to lesion",
				}},
			},
		}},
		{"UnLesionNeurons", ki.Props{
			"icon": "reset",
			"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
		}},
	},
}
