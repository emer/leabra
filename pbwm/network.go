// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// pbwm.Network has parameters for running a DeepLeabra network
type Network struct {
	deep.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = deep.NetworkProps

// NewLayer returns new layer of default pbwm.Layer type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPrjn returns new prjn of default type
func (nt *Network) NewPrjn() emer.Prjn {
	return &deep.Prjn{}
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.Network.Defaults()
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	nt.Network.UpdateParams()
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMatrixLayer(name string, nY, nMaint, nOut, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	tX := nMaint + nOut
	mtx := &MatrixLayer{}
	nt.AddLayerInit(mtx, name, []int{nY, tX, nNeurY, nNeurX}, emer.Hidden)
	mtx.DaR = da
	mtx.GateShp.Set(nY, nMaint, nOut)
	return mtx
}

// AddGPeLayer adds a ModLayer to serve as a GPe layer, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func (nt *Network) AddGPeLayer(name string, nY, nMaint, nOut int) *ModLayer {
	tX := nMaint + nOut
	gpe := &ModLayer{}
	nt.AddLayerInit(gpe, name, []int{nY, tX, 1, 1}, emer.Hidden)
	return gpe
}

// AddGPiThalLayer adds a GPiThalLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func (nt *Network) AddGPiThalLayer(name string, nY, nMaint, nOut int) *GPiThalLayer {
	tX := nMaint + nOut
	gpi := &GPiThalLayer{}
	nt.AddLayerInit(gpi, name, []int{nY, tX, 1, 1}, emer.Hidden)
	gpi.GateShp.Set(nY, nMaint, nOut)
	return gpi
}

// AddDorsalBG adds MatrixGo, NoGo, GPe, and GPiThal layers, with given optional prefix.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  Appropriate PoolOneToOne connections
// are made to drive GPiThal, with BgFixed class name set so
// they can be styled appropriately (no learning, WtRnd.Mean=0.8, Var=0)
func (nt *Network) AddDorsalBG(prefix string, nY, nMaint, nOut, nNeurY, nNeurX int) (mtxGo, mtxNoGo, gpe, gpi leabra.LeabraLayer) {
	mtxGo = nt.AddMatrixLayer(prefix+"MatrixGo", nY, nMaint, nOut, nNeurY, nNeurX, D1R)
	mtxNoGo = nt.AddMatrixLayer(prefix+"MatrixNoGo", nY, nMaint, nOut, nNeurY, nNeurX, D2R)
	gpe = nt.AddGPeLayer(prefix+"GPeNoGo", nY, nMaint, nOut)
	gpi = nt.AddGPiThalLayer(prefix+"GPiThal", nY, nMaint, nOut)

	mtxNoGo.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: mtxGo.Name(), XAlign: relpos.Left, Space: 2})
	gpe.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxNoGo.Name(), YAlign: relpos.Front, Space: 2})
	gpi.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxGo.Name(), YAlign: relpos.Front, Space: 2})

	pj := nt.ConnectLayersPrjn(mtxGo, gpi, prjn.NewPoolOneToOne(), emer.Forward, &GPiThalPrjn{})
	pj.SetClass("BgFixed")
	pj = nt.ConnectLayers(mtxNoGo, gpe, prjn.NewPoolOneToOne(), emer.Forward)
	pj.SetClass("BgFixed")
	pj = nt.ConnectLayersPrjn(gpe, gpi, prjn.NewPoolOneToOne(), emer.Forward, &GPiThalPrjn{})
	pj.SetClass("BgFixed")
	return
}

// AddPFCLayer adds a PFCLayer, super and deep, of given size, with given name.
// nY, nX = number of pools in Y, X dimensions, and each pool has nNeurY, nNeurX neurons.
// out is true for output-gating layer. Both have the class "PFC" set.
// deep receives one-to-one projections of class "PFCToDeep" from super, and sends "PFCFmDeep",
// and is positioned behind it.
func (nt *Network) AddPFCLayer(name string, nY, nX, nNeurY, nNeurX int, out bool) (sp, dp *PFCLayer) {
	sp = &PFCLayer{}
	nt.AddLayerInit(sp, name, []int{nY, nX, nNeurY, nNeurX}, emer.Hidden)
	dp = &PFCLayer{}
	nt.AddLayerInit(dp, name+"D", []int{nY, nX, nNeurY, nNeurX}, deep.Deep)
	sp.SetClass("PFC")
	dp.SetClass("PFC")
	sp.Gate.OutGate = out
	dp.Gate.OutGate = out
	dp.Dyns.MaintOnly()
	dp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	pj := nt.ConnectLayers(sp, dp, prjn.NewOneToOne(), deep.BurstCtxt)
	pj.SetClass("PFCToDeep")
	pj = nt.ConnectLayers(dp, sp, prjn.NewOneToOne(), deep.DeepAttn)
	pj.SetClass("PFCFmDeep")
	return
}

// AddPFC adds paired PFCmnt, PFCout and associated Deep layers,
// with given optional prefix.
// nY = number of pools in Y dimension, nMaint, nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  Appropriate PoolOneToOne connections
// are made within super / deep (see AddPFCLayer) and between PFCmntD -> PFCout.
func (nt *Network) AddPFC(prefix string, nY, nMaint, nOut, nNeurY, nNeurX int) (pfcMnt, pfcMntD, pfcOut, pfcOutD leabra.LeabraLayer) {
	if prefix == "" {
		prefix = "PFC"
	}
	if nMaint > 0 {
		pfcMnt, pfcMntD = nt.AddPFCLayer(prefix+"mnt", nY, nMaint, nNeurY, nNeurX, false)
	}
	if nOut > 0 {
		pfcOut, pfcOutD = nt.AddPFCLayer(prefix+"out", nY, nOut, nNeurY, nNeurX, true)
	}
	if pfcOut != nil && pfcMnt != nil {
		pfcOut.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: pfcMnt.Name(), YAlign: relpos.Front, Space: 2})
		pj := nt.ConnectLayers(pfcMntD, pfcOut, prjn.NewOneToOne(), emer.Forward)
		pj.SetClass("PFCMntDToOut")
	}
	return
}

// AddPBWM adds a DorsalBG an PFC with given params
func (nt *Network) AddPBWM(prefix string, nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX int) (mtxGo, mtxNoGo, gpe, gpi, pfcMnt, pfcMntD, pfcOut, pfcOutD leabra.LeabraLayer) {
	mtxGo, mtxNoGo, gpe, gpi = nt.AddDorsalBG(prefix, nY, nMaint, nOut, nNeurBgY, nNeurBgX)
	pfcMnt, pfcMntD, pfcOut, pfcOutD = nt.AddPFC(prefix, nY, nMaint, nOut, nNeurPfcY, nNeurPfcX)
	if pfcMnt != nil {
		pfcMnt.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: mtxGo.Name(), YAlign: relpos.Front, XAlign: relpos.Left})
	}
	gpl := gpi.(*GPiThalLayer)
	gpl.SendToMatrixPFC(prefix) // sends gating to all these layers
	gpl.SendGateShape()
	return
}

// AddClampDaLayer adds a ClampDaLayer of given name
func (nt *Network) AddClampDaLayer(name string) *ClampDaLayer {
	da := &ClampDaLayer{}
	nt.AddLayerInit(da, name, []int{1, 1}, emer.Input)
	return da
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func (nt *Network) AddTDLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td leabra.LeabraLayer) {
	rew = &Layer{}
	nt.AddLayerInit(rew, prefix+"Rew", []int{1, 1}, emer.Input)
	rp = &TDRewPredLayer{}
	nt.AddLayerInit(rp, prefix+"RewPred", []int{1, 1}, emer.Hidden)
	ri = &TDRewIntegLayer{}
	nt.AddLayerInit(ri, prefix+"RewInteg", []int{1, 1}, emer.Hidden)
	td = &TDDaLayer{}
	nt.AddLayerInit(td, prefix+"TD", []int{1, 1}, emer.Hidden)
	ri.(*TDRewIntegLayer).RewInteg.RewPred = rp.Name()
	td.(*TDDaLayer).RewInteg = ri.Name()
	rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	ri.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})
	td.SetRelPos(relpos.Rel{Rel: rel, Other: ri.Name(), YAlign: relpos.Front, Space: space})

	pj := nt.ConnectLayers(rew, ri, prjn.NewFull(), emer.Forward).(leabra.LeabraPrjn).AsLeabra()
	pj.SetClass("TDRewToInteg")
	pj.Learn.Learn = false
	pj.WtInit.Mean = 1
	pj.WtInit.Var = 0
	pj.WtInit.Sym = false
	// {Sel: ".TDRewToInteg", Desc: "rew to integ",
	// 	Params: params.Params{
	// 		"Prjn.Learn.Learn": "false",
	// 		"Prjn.WtInit.Mean": "1",
	// 		"Prjn.WtInit.Var":  "0",
	// 		"Prjn.WtInit.Sym":  "false",
	// 	}},
	return
}

// AddRWLayers adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
// Projection from RWPred to DA is given class RWPredToDA -- should
// have no learning and 1 weight.
func (nt *Network) AddRWLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, da leabra.LeabraLayer) {
	rew = &Layer{}
	nt.AddLayerInit(rew, prefix+"Rew", []int{1, 1}, emer.Input)
	rp = &RWPredLayer{}
	nt.AddLayerInit(rp, prefix+"RWPred", []int{1, 1}, emer.Hidden)
	da = &RWDaLayer{}
	nt.AddLayerInit(da, prefix+"DA", []int{1, 1}, emer.Hidden)
	da.(*RWDaLayer).RewLay = rew.Name()
	rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	da.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})

	pj := nt.ConnectLayers(rp, da, prjn.NewFull(), emer.Forward).(leabra.LeabraPrjn).AsLeabra()
	pj.SetClass("RWPredToDA")
	pj.Learn.Learn = false
	pj.WtInit.Mean = 1
	pj.WtInit.Var = 0
	pj.WtInit.Sym = false
	// {Sel: ".RWPredToDA", Desc: "rew to da",
	// 	Params: params.Params{
	// 		"Prjn.Learn.Learn": "false",
	// 		"Prjn.WtInit.Mean": "1",
	// 		"Prjn.WtInit.Var":  "0",
	// 		"Prjn.WtInit.Sym":  "false",
	// 	}},
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// Cycle runs one cycle of activation updating
// PBWM calls GateSend after Cycle and before DeepBurst
// Deep version adds call to update DeepBurst at end
func (nt *Network) Cycle(ltime *leabra.Time) {
	nt.Network.Network.Cycle(ltime) // basic version from leabra.Network (not deep.Network, which calls DeepBurst)
	nt.GateSend(ltime)              // GateLayer (GPiThal) computes gating, sends to other layers
	nt.RecGateAct(ltime)            // Record activation state at time of gating (in ActG neuron var)
	nt.DeepBurst(ltime)             // Act -> Burst (during BurstQtr) (see deep for details)
	nt.SendMods(ltime)              // send modulators (DA)
}

// GateSend is called at end of Cycle, computes Gating and sends to other layers
func (nt *Network) GateSend(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(PBWMLayer).GateSend(ltime) }, "GateSend")
}

// RecGateAct is called after GateSend, to record gating activations at time of gating
func (nt *Network) RecGateAct(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(PBWMLayer).RecGateAct(ltime) }, "RecGateAct")
}

// SendMods is called at end of Cycle to send modulator signals (DA, etc)
// which will then be active for the next cycle of processing
func (nt *Network) SendMods(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(PBWMLayer).SendMods(ltime) }, "SendMods")
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods
