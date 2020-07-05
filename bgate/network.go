// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgate

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// bgate.Network has methods for configuring specialized BGATE network components
type Network struct {
	deep.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = deep.NetworkProps

// NewLayer returns new layer of default leabra.Layer type
func (nt *Network) NewLayer() emer.Layer {
	return &leabra.Layer{}
}

// NewPrjn returns new prjn of default leabra.Prjn type
func (nt *Network) NewPrjn() emer.Prjn {
	return &leabra.Prjn{}
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
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	mtx := &MatrixLayer{}
	nt.AddLayerInit(mtx, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	mtx.DaR = da
	return mtx
}

// ConnectToMatrix adds a MatrixTracePrjn from given sending layer to a matrix layer
func (nt *Network) ConnectToMatrix(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, emer.Forward, &MatrixTracePrjn{})
}

// AddGPLayer adds a GPLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func (nt *Network) AddGPeLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPLayer {
	gp := &GPLayer{}
	nt.AddLayerInit(gp, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return gp
}

// AddGPiLayer adds a GPiLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func (nt *Network) AddGPiLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPiLayer {
	gpi := &GPiLayer{}
	nt.AddLayerInit(gpi, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return gpi
}

// AddVThalLayer adds a ventral thalamus (VA/VL/VM) Layer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func (nt *Network) AddVThalLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) leabra.LeabraLayer {
	return nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, emer.Hidden).(leabra.LeabraLayer)
}

// AddBG adds MtxGo, No, GPeOut, GPeIn, GPeTA, STN, GPi, and VThal layers, with given optional prefix.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Only Matrix has more than 1 unit per Pool by default.
// Appropriate PoolOneToOne connections are made between layers,
// using standard styles
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (mtxGo, mtxNo, gpeOut, gpeIn, gpeTA, stn, gpi, vthal leabra.LeabraLayer) {
	gpi = nt.AddGPiLayer(prefix+"GPi", nPoolsY, nPoolsX, 1, 1)
	vthal = nt.AddVThalLayer(prefix+"VThal", nPoolsY, nPoolsX, 1, 1)
	gpeOut = nt.AddGPeLayer(prefix+"GPeOut", nPoolsY, nPoolsX, 1, 1)
	gpeIn = nt.AddGPeLayer(prefix+"GPeIn", nPoolsY, nPoolsX, 1, 1)
	gpeTA = nt.AddGPeLayer(prefix+"GPeTA", nPoolsY, nPoolsX, 1, 1)
	stn = nt.AddGPeLayer(prefix+"STN", nPoolsY, nPoolsX, 1, 1)
	mtxGo = nt.AddMatrixLayer(prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1R)
	mtxNo = nt.AddMatrixLayer(prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2R)

	vthal.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpi.Name(), YAlign: relpos.Front, Space: 2})

	gpeOut.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: gpi.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	gpeIn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeOut.Name(), YAlign: relpos.Front, Space: 2})
	gpeTA.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeIn.Name(), YAlign: relpos.Front, Space: 2})
	stn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeTA.Name(), YAlign: relpos.Front, Space: 2})

	mtxGo.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: gpeOut.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	mtxNo.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxGo.Name(), YAlign: relpos.Front, Space: 2})

	one2one := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	pj := nt.ConnectLayers(mtxGo, gpeOut, one2one, emer.Inhib)
	pj.SetClass("BgFixed")

	nt.ConnectLayersPrjn(mtxNo, gpeIn, one2one, emer.Inhib, &GPeInPrjn{})
	nt.ConnectLayersPrjn(gpeOut, gpeIn, one2one, emer.Inhib, &GPeInPrjn{})
	nt.ConnectLayersPrjn(mtxGo, gpeIn, full, emer.Inhib, &GPeInPrjn{}) // full = all but self..

	nt.ConnectLayers(gpeIn, gpeTA, one2one, emer.Inhib)
	nt.ConnectLayers(gpeIn, stn, one2one, emer.Inhib)

	nt.ConnectLayersPrjn(gpeIn, gpi, one2one, emer.Inhib, &GPiPrjn{})
	nt.ConnectLayersPrjn(mtxGo, gpi, one2one, emer.Inhib, &GPiPrjn{})

	pj = nt.ConnectLayers(stn, gpeOut, one2one, emer.Forward)
	pj.SetClass("FmSTN")
	pj = nt.ConnectLayers(stn, gpeIn, one2one, emer.Forward)
	pj.SetClass("FmSTN")
	pj = nt.ConnectLayers(stn, gpeTA, one2one, emer.Forward)
	pj.SetClass("FmSTN")
	pj = nt.ConnectLayers(stn, gpi, one2one, emer.Forward)
	pj.SetClass("FmSTN")

	pj = nt.ConnectLayers(gpeTA, mtxGo, one2one, emer.Inhib)
	pj.SetClass("FmGPeTA")
	pj = nt.ConnectLayers(gpeTA, mtxNo, one2one, emer.Inhib)
	pj.SetClass("FmGPeTA")

	nt.ConnectLayers(gpi, vthal, one2one, emer.Inhib)

	return
}

/*

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

*/
