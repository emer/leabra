// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/emer/emergent/v2/paths"
)

// GateSend is called at end of Cycle, computes Gating and sends to other layers
func (nt *Network) GateSend(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		if ly.Type == GPiThalLayer {
			ly.GPiGateSend(ctx)
		}
	}
}

// RecGateAct is called after GateSend, to record gating activations at time of gating
func (nt *Network) RecGateAct(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.RecGateAct(ctx)
	}
}

// SendMods is called at end of Cycle to send modulator signals (DA, etc)
// which will then be active for the next cycle of processing
func (nt *Network) SendMods(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.SendMods(ctx)
	}
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMatrixLayer(name string, nY, nMaint, nOut, nNeurY, nNeurX int, da DaReceptors) *Layer {
	tX := nMaint + nOut
	mtx := nt.AddLayer4D(name, nY, tX, nNeurY, nNeurX, MatrixLayer)
	mtx.PBWM.DaR = da
	mtx.PBWM.Set(nY, nMaint, nOut)
	return mtx
}

// AddGPeLayer adds a pbwm.Layer to serve as a GPe layer, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func (nt *Network) AddGPeLayer(name string, nY, nMaint, nOut int) *Layer {
	tX := nMaint + nOut
	gpe := nt.AddLayer4D(name, nY, tX, 1, 1, GPeLayer)
	return gpe
}

// AddGPiThalLayer adds a GPiThalLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func (nt *Network) AddGPiThalLayer(name string, nY, nMaint, nOut int) *Layer {
	tX := nMaint + nOut
	gpi := nt.AddLayer4D(name, nY, tX, 1, 1, GPiThalLayer)
	gpi.PBWM.Set(nY, nMaint, nOut)
	return gpi
}

// AddCINLayer adds a CINLayer, with a single neuron.
func (nt *Network) AddCINLayer(name string) *Layer {
	cin := nt.AddLayer2D(name, 1, 1, CINLayer)
	return cin
}

// AddDorsalBG adds MatrixGo, NoGo, GPe, GPiThal, and CIN layers, with given optional prefix.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  Appropriate PoolOneToOne connections
// are made to drive GPiThal, with BgFixed class name set so
// they can be styled appropriately (no learning, WtRnd.Mean=0.8, Var=0)
func (nt *Network) AddDorsalBG(prefix string, nY, nMaint, nOut, nNeurY, nNeurX int) (mtxGo, mtxNoGo, gpe, gpi, cin *Layer) {
	mtxGo = nt.AddMatrixLayer(prefix+"MatrixGo", nY, nMaint, nOut, nNeurY, nNeurX, D1R)
	mtxNoGo = nt.AddMatrixLayer(prefix+"MatrixNoGo", nY, nMaint, nOut, nNeurY, nNeurX, D2R)
	gpe = nt.AddGPeLayer(prefix+"GPeNoGo", nY, nMaint, nOut)
	gpi = nt.AddGPiThalLayer(prefix+"GPiThal", nY, nMaint, nOut)
	cin = nt.AddCINLayer(prefix + "CIN")
	cin.AddSendTo(mtxGo.Name, mtxNoGo.Name)

	mtxNoGo.PlaceBehind(mtxGo, 2)
	gpe.PlaceRightOf(mtxNoGo, 2)
	gpi.PlaceRightOf(mtxGo, 2)
	cin.PlaceBehind(gpe, 2)

	one2one := paths.NewPoolOneToOne()
	pt := nt.ConnectLayers(mtxGo, gpi, one2one, GPiThalPath)
	pt.AddClass("BgFixed")
	pt = nt.ConnectLayers(mtxNoGo, gpe, one2one, ForwardPath)
	pt.AddClass("BgFixed")
	pt = nt.ConnectLayers(gpe, gpi, one2one, GPiThalPath)
	pt.AddClass("BgFixed")
	return
}

// AddPFCLayer adds a PFCLayer, super and deep, of given size, with given name.
// nY, nX = number of pools in Y, X dimensions, and each pool has nNeurY, nNeurX neurons.
// out is true for output-gating layer, and dynmaint is true for maintenance-only dyn,
// else Full set of 5 dynamic maintenance types. Both have the class "PFC" set.
// deep is positioned behind super.
func (nt *Network) AddPFCLayer(name string, nY, nX, nNeurY, nNeurX int, out, dynMaint bool) (sp, dp *Layer) {
	sp = nt.AddLayer4D(name, nY, nX, nNeurY, nNeurX, SuperLayer)
	dym := 1
	if !dynMaint {
		dym = 5
	}
	dp = nt.AddLayer4D(name+"D", nY, nX, dym*nNeurY, nNeurX, PFCDeepLayer)
	sp.AddClass("PFC")
	dp.AddClass("PFC")
	dp.PFCGate.OutGate = out
	if dynMaint {
		dp.PFCDyns.MaintOnly()
	} else {
		dp.PFCDyns.FullDyn(10)
	}
	dp.PlaceBehind(sp, 2)
	return
}

// AddPFC adds paired PFCmnt, PFCout and associated Deep layers,
// with given optional prefix.
// nY = number of pools in Y dimension, nMaint, nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.
// dynMaint is true for maintenance-only dyn, else full set of 5 dynamic maintenance types.
// Appropriate OneToOne connections are made between PFCmntD -> PFCout.
func (nt *Network) AddPFC(prefix string, nY, nMaint, nOut, nNeurY, nNeurX int, dynMaint bool) (pfcMnt, pfcMntD, pfcOut, pfcOutD *Layer) {
	if prefix == "" {
		prefix = "PFC"
	}
	if nMaint > 0 {
		pfcMnt, pfcMntD = nt.AddPFCLayer(prefix+"mnt", nY, nMaint, nNeurY, nNeurX, false, dynMaint)
	}
	if nOut > 0 {
		pfcOut, pfcOutD = nt.AddPFCLayer(prefix+"out", nY, nOut, nNeurY, nNeurX, true, dynMaint)
	}

	// todo: need a Rect pathway from MntD -> out if !dynMaint, or something else..

	if pfcOut != nil && pfcMnt != nil {
		pfcOut.PlaceRightOf(pfcMnt, 2)
		pt := nt.ConnectLayers(pfcMntD, pfcOut, paths.NewOneToOne(), ForwardPath)
		pt.AddClass("PFCMntDToOut")
	}
	return
}

// AddPBWM adds a DorsalBG and PFC with given params
// Defaults to simple case of basic maint dynamics in Deep
func (nt *Network) AddPBWM(prefix string, nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX int) (mtxGo, mtxNoGo, gpe, gpi, cin, pfcMnt, pfcMntD, pfcOut, pfcOutD *Layer) {
	mtxGo, mtxNoGo, gpe, gpi, cin = nt.AddDorsalBG(prefix, nY, nMaint, nOut, nNeurBgY, nNeurBgX)
	pfcMnt, pfcMntD, pfcOut, pfcOutD = nt.AddPFC(prefix, nY, nMaint, nOut, nNeurPfcY, nNeurPfcX, true) // default dynmaint
	if pfcMnt != nil {
		pfcMnt.PlaceAbove(mtxGo)
	}
	gpi.SendToMatrixPFC(prefix) // sends gating to all these layers
	gpi.SendPBWMParams()
	return
}
