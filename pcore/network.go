// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// pcore.Network has methods for configuring specialized PCore network components
// PCore = Pallidal Core mode of BG
type Network struct {
	leabra.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = leabra.NetworkProps

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

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return NeuronVarsAll
}

// SynVarNames returns the names of all the variables on the synapses in this network.
func (nt *Network) SynVarNames() []string {
	return SynVarsAll
}

// AddCINLayer adds a CINLayer, with a single neuron.
func (nt *Network) AddCINLayer(name string) *CINLayer {
	return AddCINLayer(&nt.Network, name)
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMatrixLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	return AddMatrixLayer(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, da)
}

// ConnectToMatrix adds a MatrixTracePrjn from given sending layer to a matrix layer
func (nt *Network) ConnectToMatrix(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return ConnectToMatrix(&nt.Network, send, recv, pat)
}

// AddGPLayer adds a GPLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func (nt *Network) AddGPeLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPLayer {
	return AddGPeLayer(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddGPiLayer adds a GPiLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func (nt *Network) AddGPiLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPiLayer {
	return AddGPiLayer(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddSTNLayer adds a subthalamic nucleus Layer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func (nt *Network) AddSTNLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *STNLayer {
	return AddSTNLayer(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddVThalLayer adds a ventral thalamus (VA/VL/VM) Layer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func (nt *Network) AddVThalLayer(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *VThalLayer {
	return AddVThalLayer(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddBG adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi, and VThal layers,
// with given optional prefix.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Only Matrix has more than 1 unit per Pool by default.
// Appropriate PoolOneToOne connections are made between layers,
// using standard styles
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi, vthal leabra.LeabraLayer) {
	return AddBG(&nt.Network, prefix, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

////////////////////////////////////////////////////////////////////////
// Network routines available here for mixing in to other models

// AddCINLayer adds a CINLayer, with a single neuron.
func AddCINLayer(nt *leabra.Network, name string) *CINLayer {
	tan := &CINLayer{}
	nt.AddLayerInit(tan, name, []int{1, 1}, emer.Hidden)
	return tan
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func AddMatrixLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	mtx := &MatrixLayer{}
	nt.AddLayerInit(mtx, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	mtx.DaR = da
	return mtx
}

// ConnectToMatrix adds a MatrixTracePrjn from given sending layer to a matrix layer
func ConnectToMatrix(nt *leabra.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, emer.Forward, &MatrixPrjn{})
}

// AddGPLayer adds a GPLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func AddGPeLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPLayer {
	gp := &GPLayer{}
	nt.AddLayerInit(gp, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	gp.SetClass("GP")
	return gp
}

// AddGPiLayer adds a GPiLayer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func AddGPiLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *GPiLayer {
	gpi := &GPiLayer{}
	nt.AddLayerInit(gpi, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	gpi.SetClass("GP")
	return gpi
}

// AddSTNLayer adds a subthalamic nucleus Layer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func AddSTNLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *STNLayer {
	stn := &STNLayer{}
	nt.AddLayerInit(stn, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return stn
}

// AddVThalLayer adds a ventral thalamus (VA/VL/VM) Layer of given size, with given name.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Typically nNeurY, nNeurX will both be 1, but could have more for noise etc.
func AddVThalLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *VThalLayer {
	vthal := &VThalLayer{}
	nt.AddLayerInit(vthal, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return vthal
}

// AddBG adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi, and VThal layers,
// with given optional prefix.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Only Matrix has more than 1 unit per Pool by default.
// Appropriate PoolOneToOne connections are made between layers,
// using standard styles
func AddBG(nt *leabra.Network, prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi, vthal leabra.LeabraLayer) {
	gpi = AddGPiLayer(nt, prefix+"GPi", nPoolsY, nPoolsX, 1, 1)
	vthal = AddVThalLayer(nt, prefix+"VThal", nPoolsY, nPoolsX, 1, 1)
	gpeOuti := AddGPeLayer(nt, prefix+"GPeOut", nPoolsY, nPoolsX, 1, 1)
	gpeOuti.GPLay = GPeOut
	gpeOut = gpeOuti
	gpeIni := AddGPeLayer(nt, prefix+"GPeIn", nPoolsY, nPoolsX, 1, 1)
	gpeIni.GPLay = GPeIn
	gpeIn = gpeIni
	gpeTAi := AddGPeLayer(nt, prefix+"GPeTA", nPoolsY, nPoolsX, 1, 1)
	gpeTAi.GPLay = GPeTA
	gpeTA = gpeTAi
	stnp = AddSTNLayer(nt, prefix+"STNp", nPoolsY, nPoolsX, 1, 1)
	stns = AddSTNLayer(nt, prefix+"STNs", nPoolsY, nPoolsX, 1, 1)
	mtxGo = AddMatrixLayer(nt, prefix+"MtxGo", nPoolsY, nPoolsX, nNeurY, nNeurX, D1R)
	mtxNo = AddMatrixLayer(nt, prefix+"MtxNo", nPoolsY, nPoolsX, nNeurY, nNeurX, D2R)
	cin = AddCINLayer(nt, prefix+"CIN")

	vthal.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpi.Name(), YAlign: relpos.Front, Space: 2})

	gpeOut.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: gpi.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	gpeIn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeOut.Name(), YAlign: relpos.Front, Space: 2})
	gpeTA.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeIn.Name(), YAlign: relpos.Front, Space: 2})
	stnp.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gpeTA.Name(), YAlign: relpos.Front, Space: 2})
	stns.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: stnp.Name(), YAlign: relpos.Front, Space: 2})

	mtxGo.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: gpeOut.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	mtxNo.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxGo.Name(), YAlign: relpos.Front, Space: 2})
	cin.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxNo.Name(), YAlign: relpos.Front, Space: 2})

	one2one := prjn.NewPoolOneToOne()
	full := prjn.NewFull()

	pj := nt.ConnectLayers(mtxGo, gpeOut, one2one, emer.Inhib)
	pj.SetClass("BgFixed")

	nt.ConnectLayers(mtxNo, gpeIn, one2one, emer.Inhib)
	nt.ConnectLayers(gpeOut, gpeIn, one2one, emer.Inhib)

	pj = nt.ConnectLayers(gpeIn, gpeTA, one2one, emer.Inhib)
	pj.SetClass("BgFixed")
	pj = nt.ConnectLayers(gpeIn, stnp, one2one, emer.Inhib)
	pj.SetClass("BgFixed")

	// note: this projection exists in bio, but does weird things with Ca dynamics in STNs..
	// pj = nt.ConnectLayers(gpeIn, stns, one2one, emer.Inhib)
	// pj.SetClass("BgFixed")

	nt.ConnectLayers(gpeIn, gpi, one2one, emer.Inhib)
	nt.ConnectLayers(mtxGo, gpi, one2one, emer.Inhib)

	pj = nt.ConnectLayers(stnp, gpeOut, one2one, emer.Forward)
	pj.SetClass("FmSTNp")
	pj = nt.ConnectLayers(stnp, gpeIn, one2one, emer.Forward)
	pj.SetClass("FmSTNp")
	pj = nt.ConnectLayers(stnp, gpeTA, full, emer.Forward)
	pj.SetClass("FmSTNp")
	pj = nt.ConnectLayers(stnp, gpi, one2one, emer.Forward)
	pj.SetClass("FmSTNp")

	pj = nt.ConnectLayers(stns, gpi, one2one, emer.Forward)
	pj.SetClass("FmSTNs")

	pj = nt.ConnectLayers(gpeTA, mtxGo, full, emer.Inhib)
	pj.SetClass("GPeTAToMtx")
	pj = nt.ConnectLayers(gpeTA, mtxNo, full, emer.Inhib)
	pj.SetClass("GPeTAToMtx")

	pj = nt.ConnectLayers(gpeIn, mtxGo, full, emer.Inhib)
	pj.SetClass("GPeInToMtx")
	pj = nt.ConnectLayers(gpeIn, mtxNo, full, emer.Inhib)
	pj.SetClass("GPeInToMtx")

	pj = nt.ConnectLayers(gpi, vthal, one2one, emer.Inhib)
	pj.SetClass("BgFixed")

	return
}
