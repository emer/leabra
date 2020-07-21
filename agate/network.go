// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/leabra/deep"
	"github.com/emer/leabra/leabra"
	"github.com/emer/leabra/pcore"
	"github.com/goki/ki/kit"
)

// agate.Network has methods for configuring specialized AGate network components
// for Attentional & adaptive Gating of Action and Thought for Executive function.
type Network struct {
	deep.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = deep.NetworkProps

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

// AddBG adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi, and VThal layers,
// with given optional prefix.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Only Matrix has more than 1 unit per Pool by default.
// Appropriate PoolOneToOne connections are made between layers,
// using standard styles
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi, vthal leabra.LeabraLayer) {
	return pcore.AddBG(&nt.Network.Network, prefix, nPoolsY, nPoolsX, nNeurY, nNeurX)
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
