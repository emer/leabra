// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/leabra/v2/leabra"
)

// pbwm.Network has methods for configuring specialized PBWM network components
type Network struct {
	leabra.Network
}

// NewLayer returns new layer of default pbwm.Layer type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPath returns new path of default type
func (nt *Network) NewPath() emer.Path {
	return &leabra.Path{}
}

// Defaults sets all the default parameters for all layers and pathways
func (nt *Network) Defaults() {
	nt.Network.Defaults()
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and pathways
func (nt *Network) UpdateParams() {
	nt.Network.UpdateParams()
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return NeuronVarsAll
}

// SynVarsAll is the pbwm collection of all synapse-level vars (includes TraceSynVars)
var SynVarsAll []string

func init() {
	ln := len(leabra.SynapseVars)
	SynVarsAll = make([]string, len(TraceSynVars)+ln)
	copy(SynVarsAll, leabra.SynapseVars)
	copy(SynVarsAll[ln:], TraceSynVars)
}

// SynVarNames returns the names of all the variables on the synapses in this network.
func (nt *Network) SynVarNames() []string {
	return SynVarsAll
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMatrixLayer(name string, nY, nMaint, nOut, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	tX := nMaint + nOut
	mtx := &MatrixLayer{}
	nt.AddLayerInit(mtx, name, []int{nY, tX, nNeurY, nNeurX}, leabra.SuperLayer)
	mtx.DaR = da
	mtx.GateShp.Set(nY, nMaint, nOut)
	return mtx
}

// AddGPeLayer adds a pbwm.Layer to serve as a GPe layer, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func (nt *Network) AddGPeLayer(name string, nY, nMaint, nOut int) *Layer {
	return AddGPeLayer(&nt.Network, name, nY, nMaint, nOut)
}

// AddGPiThalLayer adds a GPiThalLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func (nt *Network) AddGPiThalLayer(name string, nY, nMaint, nOut int) *GPiThalLayer {
	return AddGPiThalLayer(&nt.Network, name, nY, nMaint, nOut)
}

// AddCINLayer adds a CINLayer, with a single neuron.
func (nt *Network) AddCINLayer(name string) *CINLayer {
	return AddCINLayer(&nt.Network, name)
}

// AddDorsalBG adds MatrixGo, NoGo, GPe, GPiThal, and CIN layers, with given optional prefix.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  Appropriate PoolOneToOne connections
// are made to drive GPiThal, with BgFixed class name set so
// they can be styled appropriately (no learning, WtRnd.Mean=0.8, Var=0)
func (nt *Network) AddDorsalBG(prefix string, nY, nMaint, nOut, nNeurY, nNeurX int) (mtxGo, mtxNoGo, gpe, gpi, cin leabra.LeabraLayer) {
	return AddDorsalBG(&nt.Network, prefix, nY, nMaint, nOut, nNeurY, nNeurX)
}

// AddPFCLayer adds a PFCLayer, super and deep, of given size, with given name.
// nY, nX = number of pools in Y, X dimensions, and each pool has nNeurY, nNeurX neurons.
// out is true for output-gating layer, and dynmaint is true for maintenance-only dyn,
// else Full set of 5 dynamic maintenance types. Both have the class "PFC" set.
// deep is positioned behind super.
func (nt *Network) AddPFCLayer(name string, nY, nX, nNeurY, nNeurX int, out, dynMaint bool) (sp, dp leabra.LeabraLayer) {
	return AddPFCLayer(&nt.Network, name, nY, nX, nNeurY, nNeurX, out, dynMaint)
}

// AddPFC adds paired PFCmnt, PFCout and associated Deep layers,
// with given optional prefix.
// nY = number of pools in Y dimension, nMaint, nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.
// dynMaint is true for maintenance-only dyn, else full set of 5 dynamic maintenance types.
// Appropriate OneToOne connections are made between PFCmntD -> PFCout.
func (nt *Network) AddPFC(prefix string, nY, nMaint, nOut, nNeurY, nNeurX int, dynMaint bool) (pfcMnt, pfcMntD, pfcOut, pfcOutD leabra.LeabraLayer) {
	return AddPFC(&nt.Network, prefix, nY, nMaint, nOut, nNeurY, nNeurX, dynMaint)
}

// AddPBWM adds a DorsalBG and PFC with given params
// Defaults to simple case of basic maint dynamics in Deep
func (nt *Network) AddPBWM(prefix string, nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX int) (mtxGo, mtxNoGo, gpe, gpi, cin, pfcMnt, pfcMntD, pfcOut, pfcOutD leabra.LeabraLayer) {
	return AddPBWM(&nt.Network, prefix, nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX)
}

////////////////////////////////////////////////////////////////////////
// Network functions available here as standalone functions
//         for mixing in to other models

// AddCINLayer adds a CINLayer, with a single neuron.
func AddCINLayer(nt *leabra.Network, name string) *CINLayer {
	ly := &CINLayer{}
	nt.AddLayerInit(ly, name, []int{1, 1}, leabra.SuperLayer)
	return ly
}

// AddMatrixLayer adds a MatrixLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func AddMatrixLayer(nt *leabra.Network, name string, nY, nMaint, nOut, nNeurY, nNeurX int, da DaReceptors) *MatrixLayer {
	tX := nMaint + nOut
	mtx := &MatrixLayer{}
	nt.AddLayerInit(mtx, name, []int{nY, tX, nNeurY, nNeurX}, leabra.SuperLayer)
	mtx.DaR = da
	mtx.GateShp.Set(nY, nMaint, nOut)
	return mtx
}

// AddGPeLayer adds a pbwm.Layer to serve as a GPe layer, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func AddGPeLayer(nt *leabra.Network, name string, nY, nMaint, nOut int) *Layer {
	tX := nMaint + nOut
	gpe := &Layer{}
	nt.AddLayerInit(gpe, name, []int{nY, tX, 1, 1}, leabra.SuperLayer)
	return gpe
}

// AddGPiThalLayer adds a GPiThalLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has 1x1 neurons.
func AddGPiThalLayer(nt *leabra.Network, name string, nY, nMaint, nOut int) *GPiThalLayer {
	tX := nMaint + nOut
	gpi := &GPiThalLayer{}
	nt.AddLayerInit(gpi, name, []int{nY, tX, 1, 1}, leabra.SuperLayer)
	gpi.GateShp.Set(nY, nMaint, nOut)
	return gpi
}

// AddDorsalBG adds MatrixGo, NoGo, GPe, GPiThal, and CIN layers, with given optional prefix.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  Appropriate PoolOneToOne connections
// are made to drive GPiThal, with BgFixed class name set so
// they can be styled appropriately (no learning, WtRnd.Mean=0.8, Var=0)
func AddDorsalBG(nt *leabra.Network, prefix string, nY, nMaint, nOut, nNeurY, nNeurX int) (mtxGo, mtxNoGo, gpe, gpi, cin leabra.LeabraLayer) {
	mtxGo = AddMatrixLayer(nt, prefix+"MatrixGo", nY, nMaint, nOut, nNeurY, nNeurX, D1R)
	mtxNoGo = AddMatrixLayer(nt, prefix+"MatrixNoGo", nY, nMaint, nOut, nNeurY, nNeurX, D2R)
	gpe = AddGPeLayer(nt, prefix+"GPeNoGo", nY, nMaint, nOut)
	gpi = AddGPiThalLayer(nt, prefix+"GPiThal", nY, nMaint, nOut)
	cini := AddCINLayer(nt, prefix+"CIN")
	cin = cini
	cini.SendACh.Add(mtxGo.Name(), mtxNoGo.Name())

	mtxNoGo.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: mtxGo.Name(), XAlign: relpos.Left, Space: 2})
	gpe.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxNoGo.Name(), YAlign: relpos.Front, Space: 2})
	gpi.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: mtxGo.Name(), YAlign: relpos.Front, Space: 2})
	cin.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: gpe.Name(), XAlign: relpos.Left, Space: 2})

	pj := nt.ConnectLayersPath(mtxGo, gpi, paths.NewPoolOneToOne(), leabra.ForwardPath, &GPiThalPath{})
	pj.SetClass("BgFixed")
	pj = nt.ConnectLayers(mtxNoGo, gpe, paths.NewPoolOneToOne(), leabra.ForwardPath)
	pj.SetClass("BgFixed")
	pj = nt.ConnectLayersPath(gpe, gpi, paths.NewPoolOneToOne(), leabra.ForwardPath, &GPiThalPath{})
	pj.SetClass("BgFixed")

	return
}

// AddPFCLayer adds a PFCLayer, super and deep, of given size, with given name.
// nY, nX = number of pools in Y, X dimensions, and each pool has nNeurY, nNeurX neurons.
// out is true for output-gating layer, and dynmaint is true for maintenance-only dyn,
// else Full set of 5 dynamic maintenance types. Both have the class "PFC" set.
// deep is positioned behind super.
func AddPFCLayer(nt *leabra.Network, name string, nY, nX, nNeurY, nNeurX int, out, dynMaint bool) (sp, dp leabra.LeabraLayer) {
	sp = nt.AddLayer(name, []int{nY, nX, nNeurY, nNeurX}, leabra.SuperLayer).(leabra.LeabraLayer)
	ddp := &PFCDeepLayer{}
	dp = ddp
	dym := 1
	if !dynMaint {
		dym = 5
	}
	nt.AddLayerInit(ddp, name+"D", []int{nY, nX, dym * nNeurY, nNeurX}, leabra.SuperLayer)
	sp.SetClass("PFC")
	ddp.SetClass("PFC")
	ddp.Gate.OutGate = out
	if dynMaint {
		ddp.Dyns.MaintOnly()
	} else {
		ddp.Dyns.FullDyn(10)
	}
	dp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	return
}

// AddPFC adds paired PFCmnt, PFCout and associated Deep layers,
// with given optional prefix.
// nY = number of pools in Y dimension, nMaint, nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.
// dynMaint is true for maintenance-only dyn, else full set of 5 dynamic maintenance types.
// Appropriate OneToOne connections are made between PFCmntD -> PFCout.
func AddPFC(nt *leabra.Network, prefix string, nY, nMaint, nOut, nNeurY, nNeurX int, dynMaint bool) (pfcMnt, pfcMntD, pfcOut, pfcOutD leabra.LeabraLayer) {
	if prefix == "" {
		prefix = "PFC"
	}
	if nMaint > 0 {
		pfcMnt, pfcMntD = AddPFCLayer(nt, prefix+"mnt", nY, nMaint, nNeurY, nNeurX, false, dynMaint)
	}
	if nOut > 0 {
		pfcOut, pfcOutD = AddPFCLayer(nt, prefix+"out", nY, nOut, nNeurY, nNeurX, true, dynMaint)
	}

	// todo: need a Rect pathway from MntD -> out if !dynMaint, or something else..

	if pfcOut != nil && pfcMnt != nil {
		pfcOut.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: pfcMnt.Name(), YAlign: relpos.Front, Space: 2})
		pj := nt.ConnectLayers(pfcMntD, pfcOut, paths.NewOneToOne(), leabra.ForwardPath)
		pj.SetClass("PFCMntDToOut")
	}
	return
}

// AddPBWM adds a DorsalBG and PFC with given params
// Defaults to simple case of basic maint dynamics in Deep
func AddPBWM(nt *leabra.Network, prefix string, nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX int) (mtxGo, mtxNoGo, gpe, gpi, cin, pfcMnt, pfcMntD, pfcOut, pfcOutD leabra.LeabraLayer) {
	mtxGo, mtxNoGo, gpe, gpi, cin = AddDorsalBG(nt, prefix, nY, nMaint, nOut, nNeurBgY, nNeurBgX)
	pfcMnt, pfcMntD, pfcOut, pfcOutD = AddPFC(nt, prefix, nY, nMaint, nOut, nNeurPfcY, nNeurPfcX, true) // default dynmaint
	if pfcMnt != nil {
		pfcMnt.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: mtxGo.Name(), YAlign: relpos.Front, XAlign: relpos.Left})
	}
	gpl := gpi.(*GPiThalLayer)
	gpl.SendToMatrixPFC(prefix) // sends gating to all these layers
	gpl.SendGateShape()
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Python versions

// AddDorsalBGPy adds MatrixGo, NoGo, GPe, GPiThal, and CIN layers, with given optional prefix.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  Appropriate PoolOneToOne connections
// are made to drive GPiThal, with BgFixed class name set so
// they can be styled appropriately (no learning, WtRnd.Mean=0.8, Var=0)
// Py is Python version, returns layers as a slice
func AddDorsalBGPy(nt *leabra.Network, prefix string, nY, nMaint, nOut, nNeurY, nNeurX int) []leabra.LeabraLayer {
	mtxGo, mtxNoGo, gpe, gpi, cin := AddDorsalBG(nt, prefix, nY, nMaint, nOut, nNeurY, nNeurX)
	return []leabra.LeabraLayer{mtxGo, mtxNoGo, gpe, gpi, cin}
}

// AddPFCPy adds paired PFCmnt, PFCout and associated Deep layers,
// with given optional prefix.
// nY = number of pools in Y dimension, nMaint, nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.
// dynMaint is true for maintenance-only dyn, else full set of 5 dynamic maintenance types.
// Appropriate OneToOne connections are made between PFCmntD -> PFCout.
// Py is Python version, returns layers as a slice
func AddPFCPy(nt *leabra.Network, prefix string, nY, nMaint, nOut, nNeurY, nNeurX int, dynMaint bool) []leabra.LeabraLayer {
	pfcMnt, pfcMntD, pfcOut, pfcOutD := AddPFC(nt, prefix, nY, nMaint, nOut, nNeurY, nNeurX, dynMaint)
	return []leabra.LeabraLayer{pfcMnt, pfcMntD, pfcOut, pfcOutD}
}

// AddPBWMPy adds a DorsalBG and PFC with given params
// Defaults to simple case of basic maint dynamics in Deep
// Py is Python version, returns layers as a slice
func AddPBWMPy(nt *leabra.Network, prefix string, nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX int) []leabra.LeabraLayer {
	mtxGo, mtxNoGo, gpe, gpi, cin, pfcMnt, pfcMntD, pfcOut, pfcOutD := AddPBWM(nt, prefix, nY, nMaint, nOut, nNeurBgY, nNeurBgX, nNeurPfcY, nNeurPfcX)
	return []leabra.LeabraLayer{mtxGo, mtxNoGo, gpe, gpi, cin, pfcMnt, pfcMntD, pfcOut, pfcOutD}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// CycleImpl runs one cycle of activation updating
// PBWM calls GateSend after Cycle and before DeepBurst
func (nt *Network) CycleImpl(ctx *leabra.Context) {
	nt.Network.CycleImpl(ctx) // basic version from leabra.Network
	nt.GateSend(ctx)          // GateLayer (GPiThal) computes gating, sends to other layers
	nt.RecGateAct(ctx)        // Record activation state at time of gating (in ActG neuron var)

	nt.EmerNet.(leabra.LeabraNetwork).CyclePostImpl(ctx) // always call this after std cycle..
}

// GateSend is called at end of Cycle, computes Gating and sends to other layers
func (nt *Network) GateSend(ctx *leabra.Context) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(PBWMLayer); ok {
			pl.GateSend(ctx)
		}
	}, "GateSend")
}

// RecGateAct is called after GateSend, to record gating activations at time of gating
func (nt *Network) RecGateAct(ctx *leabra.Context) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(PBWMLayer); ok {
			pl.RecGateAct(ctx)
		}
	}, "RecGateAct")
}

// SendMods is called at end of Cycle to send modulator signals (DA, etc)
// which will then be active for the next cycle of processing
func (nt *Network) SendMods(ctx *leabra.Context) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(PBWMLayer); ok {
			pl.SendMods(ctx)
		}
	}, "SendMods")
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods
