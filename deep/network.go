// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/leabra/v2/leabra"
)

// deep.Network has parameters for running a DeepLeabra network
type Network struct {
	leabra.Network
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

//////////////////////////////////////////////////////////////////////////////////////
//  Basic Add Layer methods (independent of Network)

// AddSuperLayer2D adds a SuperLayer of given size, with given name.
func AddSuperLayer2D(nt *leabra.Network, name string, nNeurY, nNeurX int) *SuperLayer {
	ly := &SuperLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, leabra.SuperLayer)
	return ly
}

// AddSuperLayer4D adds a SuperLayer of given size, with given name.
func AddSuperLayer4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *SuperLayer {
	ly := &SuperLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, leabra.SuperLayer)
	return ly
}

// AddCTLayer2D adds a CTLayer of given size, with given name.
func AddCTLayer2D(nt *leabra.Network, name string, nNeurY, nNeurX int) *CTLayer {
	ly := &CTLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, CT)
	return ly
}

// AddCTLayer4D adds a CTLayer of given size, with given name.
func AddCTLayer4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *CTLayer {
	ly := &CTLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, CT)
	return ly
}

// AddTRCLayer2D adds a TRCLayer of given size, with given name.
func AddTRCLayer2D(nt *leabra.Network, name string, nNeurY, nNeurX int) *TRCLayer {
	ly := &TRCLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, TRC)
	return ly
}

// AddTRCLayer4D adds a TRCLayer of given size, with given name.
func AddTRCLayer4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *TRCLayer {
	ly := &TRCLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, TRC)
	return ly
}

// ConnectSuperToCT adds a CTCtxtPath from given sending Super layer to a CT layer
// This automatically sets the FromSuper flag to engage proper defaults,
// uses a OneToOne path pattern, and sets the class to CTFromSuper
func ConnectSuperToCT(nt *leabra.Network, send, recv emer.Layer) emer.Path {
	pj := nt.ConnectLayersPath(send, recv, paths.NewOneToOne(), CTCtxt, &CTCtxtPath{}).(*CTCtxtPath)
	pj.SetClass("CTFromSuper")
	pj.FromSuper = true
	return pj
}

// ConnectCtxtToCT adds a CTCtxtPath from given sending layer to a CT layer
// Use ConnectSuperToCT for main pathway from corresponding superficial layer.
func ConnectCtxtToCT(nt *leabra.Network, send, recv emer.Layer, pat paths.Pattern) emer.Path {
	return nt.ConnectLayersPath(send, recv, pat, CTCtxt, &CTCtxtPath{})
}

// ConnectSuperToCTFake adds a FAKE CTCtxtPath from given sending Super layer to a CT layer
// uses a OneToOne path pattern, and sets the class to CTFromSuper.
// This does NOT make a CTCtxtPath -- instead makes a regular leabra.Path -- for testing!
func ConnectSuperToCTFake(nt *leabra.Network, send, recv emer.Layer) emer.Path {
	pj := nt.ConnectLayers(send, recv, paths.NewOneToOne(), CTCtxt)
	pj.SetClass("CTFromSuper")
	return pj
}

// ConnectCtxtToCTFake adds a FAKE CTCtxtPath from given sending layer to a CT layer
// This does NOT make a CTCtxtPath -- instead makes a regular leabra.Path -- for testing!
func ConnectCtxtToCTFake(nt *leabra.Network, send, recv emer.Layer, pat paths.Pattern) emer.Path {
	return nt.ConnectLayers(send, recv, pat, CTCtxt)
}

// AddDeep2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, type = Back, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
func AddDeep2D(nt *leabra.Network, name string, shapeY, shapeX int) (super, ct, trc emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	full := paths.NewFull()
	ConnectSuperToCT(nt, super, ct)
	trci := AddTRCLayer2D(nt, name+"P", shapeY, shapeX)
	trc = trci
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(ct, trc, full, leabra.ForwardPath)
	nt.ConnectLayers(trc, super, full, BackPath).SetClass("FromPulv")
	nt.ConnectLayers(trc, ct, full, BackPath).SetClass("FromPulv")
	return
}

// AddDeep4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, also PoolOneToOne, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
func AddDeep4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, trc emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	pone2one := paths.NewPoolOneToOne()
	ConnectSuperToCT(nt, super, ct)
	trci := AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	trc = trci
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(ct, trc, pone2one, leabra.ForwardPath)
	nt.ConnectLayers(trc, super, pone2one, BackPath).SetClass("FromPulv")
	nt.ConnectLayers(trc, ct, pone2one, BackPath).SetClass("FromPulv")
	return
}

// AddDeepNoTRC2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func AddDeepNoTRC2D(nt *leabra.Network, name string, shapeY, shapeX int) (super, ct emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectSuperToCT(nt, super, ct)
	return
}

// AddDeepNoTRC4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func AddDeepNoTRC4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectSuperToCT(nt, super, ct)
	return
}

// AddDeep2DFakeCT adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with FAKE CTCtxtPath OneToOne pathway from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, type = Back, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
// This does NOT make a CTCtxtPath -- instead makes a regular leabra.Path -- for testing!
func AddDeep2DFakeCT(nt *leabra.Network, name string, shapeY, shapeX int) (super, ct, trc emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	full := paths.NewFull()
	ConnectSuperToCTFake(nt, super, ct)
	trci := AddTRCLayer2D(nt, name+"P", shapeY, shapeX)
	trc = trci
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(ct, trc, full, leabra.ForwardPath)
	nt.ConnectLayers(trc, super, full, BackPath).SetClass("FromPulv")
	nt.ConnectLayers(trc, ct, full, BackPath).SetClass("FromPulv")
	return
}

// AddDeep4DFakeCT adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with FAKE CTCtxtPath OneToOne pathway from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, also PoolOneToOne, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
// This does NOT make a CTCtxtPath -- instead makes a regular leabra.Path -- for testing!
func AddDeep4DFakeCT(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, trc emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	pone2one := paths.NewPoolOneToOne()
	ConnectSuperToCTFake(nt, super, ct)
	trci := AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	trc = trci
	trci.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(ct, trc, pone2one, leabra.ForwardPath)
	nt.ConnectLayers(trc, super, pone2one, BackPath).SetClass("FromPulv")
	nt.ConnectLayers(trc, ct, pone2one, BackPath).SetClass("FromPulv")
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Python versions

// AddDeep2DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath Full pathway from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, type = Back, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
// Py is Python version, returns layers as a slice
func AddDeep2DPy(nt *leabra.Network, name string, shapeY, shapeX int) []emer.Layer {
	super, ct, trc := AddDeep2D(nt, name, shapeY, shapeX)
	return []emer.Layer{super, ct, trc}
}

// AddDeep4DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath PoolOneToOne pathway from Super to CT, and TRC Pulvinar for Super (P suffix).
// TRC projects back to Super and CT layers, also PoolOneToOne, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the TRC layer, and it must be sized appropriately for those drivers.
// Py is Python version, returns layers as a slice
func AddDeep4DPy(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) []emer.Layer {
	super, ct, trc := AddDeep4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	return []emer.Layer{super, ct, trc}
}

// AddDeepNoTRC2DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath Full pathway from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
// Py is Python version, returns layers as a slice
func AddDeepNoTRC2DPy(nt *leabra.Network, name string, shapeY, shapeX int) []emer.Layer {
	super, ct := AddDeepNoTRC2D(nt, name, shapeY, shapeX)
	return []emer.Layer{super, ct}
}

// AddDeepNoTRC4DPy adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath PoolOneToOne pathway from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
// Py is Python version, returns layers as a slice
func AddDeepNoTRC4DPy(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) []emer.Layer {
	super, ct := AddDeepNoTRC4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	return []emer.Layer{super, ct}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network versions of Add Layer methods

// AddDeep2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddDeep2D(name string, shapeY, shapeX int) (super, ct, pulv emer.Layer) {
	return AddDeep2D(&nt.Network, name, shapeY, shapeX)
}

// AddDeep4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddDeep4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, pulv emer.Layer) {
	return AddDeep4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddDeep2DFakeCT adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with FAKE CTCtxtPath OneToOne pathway from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddDeep2DFakeCT(name string, shapeY, shapeX int) (super, ct, pulv emer.Layer) {
	return AddDeep2DFakeCT(&nt.Network, name, shapeY, shapeX)
}

// AddDeep4DFakeCT adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with FAKE CTCtxtPath OneToOne pathway from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddDeep4DFakeCT(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, pulv emer.Layer) {
	return AddDeep4DFakeCT(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddDeepNoTRC2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddDeepNoTRC2D(name string, shapeY, shapeX int) (super, ct emer.Layer) {
	return AddDeepNoTRC2D(&nt.Network, name, shapeY, shapeX)
}

// AddDeepNoTRC4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath PoolOneToOne pathway from Super to CT, and NO TRC Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddDeepNoTRC4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct emer.Layer) {
	return AddDeepNoTRC4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// ConnectCtxtToCT adds a CTCtxtPath from given sending layer to a CT layer
func (nt *Network) ConnectCtxtToCT(send, recv emer.Layer, pat paths.Pattern) emer.Path {
	return ConnectCtxtToCT(&nt.Network, send, recv, pat)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Compute methods

// QuarterFinal does updating after end of a quarter
func (nt *Network) QuarterFinal(ltime *leabra.Time) {
	nt.Network.QuarterFinal(ltime)
	nt.CTCtxt(ltime)
}

// CTCtxt sends context to CT layers and integrates CtxtGe on CT layers
func (nt *Network) CTCtxt(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if dl, ok := ly.(CtxtSender); ok {
			dl.SendCtxtGe(ltime)
		}
	}, "SendCtxtGe")

	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if dl, ok := ly.(*CTLayer); ok {
			dl.CtxtFromGe(ltime)
		}
	}, "CtxtFromGe")
}
