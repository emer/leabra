// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/leabra/attrn"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// deep.Network has parameters for running a DeepLeabra network
type Network struct {
	leabra.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = leabra.NetworkProps

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

//////////////////////////////////////////////////////////////////////////////////////
//  Basic Add Layer methods (independent of Network)

// AddSuperLayer2D adds a SuperLayer of given size, with given name.
func AddSuperLayer2D(nt *leabra.Network, name string, nNeurY, nNeurX int) *SuperLayer {
	ly := &SuperLayer{}
	nt.AddLayerInit(ly, name, []int{nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddSuperLayer4D adds a SuperLayer of given size, with given name.
func AddSuperLayer4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *SuperLayer {
	ly := &SuperLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
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

// AddInputPulv2D adds an input and corresponding Pulvinar (P suffix) TRC layer.
// Pulvinar is placed behind Input.
func AddInputPulv2D(nt *leabra.Network, name string, shapeY, shapeX int) (input, pulv emer.Layer) {
	input = AddSuperLayer2D(nt, name, shapeY, shapeX)
	input.SetType(emer.Input)
	pulvi := AddTRCLayer2D(nt, name+"P", shapeY, shapeX)
	pulv = pulvi
	pulvi.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	pulvi.DriverLay = name
	return
}

// AddInputPulv4D adds an input and corresponding Pulvinar (P suffix) TRC layer
// Pulvinar is placed Behind Input.
func AddInputPulv4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (input, pulv emer.Layer) {
	input = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	input.SetType(emer.Input)
	pulvi := AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	pulv = pulvi
	pulvi.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	pulvi.DriverLay = name
	return
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
func ConnectCtxtToCT(nt *leabra.Network, send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayersPrjn(send, recv, pat, CTCtxt, &CTCtxtPrjn{})
}

// bool args for greater clarity
const (
	AddPulv bool = true
	NoPulv       = false
)

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func AddSuperCT2D(nt *leabra.Network, name string, shapeY, shapeX int, pulvLay bool) (super, ct, pulv emer.Layer) {
	super = AddSuperLayer2D(nt, name, shapeY, shapeX)
	ct = AddCTLayer2D(nt, name+"CT", shapeY, shapeX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectCtxtToCT(nt, super, ct, prjn.NewFull())
	if pulvLay {
		pulvi := AddTRCLayer2D(nt, name+"P", shapeY, shapeX)
		pulv = pulvi
		pulvi.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
		pulvi.DriverLay = name
	}
	return
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func AddSuperCT4D(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) (super, ct, pulv emer.Layer) {
	super = AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	ConnectCtxtToCT(nt, super, ct, prjn.NewFull())
	if pulvLay {
		pulvi := AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
		pulv = pulvi
		pulvi.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
		pulvi.DriverLay = name
	}
	return
}

// AddSuperCTAttn adds a 4D superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT, and a TRN attentional layer.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func AddSuperCTAttn(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) (super, ct, trn, pulv emer.Layer) {
	super, ct, pulv = AddSuperCT4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX, pulvLay)
	trni := attrn.AddTRNLayer(nt, name+"T", nPoolsY, nPoolsX)
	trn = trni
	trni.EPools.Add(name+"CT", 1)
	if pulvLay {
		trni.EPools.Add(name+"P", .2)
		trni.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "P", XAlign: relpos.Left, Space: 2})
	} else {
		trni.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "CT", XAlign: relpos.Left, Space: 2})
	}
	trni.SendTo.Add(name)
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network versions of Add Layer methods

// AddInputPulv2D adds an input and corresponding Pulvinar (P suffix) TRC layer
// Pulvinar is placed Behind Input.
func (nt *Network) AddInputPulv2D(name string, shapeY, shapeX int) (input, pulv emer.Layer) {
	return AddInputPulv2D(&nt.Network, name, shapeY, shapeX)
}

// AddInputPulv4D adds an input and corresponding Pulvinar (P suffix) TRC layer
// Pulvinar is placed Behind Input.
func (nt *Network) AddInputPulv4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (input, pulv emer.Layer) {
	return AddInputPulv4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
}

// AddSuperCT2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddSuperCT2D(name string, shapeY, shapeX int, pulvLay bool) (super, ct, pulv emer.Layer) {
	return AddSuperCT2D(&nt.Network, name, shapeY, shapeX, pulvLay)
}

// AddSuperCT4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddSuperCT4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) (super, ct, pulv emer.Layer) {
	return AddSuperCT4D(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, pulvLay)
}

// AddSuperCTAttn adds a 4D superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPrjn Full projection from Super to CT, and a TRN attentional layer.
// CT is placed Behind Super, and Pulvinar behind CT if created.
func (nt *Network) AddSuperCTAttn(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) (super, ct, trn, pulv emer.Layer) {
	return AddSuperCTAttn(&nt.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, pulvLay)
}

// ConnectCtxtToCT adds a CTCtxtPrjn from given sending layer to a CT layer
func (nt *Network) ConnectCtxtToCT(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
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
			dl.CtxtFmGe(ltime)
		}
	}, "CtxtFmGe")
}
