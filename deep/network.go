// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// deep.Network has parameters for running a DeepLeabra network
type Network struct {
	leabra.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = leabra.NetworkProps

// NewLayer returns new layer of proper type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPrjn returns new prjn of proper type
func (nt *Network) NewPrjn() emer.Prjn {
	return &Prjn{}
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

// AddInputPulv2D adds an input and corresponding Pulvinar (P suffix) layer
// with BurstTRC one-to-one projection from Input to Pulvinar.
// Pulvinar is placed Behind Input.
func (nt *Network) AddInputPulv2D(name string, shapeY, shapeX int) (input, pulv emer.Layer) {
	input = nt.AddLayer2D(name, shapeY, shapeX, emer.Input)
	pulv = nt.AddLayer2D(name+"P", shapeY, shapeX, TRC)
	pulv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(input, pulv, prjn.NewOneToOne(), BurstTRC)
	return
}

// AddInputPulv4D adds an input and corresponding Pulvinar (P suffix) layer
// with BurstTRC one-to-one projection from Input to Pulvinar.
// Pulvinar is placed Behind Input.
func (nt *Network) AddInputPulv4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (input, pulv emer.Layer) {
	input = nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, emer.Input)
	pulv = nt.AddLayer4D(name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX, TRC)
	pulv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(input, pulv, prjn.NewOneToOne(), BurstTRC)
	return
}

// bool args for greater clarity
const (
	AddPulv    bool = true
	NoPulv          = false
	AttnPrjn        = true
	NoAttnPrjn      = false
)

// AddSuperDeep2D adds a superficial (hidden) and corresponding Deep (D suffix) layer
// with BurstCtxt Full projection from Hidden to Deep.  Optionally
// creates a Pulvinar for Hidden with One-to-One BurstTRC to Pulvinar, and
// optionally a DeepAttn projection back from Deep to Super (OneToOne).
// Deep is placed Behind Super, and Pulvinar behind Deep if created.
func (nt *Network) AddSuperDeep2D(name string, shapeY, shapeX int, pulvLay, attn bool) (super, deep, pulv emer.Layer) {
	super = nt.AddLayer2D(name, shapeY, shapeX, emer.Hidden)
	deep = nt.AddLayer2D(name+"D", shapeY, shapeX, Deep)
	deep.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(super, deep, prjn.NewFull(), BurstCtxt)
	if attn {
		nt.ConnectLayers(deep, super, prjn.NewOneToOne(), DeepAttn)
	}
	if pulvLay {
		pulv = nt.AddLayer2D(name+"P", shapeY, shapeX, TRC)
		pulv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "D", XAlign: relpos.Left, Space: 2})
		nt.ConnectLayers(super, pulv, prjn.NewOneToOne(), BurstTRC)
	}
	return
}

// AddSuperDeep4D adds a superficial (hidden) and corresponding Deep (D suffix) layer
// with BurstCtxt Full projection from Hidden to Deep.  Optionally
// creates a Pulvinar for Hidden with One-to-One BurstTRC to Pulvinar, and
// optionally a DeepAttn projection back from Deep to Super (OneToOne)
// Deep is placed Behind Super, and Pulvinar behind Deep if created.
func (nt *Network) AddSuperDeep4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay, attn bool) (super, deep, pulv emer.Layer) {
	super = nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, emer.Hidden)
	deep = nt.AddLayer4D(name+"D", nPoolsY, nPoolsX, nNeurY, nNeurX, Deep)
	deep.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	nt.ConnectLayers(super, deep, prjn.NewFull(), BurstCtxt)
	if attn {
		nt.ConnectLayers(deep, super, prjn.NewOneToOne(), DeepAttn)
	}
	if pulvLay {
		pulv = nt.AddLayer4D(name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX, TRC)
		pulv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name + "D", XAlign: relpos.Left, Space: 2})
		nt.ConnectLayers(super, pulv, prjn.NewOneToOne(), BurstTRC)
	}
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// Cycle runs one cycle of activation updating
// Deep version adds call to update DeepBurst at end
func (nt *Network) Cycle(ltime *leabra.Time) {
	nt.Network.Cycle(ltime)
	nt.DeepBurst(ltime)
}

// DeepBurst is called at end of Cycle, computes Burst and sends it to other layers
func (nt *Network) DeepBurst(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(DeepLayer).BurstFmAct(ltime) }, "BurstFmAct")
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(DeepLayer).SendTRCBurstGeDelta(ltime) }, "SendTRCBurstGeDelta")
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(DeepLayer).TRCBurstGeFmInc(ltime) }, "TRCBurstGeFmInc")
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(DeepLayer).AvgMaxTRCBurstGe(ltime) }, "AvgMaxTRCBurstGe")
}

// QuarterFinal does updating after end of a quarter
func (nt *Network) QuarterFinal(ltime *leabra.Time) {
	nt.Network.QuarterFinal(ltime)
	nt.DeepCtxt(ltime)
}

// DeepCtxt sends DeepBurst to Deep layers and integrates DeepCtxt on Deep layers
func (nt *Network) DeepCtxt(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(DeepLayer).SendCtxtGe(ltime) }, "SendCtxtGe")
	nt.ThrLayFun(func(ly leabra.LeabraLayer) { ly.(DeepLayer).CtxtFmGe(ltime) }, "CtxtFmGe")
}
