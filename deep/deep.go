// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/leabra/leabra"
)

// DeepLayer defines the essential algorithmic API for DeepLeabra at the layer level.
type DeepLayer interface {
	leabra.LeabraLayer

	// AsDeep returns this layer as a deep.Layer -- provides direct access to variables
	AsDeep() *Layer

	// AttnGeInc integrates new AttnGe from increments sent during last SendGDelta.
	AttnGeInc(ltime *leabra.Time)

	// AvgMaxAttnGe computes the average and max AttnGe stats
	AvgMaxAttnGe(ltime *leabra.Time)

	// DeepAttnFmG computes DeepAttn and DeepLrn from AttnGe input,
	// and then applies the DeepAttn modulation to the Act activation value.
	DeepAttnFmG(ltime *leabra.Time)

	// AvgMaxActNoAttn computes the average and max ActNoAttn stats
	AvgMaxActNoAttn(ltime *leabra.Time)

	// BurstFmAct updates Burst layer 5 IB bursting value from current Act (superficial activation)
	// Subject to thresholding.
	BurstFmAct(ltime *leabra.Time)

	// SendTRCBurstGeDelta sends change in Burst activation since last sent, over BurstTRC
	// projections.
	SendTRCBurstGeDelta(ltime *leabra.Time)

	// TRCBurstGeFmInc computes the TRCBurstGe input from sent values
	TRCBurstGeFmInc(ltime *leabra.Time)

	// AvgMaxTRCBurstGe computes the average and max TRCBurstGe stats
	AvgMaxTRCBurstGe(ltime *leabra.Time)

	// SendCtxtGe sends full Burst activation over BurstCtxt projections to integrate
	// CtxtGe excitatory conductance on deep layers.
	// This must be called at the end of the Burst quarter for this layer.
	SendCtxtGe(ltime *leabra.Time)

	// CtxtFmGe integrates new CtxtGe excitatory conductance from projections, and computes
	// overall Ctxt value.  This must be called at the end of the Burst quarter for this layer,
	// after SendCtxtGe.
	CtxtFmGe(ltime *leabra.Time)

	// BurstPrv saves Burst as BurstPrv
	BurstPrv(ltime *leabra.Time)
}

// DeepPrjn defines the essential algorithmic API for DeepLeabra at the projection level.
type DeepPrjn interface {
	leabra.LeabraPrjn

	// SendCtxtGe sends the full Burst activation from sending neuron index si,
	// to integrate CtxtGe excitatory conductance on receivers
	SendCtxtGe(si int, dburst float32)

	// SendTRCBurstGeDelta sends the delta-Burst activation from sending neuron index si,
	// to integrate TRCBurstGe excitatory conductance on receivers
	SendTRCBurstGeDelta(si int, delta float32)

	// SendAttnGeDelta sends the delta-activation from sending neuron index si,
	// to integrate into AttnGeInc excitatory conductance on receivers
	SendAttnGeDelta(si int, delta float32)

	// RecvCtxtGeInc increments the receiver's CtxtGe from that of all the projections
	RecvCtxtGeInc()

	// RecvTRCBurstGeInc increments the receiver's TRCBurstGe from that of all the projections
	RecvTRCBurstGeInc()

	// RecvAttnGeInc increments the receiver's AttnGe from that of all the projections
	RecvAttnGeInc()

	// DWtDeepCtxt computes the weight change (learning) -- for DeepCtxt projections
	DWtDeepCtxt()
}
