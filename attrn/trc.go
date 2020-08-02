// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attrn

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// TRNLayer receives excitation based on the average activation in CT and TRC layers,
// and integrates inhibition across other such TRN layers
// and sends
type TRNLayer struct {
	leabra.Layer
	EPools EPools        `desc:"pools that we get excitation from, as the pool-level average activation"`
	IPools IPools        `desc:"pools in other TRN layers that we get inhibition from"`
	SendTo emer.LayNames `desc:"layers that we send attention to"`
}

var KiT_TRNLayer = kit.Types.AddType(&TRNLayer{}, leabra.LayerProps)

func (ly *TRNLayer) Defaults() {
	ly.Layer.Defaults()
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *TRNLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	ly.GFmIncNeur(ltime)
}

// SendAttn sends a normalized version of this layer's activity as attentional modulator
// to corresponding AttnLayers listed in SendTo
func (ly *TRNLayer) SendAttn(net emer.Network) {
	lpl := &ly.Pools[0]
	amax := lpl.Inhib.Act.Max
	for _, lnm := range ly.SendTo {
		oli := net.LayerByName(lnm)
		if oli == nil {
			continue
		}
		ol, ok := oli.(AttnSetLayer)
		if !ok {
			continue
		}
		for ni := range ly.Neurons { // our neurons map to their pools
			nrn := &ly.Neurons[ni]
			attn := nrn.Act / amax
			ol.SetAttn(ni, attn)
		}
	}
	return gi
}

// EFmPools receives excitation from pools
func (ly *TRNLayer) EFmPools(net emer.Network) {
	for _, ep := range ly.EPools {
		oli := net.LayerByName(ep.LayNm)
		if oli == nil {
			continue
		}
		ol := oli.(leabra.LeabraLayer).AsLeabra()
		for ni := range ly.Neurons { // our neurons map to their pools
			nrn := &ly.Neurons[ni]
			opl := ol.Pools[1+ni]
			nrn.GeRaw += ep.Wt * opl.Inhib.Act.Avg
		}
	}
}
