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

// GFmInc gets activity from pools instead of projections
func (ly *TRNLayer) GFmInc(ltime *leabra.Time) {
	ly.GeFmEPools()
}

// GeFmEPools receives Ge excitation from EPools Act.Avg activity
func (ly *TRNLayer) GeFmEPools() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.GeRaw = 0
	}
	for _, ep := range ly.EPools {
		oli := ly.Network.LayerByName(ep.LayNm)
		if oli == nil {
			continue
		}
		ol := oli.(leabra.LeabraLayer).AsLeabra()
		for ni := range ly.Neurons { // our neurons map to their pools
			nrn := &ly.Neurons[ni]
			opl := ol.Pools[1+ni]
			nrn.GeRaw += ep.Wt * opl.Inhib.Act.Avg
			ly.Act.GeFmRaw(nrn, nrn.GeRaw)
		}
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *TRNLayer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	// todo: spread the inhibition..
	// mxact := ly.InterInhibMaxAct(ltime)
	// lpl.Inhib.Act.Avg = math32.Max(ly.InterInhib.Gi*mxact, lpl.Inhib.Act.Avg)
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime) // this one does GABA-B
}

// GiFmIPools receives Ge excitation from IPools Act.Avg activity
// func (ly *TRNLayer) GiFmIPools() {
// 	for ni := range ly.Neurons {
// 		nrn := &ly.Neurons[ni]
// 		nrn.GeRaw = 0
// 	}
// 	for _, ep := range ly.EPools {
// 		oli := ly.Network.LayerByName(ep.LayNm)
// 		if oli == nil {
// 			continue
// 		}
// 		ol := oli.(leabra.LeabraLayer).AsLeabra()
// 		for ni := range ly.Neurons { // our neurons map to their pools
// 			nrn := &ly.Neurons[ni]
// 			opl := ol.Pools[1+ni]
// 			nrn.GeRaw += ep.Wt * opl.Inhib.Act.Avg
// 			ly.Act.GeFmRaw(nrn, nrn.GeRaw)
// 		}
// 	}
// }

func (ly *TRNLayer) CyclePost(ltime *leabra.Time) {
	ly.SendAttn()
}

// SendAttn sends a normalized version of this layer's activity as attentional modulator
// to corresponding AttnLayers listed in SendTo
func (ly *TRNLayer) SendAttn() {
	lpl := &ly.Pools[0]
	amax := lpl.Inhib.Act.Max
	for _, lnm := range ly.SendTo {
		oli := ly.Network.LayerByName(lnm)
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
}
