// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attrn

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/evec"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/kit"
)

// TopoInhib provides for topographic gaussian inhibition integrating over neighborhood
type TopoInhib struct {
	On    bool      `desc:"use topographic inhibition"`
	Width int       `desc:"half-width of topographic inhibition within layer"`
	Sigma float32   `desc:"normalized gaussian sigma as proportion of Width, for gaussian weighting"`
	Gi    float32   `desc:"inhibition multiplier for topographic inhibition"`
	Wts   []float32 `desc:"gaussian weights as function of distance, precomputed.  index 0 = dist 1"`
}

func (ti *TopoInhib) Defaults() {
	ti.Width = 4
	ti.Sigma = 0.2
	ti.Gi = 10
	ti.Update()
}

func (ti *TopoInhib) Update() {
	if len(ti.Wts) != ti.Width {
		ti.Wts = make([]float32, ti.Width)
	}
	sig := float32(ti.Width) * ti.Sigma
	for i := range ti.Wts {
		ti.Wts[i] = ti.Gi * evec.Gauss1DNoNorm(float32(i+1), sig)
	}
}

// TRNLayer receives excitation based on the average activation in CT and TRC layers,
// and distributes inhibition within the layer according to a gaussian distribution,
// and across other such TRN layers.
// Unlike the real TRN, this layer sends a multiplicative Attn factor to other layers
// representing the activation relative to the pooled TRN inhibition.
type TRNLayer struct {
	leabra.Layer
	Topo   TopoInhib     `desc:"topographic inhibition parameters"`
	EPools EPools        `desc:"pools that we get excitation from, as the pool-level average activation"`
	IPools IPools        `desc:"pools in other TRN layers that we get inhibition from"`
	SendTo emer.LayNames `desc:"layers that we send attention to"`
}

var KiT_TRNLayer = kit.Types.AddType(&TRNLayer{}, leabra.LayerProps)

func (ly *TRNLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Topo.Defaults()
	ly.Inhib.Layer.FB = 0 // TRN does not plausibly have FB inhibition
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *TRNLayer) InitActs() {
	ly.Layer.InitActs()
	ly.Topo.Update() // ensure recomputed
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
		}
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Act.GeFmRaw(nrn, nrn.GeRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *TRNLayer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime)
	if ly.Topo.On {
		ly.TopoGi(ltime)
	}
}

// TopoGiPos returns position-specific Gi contribution
func (ly *TRNLayer) TopoGiPos(ny, nx, y, x, d int, geavg float32) float32 {
	g := ly.Topo.Wts[d]
	if y < 0 || y >= ny {
		return g * geavg
	}
	if x < 0 || x >= nx {
		return g * geavg
	}
	ni := y*nx + x
	return g * ly.Neurons[ni].Ge
}

// TopoGi computes topographic Gi
func (ly *TRNLayer) TopoGi(ltime *leabra.Time) {
	ny := ly.Shp.Dim(0)
	nx := ly.Shp.Dim(1)
	lpl := &ly.Pools[0]
	geavg := lpl.Inhib.Ge.Avg
	gin := float32((ly.Topo.Width*2)*(ly.Topo.Width*2) + 1)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		py := ni / nx
		px := ni % nx
		gi := nrn.Ge
		for iy := 1; iy <= ly.Topo.Width; iy++ {
			for ix := 1; ix <= ly.Topo.Width; ix++ {
				gi += ly.TopoGiPos(ny, nx, py+iy, px+ix, ints.MinInt(iy-1, ix-1), geavg)
				gi += ly.TopoGiPos(ny, nx, py-iy, px+ix, ints.MinInt(iy-1, ix-1), geavg)
				gi += ly.TopoGiPos(ny, nx, py+iy, px-ix, ints.MinInt(iy-1, ix-1), geavg)
				gi += ly.TopoGiPos(ny, nx, py-iy, px-ix, ints.MinInt(iy-1, ix-1), geavg)
			}
		}
		ly.Neurons[ni].Gi += gi / float32(gin)
	}
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
