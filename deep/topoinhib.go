// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/chewxy/math32"
	"github.com/emer/emergent/evec"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/kit"
)

// TopoInhib provides for topographic gaussian inhibition integrating over neighborhood.
// Effective inhibition is
type TopoInhib struct {
	On    bool      `desc:"use topographic inhibition"`
	Width int       `desc:"half-width of topographic inhibition within layer"`
	Sigma float32   `desc:"normalized gaussian sigma as proportion of Width, for gaussian weighting"`
	Gi    float32   `desc:"overall inhibition multiplier for topographic inhibition (generally <= 1)"`
	LayGi float32   `desc:"layer-level baseline inhibition factor for Max computation -- ensures a baseline inhib as proportion of maximum inhib within any single pool"`
	Wts   []float32 `inactive:"+" desc:"gaussian weights as function of distance, precomputed.  index 0 = dist 1"`
}

func (ti *TopoInhib) Defaults() {
	ti.Width = 4
	ti.Sigma = 0.5
	ti.Gi = 10
	ti.LayGi = 0.5
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

// TopoInhibLayer is a layer with topographically organized inhibition among pools
type TopoInhibLayer struct {
	leabra.Layer           // access as .Layer
	TopoInhib    TopoInhib `desc:"topographic inhibition parameters for pool-level inhibition (only used for layers with pools)"`
}

var KiT_TopoInhibLayer = kit.Types.AddType(&TopoInhibLayer{}, LayerProps)

func (ly *TopoInhibLayer) Defaults() {
	ly.Layer.Defaults()
	ly.TopoInhib.Defaults()
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *TopoInhibLayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.TopoInhib.Update()
}

// TopoGiPos returns position-specific Gi contribution
func (ly *TopoInhibLayer) TopoGiPos(py, px, d int) float32 {
	pyn := ly.Shp.Dim(0)
	pxn := ly.Shp.Dim(1)
	if py < 0 || py >= pyn {
		return 0
	}
	if px < 0 || px >= pxn {
		return 0
	}
	pi := py*pxn + px
	pl := ly.Pools[pi+1]
	g := ly.TopoInhib.Wts[d]
	return g * pl.Inhib.GiOrig
}

// TopoGi computes topographic Gi between pools
func (ly *TopoInhibLayer) TopoGi(ltime *leabra.Time) {
	pyn := ly.Shp.Dim(0)
	pxn := ly.Shp.Dim(1)
	wd := ly.TopoInhib.Width

	laymax := float32(0)
	np := len(ly.Pools)
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		laymax = math32.Max(laymax, pl.Inhib.GiOrig)
	}

	laymax *= ly.TopoInhib.LayGi

	for py := 0; py < pyn; py++ {
		for px := 0; px < pxn; px++ {
			max := laymax
			for iy := 1; iy <= wd; iy++ {
				for ix := 1; ix <= wd; ix++ {
					max = math32.Max(max, ly.TopoGiPos(py+iy, px+ix, ints.MinInt(iy-1, ix-1)))
					max = math32.Max(max, ly.TopoGiPos(py-iy, px+ix, ints.MinInt(iy-1, ix-1)))
					max = math32.Max(max, ly.TopoGiPos(py+iy, px-ix, ints.MinInt(iy-1, ix-1)))
					max = math32.Max(max, ly.TopoGiPos(py-iy, px-ix, ints.MinInt(iy-1, ix-1)))
				}
			}
			pi := py*pxn + px
			pl := &ly.Pools[pi+1]
			pl.Inhib.Gi = math32.Max(max, pl.Inhib.Gi)
		}
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *TopoInhibLayer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime)
	if ly.Is4D() && ly.TopoInhib.On {
		ly.TopoGi(ltime)
	}
	ly.InhibFmPool(ltime)
}
