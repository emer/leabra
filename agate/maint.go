// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/chewxy/math32"
	"github.com/emer/leabra/glong"
	"github.com/emer/leabra/interinhib"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

///////////////////////////////////////////////////////////////////////////
// MaintLayer

// MaintLayer is a layer with NMDA channels that supports active maintenance
// in frontal cortex, via NMDA channels (in an NMDAMaintPrjn).
type MaintLayer struct {
	glong.Layer
	InterInhib interinhib.InterInhib `desc:"inhibition from output layer"`
}

var KiT_MaintLayer = kit.Types.AddType(&MaintLayer{}, leabra.LayerProps)

func (ly *MaintLayer) Defaults() {
	ly.Layer.Defaults()
	ly.NMDA.Gbar = 0.02
	ly.InterInhib.Defaults()
	ly.InterInhib.Gi = 0.1
	ly.InterInhib.Add = true
	ly.Act.Init.Decay = 0
	ly.Inhib.Pool.On = true
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *MaintLayer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	mxact := ly.InterInhibMaxAct(ltime)
	lpl.Inhib.Act.Avg = math32.Max(ly.InterInhib.Gi*mxact, lpl.Inhib.Act.Avg)
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime) // this one does GABA-B
}

// InterInhibMaxAct returns the AlphaMax activation for source layers
func (ly *MaintLayer) InterInhibMaxAct(ltime *leabra.Time) float32 {
	mxact := float32(0)
	for _, lnm := range ly.InterInhib.Lays {
		oli := ly.Network.LayerByName(lnm)
		if oli == nil {
			continue
		}
		ol, ok := oli.(*OutLayer)
		if ok {
			mxact = ol.MaxAlphaMax()
		}
		// todo: anything else?
	}
	return mxact
}
