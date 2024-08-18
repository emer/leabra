// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package interinhib provides inter-layer inhibition params,
which can be added to Layer types.  Call at the start of the
Layer InhibFromGeAct method like this:

// InhibFromGeAct computes inhibition Gi from Ge and Act averages within relevant Pools

	func (ly *Layer) InhibFromGeAct(ctx *Time) {
		lpl := &ly.Pools[0]
		ly.Inhib.Layer.Inhib(&lpl.Inhib)
		ly.InterInhib.Inhib(&ly.Layer) // does inter-layer inhibition
		ly.PoolInhibFromGeAct(ctx)
	}
*/
package interinhib

import (
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

// InterInhib specifies inhibition between layers, where
// the receiving layer either does a Max or Add of portion of
// inhibition from other layer(s).
type InterInhib struct {

	// layers to receive inhibition from
	Lays emer.LayNames

	// multiplier on Gi from other layers
	Gi float32

	// add inhibition -- otherwise Max
	Add bool
}

func (il *InterInhib) Defaults() {
	il.Gi = 0.5
}

// Inhib updates layer inhibition based on other layer inhibition
func (il *InterInhib) Inhib(ly *leabra.Layer) {
	ogi := il.Gi * il.OtherGi(ly.Network)
	lpl := &ly.Pools[0]
	if il.Add {
		lpl.Inhib.Gi += ogi
	} else {
		lpl.Inhib.Gi = math32.Max(ogi, lpl.Inhib.Gi)
	}
}

// OtherGi returns either the Sum (for Add) or Max of other layer Gi values.
// These are the raw values, not multiplied by Gi factor.
func (il *InterInhib) OtherGi(net emer.Network) float32 {
	gi := float32(0)
	for _, lnm := range il.Lays {
		oli := net.LayerByName(lnm)
		if oli == nil {
			continue
		}
		ol := oli.(leabra.LeabraLayer).AsLeabra()
		ogi := ol.Pools[0].Inhib.GiOrig
		if il.Add {
			gi += ogi
		} else {
			gi = math32.Max(gi, ogi)
		}
	}
	return gi
}
