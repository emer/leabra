// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// TRNLayer copies inhibition from pools in CT and TRC layers, and from other
// TRNLayers, and pools this inhibition using the Max operation
type TRNLayer struct {
	leabra.Layer

	// layers that we receive inhibition from
	ILayers emer.LayNames `desc:"layers that we receive inhibition from"`
}

var KiT_TRNLayer = kit.Types.AddType(&TRNLayer{}, leabra.LayerProps)

func (ly *TRNLayer) Defaults() {
	ly.Layer.Defaults()
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *TRNLayer) InitActs() {
	ly.Layer.InitActs()
}
