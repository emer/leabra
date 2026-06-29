// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ra25

import (
	"github.com/emer/leabra/v2/leabra"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = leabra.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "all defaults",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.Layer.Gi = 1.8
				ly.Act.Init.Decay = 0.0
				ly.Act.Gbar.L = 0.1 // set explictly, new default, a bit better vs 0.2
			}},
		{Sel: "#Output", Doc: "",
			Set: func(ly *leabra.LayerParams) {
				ly.Inhib.Layer.Gi = 1.4
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = leabra.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "basic path params",
			Set: func(pt *leabra.PathParams) {
				pt.Learn.Norm.On = true
				pt.Learn.Momentum.On = true
				pt.Learn.WtBal.On = true // no diff really
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *leabra.PathParams) {
				pt.WtScale.Rel = 0.2
			}},
	},
}
