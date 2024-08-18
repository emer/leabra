// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

// TRNLayer copies inhibition from pools in CT and TRC layers, and from other
// TRNLayers, and pools this inhibition using the Max operation
type TRNLayer struct {
	leabra.Layer

	// layers that we receive inhibition from
	ILayers emer.LayNames
}
