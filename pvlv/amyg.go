// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"github.com/emer/leabra/leabra"
)

// Common functionality for both BL and CEl amygdala
// More of a mixin than a complete Layer
type AmygdalaLayer struct {
	ModLayer
}

func (ly *AmygdalaLayer) ModsFmInc(_ *leabra.Time) {
	ly.SetModLevels()
}
