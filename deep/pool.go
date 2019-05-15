// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import "github.com/emer/etable/minmax"

// deep.Pool contains extra statistics used in DeepLeabra
type Pool struct {
	ActNoAttn  minmax.AvgMax32
	TRCBurstGe minmax.AvgMax32
	AttnGe     minmax.AvgMax32
}
