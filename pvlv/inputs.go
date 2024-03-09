// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import "github.com/emer/etable/v2/etensor"

type Inputs interface {
	Empty() bool
	FromString(s string) Inputs
	OneHot() int
	Tensor() etensor.Tensor
	TensorScaled(scale float32) etensor.Tensor
}
