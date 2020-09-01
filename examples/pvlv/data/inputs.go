// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package data

import "github.com/emer/etable/etensor"

type Inputs interface {
	Empty() bool
	FromString(s string) Inputs
	OneHot() int
	Tensor() etensor.Tensor
	TensorScaled(scale float32) etensor.Tensor
}
