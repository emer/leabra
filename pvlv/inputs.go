// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import "github.com/emer/table/v2/tensor"

type Inputs interface {
	Empty() bool
	FromString(s string) Inputs
	OneHot() int
	Tensor() tensor.Tensor
	TensorScaled(scale float32) tensor.Tensor
}
