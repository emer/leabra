// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/leabra/v2/sims/ra25"
)

func main() { egui.Run[ra25.Sim, ra25.Config]() }
