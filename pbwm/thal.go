// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import "github.com/emer/etable/minmax"

// ThalGateState is thalamic gating state values stored in layers that receive thalamic gating signals
// including MatrixLayer, PFCLayer, GPiThal layer, etc
type ThalGateState struct {
	Act   float32         `desc:"thalamic activation value, reflecting current thalamic gating layer activation and sent back to corresponding Matrix and PFC layers"`
	Gate  float32         `desc:"discrete thalamic gating signal -- typically activates to 1 when thalamic pathway gates, and is 0 otherwise -- PFC and BG layers receive this signal to drive updating etc at the proper time -- other layers can use the LeabraNetwork times.thal_gate_cycle signal"`
	Cnt   int             `desc:"counter for thalamic activation value -- increments for active maintenance in PFCUnitSpec"`
	GeRaw minmax.AvgMax32 `desc:"average and max Ge excitatory conductance values, which drive FF inhibition"`
}

// Init initializes the values -- call during InitActs()
func (tg *ThalGateState) Init() {
	tg.Act = 0
	tg.Gate = 0
	tg.Cnt = -1
	tg.GeRaw.Init()
}
