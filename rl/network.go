// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/leabra/v2/leabra"
)

// NOTE: rl layers are designed to be "mix-ins" with other networks so there is no
// RL network type -- just routines to add layers of different types.

// AddClampDaLayer adds a ClampDaLayer of given name
func AddClampDaLayer(nt *leabra.Network, name string) *ClampDaLayer {
	da := &ClampDaLayer{}
	nt.AddLayerInit(da, name, []int{1, 1}, emer.Input)
	return da
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func AddTDLayers(nt *leabra.Network, prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td leabra.LeabraLayer) {
	rew = nt.AddLayer2D(prefix+"Rew", 1, 1, emer.Input).(leabra.LeabraLayer)
	rp = &TDRewPredLayer{}
	nt.AddLayerInit(rp, prefix+"RewPred", []int{1, 1}, emer.Hidden)
	ri = &TDRewIntegLayer{}
	nt.AddLayerInit(ri, prefix+"RewInteg", []int{1, 1}, emer.Hidden)
	td = &TDDaLayer{}
	nt.AddLayerInit(td, prefix+"TD", []int{1, 1}, emer.Hidden)
	ri.(*TDRewIntegLayer).RewInteg.RewPred = rp.Name()
	td.(*TDDaLayer).RewInteg = ri.Name()
	rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	ri.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})
	td.SetRelPos(relpos.Rel{Rel: rel, Other: ri.Name(), YAlign: relpos.Front, Space: space})

	pj := nt.ConnectLayers(rew, ri, prjn.NewFull(), emer.Forward).(leabra.LeabraPrjn).AsLeabra()
	pj.SetClass("TDRewToInteg")
	pj.Learn.Learn = false
	pj.WtInit.Mean = 1
	pj.WtInit.Var = 0
	pj.WtInit.Sym = false
	// {Sel: ".TDRewToInteg", Desc: "rew to integ",
	// 	Params: params.Params{
	// 		"Prjn.Learn.Learn": "false",
	// 		"Prjn.WtInit.Mean": "1",
	// 		"Prjn.WtInit.Var":  "0",
	// 		"Prjn.WtInit.Sym":  "false",
	// 	}},
	return
}

// AddRWLayers adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
func AddRWLayers(nt *leabra.Network, prefix string, rel relpos.Relations, space float32) (rew, rp, da leabra.LeabraLayer) {
	rew = nt.AddLayer2D(prefix+"Rew", 1, 1, emer.Input).(leabra.LeabraLayer)
	rp = &RWPredLayer{}
	nt.AddLayerInit(rp, prefix+"RWPred", []int{1, 1}, emer.Hidden)
	da = &RWDaLayer{}
	nt.AddLayerInit(da, prefix+"DA", []int{1, 1}, emer.Hidden)
	da.(*RWDaLayer).RewLay = rew.Name()
	da.(*RWDaLayer).RewLay = rew.Name()
	rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	da.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})

	return
}

// AddTDLayersPy adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
// Py is Python version, returns layers as a slice
func AddTDLayersPy(nt *leabra.Network, prefix string, rel relpos.Relations, space float32) []leabra.LeabraLayer {
	rew, rp, ri, td := AddTDLayers(nt, prefix, rel, space)
	return []leabra.LeabraLayer{rew, rp, ri, td}
}

// AddRWLayersPy adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
// Py is Python version, returns layers as a slice
func AddRWLayersPy(nt *leabra.Network, prefix string, rel relpos.Relations, space float32) []leabra.LeabraLayer {
	rew, rp, da := AddRWLayers(nt, prefix, rel, space)
	return []leabra.LeabraLayer{rew, rp, da}
}
