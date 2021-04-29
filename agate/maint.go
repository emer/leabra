// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/leabra/glong"
	"github.com/emer/leabra/interinhib"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// PulseClearParams are parameters for the synchronous pulse of activation /
// inhibition that clears NMDA maintenance.
type PulseClearParams struct {
	GABAB float32 `desc:"GABAB value activated by the inhibitory pulse"`
}

func (pc *PulseClearParams) Defaults() {
	pc.GABAB = 2
}

///////////////////////////////////////////////////////////////////////////
// MaintLayer is a layer with NMDA channels that supports active maintenance
// in frontal cortex, via NMDA channels (in an NMDAMaintPrjn).
type MaintLayer struct {
	glong.Layer
	PulseClear PulseClearParams      `desc:"parameters for the synchronous pulse of activation / inhibition that clears NMDA maintenance."`
	InterInhib interinhib.InterInhib `desc:"inhibition from output layer"`
}

var KiT_MaintLayer = kit.Types.AddType(&MaintLayer{}, leabra.LayerProps)

func (ly *MaintLayer) Defaults() {
	ly.Layer.Defaults()
	ly.NMDA.Gbar = 0.02
	ly.PulseClear.Defaults()
	ly.InterInhib.Defaults()
	ly.InterInhib.Gi = 0.1
	ly.InterInhib.Add = true
	ly.Act.Init.Decay = 0
	ly.Inhib.Pool.On = true
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *MaintLayer) InhibFmGeAct(ltime *leabra.Time) {
	lpl := &ly.Pools[0]
	mxact := ly.InterInhibMaxAct(ltime)
	lpl.Inhib.Act.Avg = mat32.Max(ly.InterInhib.Gi*mxact, lpl.Inhib.Act.Avg)
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime)
	ly.InhibFmPool(ltime)
}

// InterInhibMaxAct returns the AlphaMax activation for source layers
func (ly *MaintLayer) InterInhibMaxAct(ltime *leabra.Time) float32 {
	mxact := float32(0)
	for _, lnm := range ly.InterInhib.Lays {
		oli := ly.Network.LayerByName(lnm)
		if oli == nil {
			continue
		}
		ol, ok := oli.(*OutLayer)
		if ok {
			mxact = ol.MaxAlphaMax()
		}
		// todo: anything else?
	}
	return mxact
}

// PulseClearNMDA simulates a synchronous pulse of activation that
// clears the NMDA and puts the layer into a refractory state by
// activating the GABAB currents.
func (ly *MaintLayer) PulseClearNMDA() {
	for ni := range ly.GlNeurs {
		nrn := &ly.Neurons[ni]
		nrn.Act = ly.Act.Init.Act
		nrn.ActLrn = nrn.Act
		nrn.Ge = ly.Act.Init.Ge
		nrn.GeRaw = 0
		nrn.Vm = ly.Act.Init.Vm

		gnr := &ly.GlNeurs[ni]
		gnr.VmEff = nrn.Vm
		gnr.Gnmda = 0
		gnr.NMDA = 0
		gnr.NMDASyn = 0
		gnr.GABAB = ly.PulseClear.GABAB
		gnr.GABABx = gnr.GABAB
	}
}

// PulseClearer is an interface for Layers that have the
// PulseClearNMDA method for clearing NMDA and activating
// GABAB refractory inhibition
type PulseClearer interface {
	leabra.LeabraLayer

	// PulseClearNMDA simulates a synchronous pulse of activation that
	// clears the NMDA and puts the layer into a refractory state by
	// activating the GABAB currents.
	PulseClearNMDA()
}
