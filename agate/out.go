// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"log"

	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// OutParams determine the behavior of OutLayer
type OutParams struct {
	ResetThr  float32       `desc:"threshold on activation, above which the ClearLays will be reset"`
	ClearLays emer.LayNames `desc:"name of corresponding layers that are reset when this layer gets activated"`
}

func (np *OutParams) Defaults() {
	np.ResetThr = 0.5
}

// OutLayer is a frontal cortex output layer (L5 PM), which typically is interconnected
// with Ventral Thalamus (VM / VA etc) for output gating, and also NMDAPrjn maintenance.
type OutLayer struct {
	MaintLayer
	Out OutParams `desc:"Parameters for output layer function"`
}

var KiT_OutLayer = kit.Types.AddType(&OutLayer{}, leabra.LayerProps)

func (ly *OutLayer) Defaults() {
	ly.MaintLayer.Defaults()
	ly.Out.Defaults()
}

// ClearLays returns the Layers by name
func (ly *OutLayer) ClearLays() ([]PulseClearer, error) {
	var lays []PulseClearer
	var err error
	for _, nm := range ly.Out.ClearLays {
		var tly emer.Layer
		tly, err = ly.Network.LayerByNameTry(nm)
		if err != nil {
			log.Printf("OutLayer %s, ClearLay: %v\n", ly.Name(), err)
		}
		lays = append(lays, tly.(PulseClearer))
	}
	return lays, err
}

// CyclePost calls ResetMaint
func (ly *OutLayer) CyclePost(ltime *leabra.Time) {
	ly.MaintLayer.CyclePost(ltime)
	ly.PulseClear(ltime)
}

// PulseClear sends a simulated synchronous pulse of activation / inhibition
// to clear ClearLays
func (ly *OutLayer) PulseClear(ltime *leabra.Time) {
	if ltime.Cycle < ly.NMDA.AlphaMaxCyc {
		return
	}
	pl := ly.Pools[0]
	maxact := pl.Inhib.Act.Max
	if maxact > ly.Out.ResetThr {
		lays, _ := ly.ClearLays()
		for _, cly := range lays {
			cly.PulseClearNMDA()
		}
	}
}
