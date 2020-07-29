// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"log"

	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// OutParams determine the behavior of OutLayer
type OutParams struct {
	MaintLay string  `desc:"name of corresponding MaintLayer that is reset when this layer gets activated"`
	ResetThr float32 `desc:"threshold on activation, above which the MaintLay will be reset"`
}

func (np *OutParams) Defaults() {
	np.ResetThr = 0.5
	if np.MaintLay == "" {
		np.MaintLay = "PFCMnt"
	}
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

// MaintLay returns the MaintLay by name
func (ly *OutLayer) MaintLay() (*MaintLayer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.Out.MaintLay)
	if err != nil {
		log.Printf("OutLayer %s, MaintLay: %v\n", ly.Name(), err)
		return nil, err
	}
	return tly.(*MaintLayer), nil
}

// CyclePost calls ResetMaint
func (ly *OutLayer) CyclePost(ltime *leabra.Time) {
	ly.MaintLayer.CyclePost(ltime)
	ly.ResetMaint(ltime)
}

// ResetMaint resets the maintenance layer if activation is above threshold
func (ly *OutLayer) ResetMaint(ltime *leabra.Time) {
	// todo: not sure if should be sub-pool or whole layer?

	pl := ly.Pools[0]
	maxact := pl.Inhib.Act.Max
	if maxact > ly.Out.ResetThr {
		mlay, err := ly.MaintLay()
		if err == nil {
			mlay.InitGlong() // note: will continue to reset..
		}
	}
}
