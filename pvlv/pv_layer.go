// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"strconv"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/leabra/v2/leabra"
)

// Primary Value input layer. Sends activation directly to its receivers, bypassing the standard mechanisms.
type PVLayer struct {
	leabra.Layer
	Net           *Network
	SendPVQuarter leabra.Quarters
	PVReceivers   emer.LayNames
}

func AddPVLayer(nt *Network, name string, nY, nX int, typ emer.LayerType) *PVLayer {
	ly := PVLayer{Net: nt}
	nt.AddLayerInit(&ly, name, []int{nY, nX, 1, 1}, typ)
	return &ly
}

func (ly *PVLayer) AddPVReceiver(lyNm string) {
	ly.PVReceivers.Add(lyNm)
	rly := ly.Network.LayerByName(lyNm).(IModLayer).AsMod()
	rly.IsPVReceiver = true
}

func (ly *PVLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.SendPVQuarter = leabra.Q4
	return nil
}

func (ly *PVLayer) SendPVAct() {
	for li := range ly.PVReceivers {
		rly := ly.Net.LayerByName(ly.PVReceivers[li]).(IModLayer).AsMod()
		for pi := range ly.Neurons {
			pnr := &ly.Neurons[pi] // WARNING: both layers must have the same shape!
			mnr := &rly.ModNeurs[pi]
			mnr.PVAct = math32.Max(pnr.Act, pnr.Ext)
		}
	}
}

func (ly *PVLayer) CyclePost(ltime *leabra.Time) {
	if ltime.Quarter == ly.SendPVQuarter {
		ly.SendPVAct()
	}
}

func (ly *PVLayer) GetMonitorValue(data []string) float64 {
	var val float32
	valType := data[0]
	unitIndex, _ := strconv.Atoi(data[1])
	switch valType {
	case "TotalAct":
		val = TotalAct(ly)
	case "Act":
		val = ly.Neurons[unitIndex].Act
	case "PoolActAvg":
		pl := &ly.Pools[unitIndex].Inhib.Act
		val = pl.Avg * float32(pl.N)
	case "PoolActMax":
		pl := &ly.Pools[unitIndex].Inhib.Act
		val = pl.Max * float32(pl.N)
	}
	return float64(val)
}
