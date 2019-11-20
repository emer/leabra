// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"log"

	"github.com/emer/leabra/deep"
	"github.com/goki/ki/kit"
)

// SeSrcLayer is the basic type of layer that sends Se to other layers.
// Uses a list of layer names to send to -- not use Prjn infrastructure
// as it is global broadcast modulator -- individual neurons
// can use it in their own special way.
type SeSrcLayer struct {
	ModLayer
	SendTo []string `desc:"list of layers to send Se to"`
}

var KiT_SeSrcLayer = kit.Types.AddType(&SeSrcLayer{}, deep.LayerProps)

// SendToCheck is called during Build to ensure that SendTo layers are valid
func (ly *SeSrcLayer) SendToCheck() error {
	var lasterr error
	for _, lnm := range ly.SendTo {
		ly, err := ly.Network.LayerByNameTry(lnm)
		if err != nil {
			log.Printf("SeSrcLayer %s SendToCheck: %v\n", ly.Name(), err)
			lasterr = err
		}
	}
	return lasterr
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *SeSrcLayer) Build() error {
	err := ly.ModLayer.Build()
	if err != nil {
		return err
	}
	err = ly.SendToCheck()
	return err
}

// SendSe sends serotonin to SendTo list of layers
func (ly *SeSrcLayer) SendSe(se float32) {
	for _, lnm := range ly.SendTo {
		ml := ly.Network.LayerByName(lnm).(PBWMLayer).AsMod()
		ml.SE = se
	}
}

// AddSendTo adds given layer name to list of those to send DA to
func (ly *SeSrcLayer) AddSendTo(laynm string) {
	ly.SendTo = append(ly.SendTo, laynm)
}
