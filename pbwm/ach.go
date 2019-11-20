// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"log"

	"github.com/emer/leabra/deep"
	"github.com/goki/ki/kit"
)

// AChSrcLayer is the basic type of layer that sends ACh to other layers.
// Uses a list of layer names to send to -- not use Prjn infrastructure
// as it is global broadcast modulator -- individual neurons
// can use it in their own special way.
type AChSrcLayer struct {
	ModLayer
	SendTo []string `desc:"list of layers to send ACh to"`
}

var KiT_AChSrcLayer = kit.Types.AddType(&AChSrcLayer{}, deep.LayerProps)

// SendToCheck is called during Build to ensure that SendTo layers are valid
func (ly *AChSrcLayer) SendToCheck() error {
	var lasterr error
	for _, lnm := range ly.SendTo {
		ly, err := ly.Network.LayerByNameTry(lnm)
		if err != nil {
			log.Printf("AChSrcLayer %s SendToCheck: %v\n", ly.Name(), err)
			lasterr = err
		}
	}
	return lasterr
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *AChSrcLayer) Build() error {
	err := ly.ModLayer.Build()
	if err != nil {
		return err
	}
	err = ly.SendToCheck()
	return err
}

// SendACh sends ACh to SendTo list of layers
func (ly *AChSrcLayer) SendACh(ach float32) {
	for _, lnm := range ly.SendTo {
		ml := ly.Network.LayerByName(lnm).(PBWMLayer).AsMod()
		ml.ACh = ach
	}
}

// AddSendTo adds given layer name to list of those to send DA to
func (ly *AChSrcLayer) AddSendTo(laynm string) {
	ly.SendTo = append(ly.SendTo, laynm)
}
