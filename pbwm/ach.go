// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import "log"

// AChSrcLayer is the basic type of layer that sends ACh to other layers.
// Uses a list of layer names to send to -- not use Prjn infrastructure
// as it is global broadcast modulator -- individual neurons
// can use it in their own special way.
type AChSrcLayer struct {
	ModLayer
	SendTo []string `desc:"list of layers to send ACh to"`
}

// SendACh sends ACh to SendTo list of layers
func (dl *AChSrcLayer) SendACh(net *Network, ach float32) {
	for _, lnm := range dl.SendTo {
		ly, err := net.LayerByNameTry(lnm)
		if err != nil {
			log.Println(err)
			continue
		}
		ml := ly.(*ModLayer)
		ml.ACh = ach
	}
}

// AddSendTo adds given layer name to list of those to send DA to
func (dl *AChSrcLayer) AddSendTo(laynm string) {
	dl.SendTo = append(dl.SendTo, laynm)
}
