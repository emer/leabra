// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import "log"

// SeSrcLayer is the basic type of layer that sends Se to other layers.
// Uses a list of layer names to send to -- not use Prjn infrastructure
// as it is global broadcast modulator -- individual neurons
// can use it in their own special way.
type SeSrcLayer struct {
	ModLayer
	SendTo []string `desc:"list of layers to send Se to"`
}

// SendSe sends serotonin to SendTo list of layers
func (dl *SeSrcLayer) SendSe(net *Network, se float32) {
	for _, lnm := range dl.SendTo {
		ly, err := net.LayerByNameTry(lnm)
		if err != nil {
			log.Println(err)
			continue
		}
		ml := ly.(*ModLayer)
		ml.SE = se
	}
}

// AddSendTo adds given layer name to list of those to send DA to
func (dl *SeSrcLayer) AddSendTo(laynm string) {
	dl.SendTo = append(dl.SendTo, laynm)
}
