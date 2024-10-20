// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/emer/emergent/v2/paths"
)

// AddSuperLayer2D adds a SuperLayer of given size, with given name.
func (nt *Network) AddSuperLayer2D(name string, nNeurY, nNeurX int) *Layer {
	return nt.AddLayer2D(name, nNeurY, nNeurX, SuperLayer)
}

// AddSuperLayer4D adds a SuperLayer of given size, with given name.
func (nt *Network) AddSuperLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	return nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, SuperLayer)
}

// AddCTLayer2D adds a CTLayer of given size, with given name.
func (nt *Network) AddCTLayer2D(name string, nNeurY, nNeurX int) *Layer {
	return nt.AddLayer2D(name, nNeurY, nNeurX, CTLayer)
}

// AddCTLayer4D adds a CTLayer of given size, with given name.
func (nt *Network) AddCTLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	return nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, CTLayer)
}

// AddPulvinarLayer2D adds a PulvinarLayer of given size, with given name.
func (nt *Network) AddPulvinarLayer2D(name string, nNeurY, nNeurX int) *Layer {
	return nt.AddLayer2D(name, nNeurY, nNeurX, PulvinarLayer)
}

// AddPulvinarLayer4D adds a PulvinarLayer of given size, with given name.
func (nt *Network) AddPulvinarLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	return nt.AddLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX, PulvinarLayer)
}

// ConnectSuperToCT adds a CTCtxtPath from given sending Super layer to a CT layer
// This automatically sets the FromSuper flag to engage proper defaults,
// uses a OneToOne path pattern, and sets the class to CTFromSuper
func (nt *Network) ConnectSuperToCT(send, recv *Layer) *Path {
	pt := nt.ConnectLayers(send, recv, paths.NewOneToOne(), CTCtxtPath)
	pt.AddClass("CTFromSuper")
	pt.FromSuper = true
	return pt
}

// ConnectCtxtToCT adds a CTCtxtPath from given sending layer to a CT layer
// Use ConnectSuperToCT for main pathway from corresponding superficial layer.
func (nt *Network) ConnectCtxtToCT(send, recv *Layer, pat paths.Pattern) *Path {
	return nt.ConnectLayers(send, recv, pat, CTCtxtPath)
}

// ConnectSuperToCTFake adds a FAKE CTCtxtPath from given sending
// Super layer to a CT layer uses a OneToOne path pattern,
// and sets the class to CTFromSuper.
// This does NOT make a CTCtxtPath -- instead makes a regular Path -- for testing!
func (nt *Network) ConnectSuperToCTFake(send, recv *Layer) *Path {
	pt := nt.ConnectLayers(send, recv, paths.NewOneToOne(), CTCtxtPath)
	pt.AddClass("CTFromSuper")
	return pt
}

// ConnectCtxtToCTFake adds a FAKE CTCtxtPath from given sending layer to a CT layer
// This does NOT make a CTCtxtPath -- instead makes a regular Path -- for testing!
func (nt *Network) ConnectCtxtToCTFake(send, recv *Layer, pat paths.Pattern) *Path {
	return nt.ConnectLayers(send, recv, pat, CTCtxtPath)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network versions of Add Layer methods

// AddDeep2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and Pulvinar Pulvinar for Super (P suffix).
// Pulvinar projects back to Super and CT layers, type = Back, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the Pulvinar layer, and it must be sized appropriately for those drivers.
func (nt *Network) AddDeep2D(name string, shapeY, shapeX int) (super, ct, pulv *Layer) {
	super = nt.AddSuperLayer2D(name, shapeY, shapeX)
	ct = nt.AddCTLayer2D(name+"CT", shapeY, shapeX)
	ct.PlaceBehind(super, 2)
	full := paths.NewFull()
	nt.ConnectSuperToCT(super, ct)
	pulv = nt.AddPulvinarLayer2D(name+"P", shapeY, shapeX)
	pulv.PlaceBehind(ct, 2)
	nt.ConnectLayers(ct, pulv, full, ForwardPath)
	nt.ConnectLayers(pulv, super, full, BackPath).AddClass("FromPulv")
	nt.ConnectLayers(pulv, ct, full, BackPath).AddClass("FromPulv")
	return
}

// AddDeep4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and Pulvinar Pulvinar for Super (P suffix).
// Pulvinar projects back to Super and CT layers, also PoolOneToOne, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the Pulvinar layer, and it must be sized appropriately for those drivers.
func (nt *Network) AddDeep4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, pulv *Layer) {
	super = nt.AddSuperLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = nt.AddCTLayer4D(name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.PlaceBehind(super, 2)
	pone2one := paths.NewPoolOneToOne()
	nt.ConnectSuperToCT(super, ct)
	pulv = nt.AddPulvinarLayer4D(name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	pulv.PlaceBehind(ct, 2)
	nt.ConnectLayers(ct, pulv, pone2one, ForwardPath)
	nt.ConnectLayers(pulv, super, pone2one, BackPath).AddClass("FromPulv")
	nt.ConnectLayers(pulv, ct, pone2one, BackPath).AddClass("FromPulv")
	return
}

// AddDeep2DFakeCT adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with FAKE CTCtxtPath OneToOne pathway from Super to CT, and Pulvinar Pulvinar for Super (P suffix).
// Pulvinar projects back to Super and CT layers, type = Back, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the Pulvinar layer, and it must be sized appropriately for those drivers.
// This does NOT make a CTCtxtPath -- instead makes a regular Path -- for testing!
func (nt *Network) AddDeep2DFakeCT(name string, shapeY, shapeX int) (super, ct, pulv *Layer) {
	super = nt.AddSuperLayer2D(name, shapeY, shapeX)
	ct = nt.AddCTLayer2D(name+"CT", shapeY, shapeX)
	ct.PlaceBehind(super, 2)
	full := paths.NewFull()
	nt.ConnectSuperToCTFake(super, ct)
	pulv = nt.AddPulvinarLayer2D(name+"P", shapeY, shapeX)
	pulv.PlaceBehind(ct, 2)
	nt.ConnectLayers(ct, pulv, full, ForwardPath)
	nt.ConnectLayers(pulv, super, full, BackPath).AddClass("FromPulv")
	nt.ConnectLayers(pulv, ct, full, BackPath).AddClass("FromPulv")
	return
}

// AddDeep4DFakeCT adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with FAKE CTCtxtPath OneToOne pathway from Super to CT, and Pulvinar Pulvinar for Super (P suffix).
// Pulvinar projects back to Super and CT layers, also PoolOneToOne, class = FromPulv
// CT is placed Behind Super, and Pulvinar behind CT.
// Drivers must be added to the Pulvinar layer, and it must be sized appropriately for those drivers.
// This does NOT make a CTCtxtPath -- instead makes a regular Path -- for testing!
func (nt *Network) AddDeep4DFakeCT(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct, pulv *Layer) {
	super = nt.AddSuperLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = nt.AddCTLayer4D(name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.PlaceBehind(super, 2)
	pone2one := paths.NewPoolOneToOne()
	nt.ConnectSuperToCTFake(super, ct)
	pulv = nt.AddPulvinarLayer4D(name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
	pulv.PlaceBehind(ct, 2)
	nt.ConnectLayers(ct, pulv, pone2one, ForwardPath)
	nt.ConnectLayers(pulv, super, pone2one, BackPath).AddClass("FromPulv")
	nt.ConnectLayers(pulv, ct, pone2one, BackPath).AddClass("FromPulv")
	return
}

// AddDeepNoPulvinar2D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath OneToOne pathway from Super to CT, and NO Pulvinar Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddDeepNoPulvinar2D(name string, shapeY, shapeX int) (super, ct *Layer) {
	super = nt.AddSuperLayer2D(name, shapeY, shapeX)
	ct = nt.AddCTLayer2D(name+"CT", shapeY, shapeX)
	ct.PlaceBehind(super, 2)
	nt.ConnectSuperToCT(super, ct)
	return
}

// AddDeepNoPulvinar4D adds a superficial (SuperLayer) and corresponding CT (CT suffix) layer
// with CTCtxtPath PoolOneToOne pathway from Super to CT, and NO Pulvinar Pulvinar.
// CT is placed Behind Super.
func (nt *Network) AddDeepNoPulvinar4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) (super, ct *Layer) {
	super = nt.AddSuperLayer4D(name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = nt.AddCTLayer4D(name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct.PlaceBehind(super, 2)
	nt.ConnectSuperToCT(super, ct)
	return
}
