// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/path"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/leabra/v2/deep"
	"github.com/emer/leabra/v2/glong"
	"github.com/emer/leabra/v2/leabra"
	"github.com/emer/leabra/v2/pcore"
)

// agate.Network has methods for configuring specialized AGate network components
// for Attentional & adaptive Gating of Action and Thought for Executive function.
type Network struct {
	deep.Network
}

// Defaults sets all the default parameters for all layers and pathways
func (nt *Network) Defaults() {
	nt.Network.Defaults()
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and pathways
func (nt *Network) UpdateParams() {
	nt.Network.UpdateParams()
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return NeuronVarsAll
}

// UnitVarProps returns properties for variables
func (nt *Network) UnitVarProps() map[string]string {
	return glong.NeuronVarProps
}

// SynVarNames returns the names of all the variables on the synapses in this network.
func (nt *Network) SynVarNames() []string {
	return pcore.SynVarsAll
}

// AddBG adds MtxGo, No, CIN, GPeOut, GPeIn, GPeTA, STNp, STNs, GPi, and VThal layers,
// with given optional prefix.
// Assumes that a 4D structure will be used, with Pools representing separable gating domains.
// Only Matrix has more than 1 unit per Pool by default.
// Appropriate PoolOneToOne connections are made between layers,
// using standard styles.
// space is the spacing between layers (2 typical)
func (nt *Network) AddBG(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int, space float32) (mtxGo, mtxNo, cin, gpeOut, gpeIn, gpeTA, stnp, stns, gpi, vthal leabra.LeabraLayer) {
	return pcore.AddBG(&nt.Network.Network, prefix, nPoolsY, nPoolsX, nNeurY, nNeurX, space)
}

// ConnectToMatrix adds a MatrixTracePath from given sending layer to a matrix layer
func (nt *Network) ConnectToMatrix(send, recv emer.Layer, pat path.Pattern) emer.Path {
	return pcore.ConnectToMatrix(&nt.Network.Network, send, recv, pat)
}

// AddPFC adds a PFC system including SuperLayer, CT with CTCtxtPath, MaintLayer,
// and OutLayer which is gated by BG.
// Name is set to "PFC" if empty.  Other layers have appropriate suffixes.
// Optionally creates a TRC Pulvinar for Super.
// CT is placed Behind Super, then Out and Maint, and Pulvinar behind CT if created.
func (nt *Network) AddPFC(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) (super, ct, maint, out, pulv emer.Layer) {
	return AddPFC(&nt.Network.Network, name, nPoolsY, nPoolsX, nNeurY, nNeurX, pulvLay)
}

////////////////////////////////////////////////////////////////////////
// Network functions available here as standalone functions
//         for mixing in to other models

// AddMaintLayer adds a MaintLayer using 4D shape with pools,
// and lateral NMDAMaint PoolOneToOne connectivity.
func AddMaintLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *MaintLayer {
	ly := &MaintLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	glong.ConnectNMDA(nt, ly, ly, path.NewPoolOneToOne())
	return ly
}

// AddOutLayer adds a OutLayer using 4D shape with pools,
// and lateral PoolOneToOne connectivity.
func AddOutLayer(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *OutLayer {
	ly := &OutLayer{}
	nt.AddLayerInit(ly, name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, emer.Hidden)
	return ly
}

// AddPFC adds a PFC system including SuperLayer, CT with CTCtxtPath, MaintLayer,
// and OutLayer which is gated by BG.
// Name is set to "PFC" if empty.  Other layers have appropriate suffixes.
// Optionally creates a TRC Pulvinar for Super.
// Standard Deep CTCtxtPath PoolOneToOne Super -> CT pathway, and
// 1to1 pathways Super -> Maint and Maint -> Out class PFCFixed are created by default.
// CT is placed Behind Super, then Out and Maint, and Pulvinar behind CT if created.
func AddPFC(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) (super, ct, maint, out, pulv emer.Layer) {
	if name == "" {
		name = "PFC"
	}
	super = deep.AddSuperLayer4D(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX)
	ct = deep.AddCTLayer4D(nt, name+"CT", nPoolsY, nPoolsX, nNeurY, nNeurX)
	mainti := AddMaintLayer(nt, name+"Mnt", nPoolsY, nPoolsX, nNeurY, nNeurX)
	maint = mainti
	outi := AddOutLayer(nt, name+"Out", nPoolsY, nPoolsX, nNeurY, nNeurX)
	out = outi

	ct.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: name, XAlign: relpos.Left, Space: 2})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: name, YAlign: relpos.Front, Space: 2})
	maint.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: out.Name(), XAlign: relpos.Left, Space: 2})

	one2one := path.NewOneToOne()
	deep.ConnectCtxtToCT(nt, super, ct, path.NewPoolOneToOne())

	pj := nt.ConnectLayers(super, maint, one2one, emer.Forward)
	pj.SetClass("PFCFixed")
	pj = nt.ConnectLayers(maint, out, one2one, emer.Forward)
	pj.SetClass("PFCFixed")
	mainti.InterInhib.Lays.Add(out.Name())

	if pulvLay {
		pulvi := deep.AddTRCLayer4D(nt, name+"P", nPoolsY, nPoolsX, nNeurY, nNeurX)
		pulv = pulvi
		pulvi.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: ct.Name(), XAlign: relpos.Left, Space: 2})
		// pulvi.DriverLay = name
	}
	return
}

// AddPFCPy adds a PFC system including SuperLayer, CT with CTCtxtPath, MaintLayer,
// and OutLayer which is gated by BG.
// Name is set to "PFC" if empty.  Other layers have appropriate suffixes.
// Optionally creates a TRC Pulvinar for Super.
// Standard Deep CTCtxtPath PoolOneToOne Super -> CT pathway, and
// 1to1 pathways Super -> Maint and Maint -> Out class PFCFixed are created by default.
// CT is placed Behind Super, then Out and Maint, and Pulvinar behind CT if created.
// Py is Python version, returns layers as a slice
func AddPFCPy(nt *leabra.Network, name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, pulvLay bool) []emer.Layer {
	super, ct, maint, out, pulv := AddPFC(nt, name, nPoolsY, nPoolsX, nNeurY, nNeurX, pulvLay)
	return []emer.Layer{super, ct, maint, out, pulv}
}
