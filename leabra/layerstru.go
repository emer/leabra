// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/giv"
	"github.com/goki/mat32"
)

// leabra.LayerStru manages the structural elements of the layer, which are common
// to any Layer type
type LayerStru struct {
	LeabraLay LeabraLayer    `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an LeabraLayer (which subsumes emer.Layer), which can always be used to extract the true underlying type of object when layer is embedded in other structs -- function receivers do not have this ability so this is necessary."`
	Network   emer.Network   `copy:"-" json:"-" xml:"-" view:"-" desc:"our parent network, in case we need to use it to find other layers etc -- set when added by network"`
	Nm        string         `desc:"Name of the layer -- this must be unique within the network, which has a map for quick lookup and layers are typically accessed directly by name"`
	Cls       string         `desc:"Class is for applying parameter styles, can be space separated multple tags"`
	Off       bool           `desc:"inactivate this layer -- allows for easy experimentation"`
	Shp       etensor.Shape  `desc:"shape of the layer -- can be 2D for basic layers and 4D for layers with sub-groups (hypercolumns) -- order is outer-to-inner (row major) so Y then X for 2D and for 4D: Y-X unit pools then Y-X neurons within pools"`
	Typ       emer.LayerType `desc:"type of layer -- Hidden, Input, Target, Compare, or extended type in specialized algorithms -- matches against .Class parameter styles (e.g., .Hidden etc)"`
	Thr       int            `desc:"the thread number (go routine) to use in updating this layer. The user is responsible for allocating layers to threads, trying to maintain an even distribution across layers and establishing good break-points."`
	Rel       relpos.Rel     `view:"inline" desc:"Spatial relationship to other layer, determines positioning"`
	Ps        mat32.Vec3     `desc:"position of lower-left-hand corner of layer in 3D space, computed from Rel.  Layers are in X-Y width - height planes, stacked vertically in Z axis."`
	Idx       int            `desc:"a 0..n-1 index of the position of the layer within list of layers in the network. For Leabra networks, it only has significance in determining who gets which weights for enforcing initial weight symmetry -- higher layers get weights from lower layers."`
	RcvPrjns  emer.Prjns     `desc:"list of receiving projections into this layer from other layers"`
	SndPrjns  emer.Prjns     `desc:"list of sending projections from this layer to other layers"`
}

// emer.Layer interface methods

// InitName MUST be called to initialize the layer's pointer to itself as an emer.Layer
// which enables the proper interface methods to be called.  Also sets the name, and
// the parent network that this layer belongs to (which layers may want to retain).
func (ls *LayerStru) InitName(lay emer.Layer, name string, net emer.Network) {
	ls.LeabraLay = lay.(LeabraLayer)
	ls.Nm = name
	ls.Network = net
}

func (ls *LayerStru) Name() string               { return ls.Nm }
func (ls *LayerStru) SetName(nm string)          { ls.Nm = nm }
func (ls *LayerStru) Label() string              { return ls.Nm }
func (ls *LayerStru) Class() string              { return ls.Typ.String() + " " + ls.Cls }
func (ls *LayerStru) SetClass(cls string)        { ls.Cls = cls }
func (ls *LayerStru) TypeName() string           { return "Layer" } // type category, for params..
func (ls *LayerStru) Type() emer.LayerType       { return ls.Typ }
func (ls *LayerStru) SetType(typ emer.LayerType) { ls.Typ = typ }
func (ls *LayerStru) IsOff() bool                { return ls.Off }
func (ls *LayerStru) SetOff(off bool)            { ls.Off = off }
func (ls *LayerStru) Shape() *etensor.Shape      { return &ls.Shp }
func (ls *LayerStru) Is2D() bool                 { return ls.Shp.NumDims() == 2 }
func (ls *LayerStru) Is4D() bool                 { return ls.Shp.NumDims() == 4 }
func (ls *LayerStru) Thread() int                { return ls.Thr }
func (ls *LayerStru) SetThread(thr int)          { ls.Thr = thr }
func (ls *LayerStru) RelPos() relpos.Rel         { return ls.Rel }
func (ls *LayerStru) Pos() mat32.Vec3            { return ls.Ps }
func (ls *LayerStru) SetPos(pos mat32.Vec3)      { ls.Ps = pos }
func (ls *LayerStru) Index() int                 { return ls.Idx }
func (ls *LayerStru) SetIndex(idx int)           { ls.Idx = idx }
func (ls *LayerStru) RecvPrjns() *emer.Prjns     { return &ls.RcvPrjns }
func (ls *LayerStru) NRecvPrjns() int            { return len(ls.RcvPrjns) }
func (ls *LayerStru) RecvPrjn(idx int) emer.Prjn { return ls.RcvPrjns[idx] }
func (ls *LayerStru) SendPrjns() *emer.Prjns     { return &ls.SndPrjns }
func (ls *LayerStru) NSendPrjns() int            { return len(ls.SndPrjns) }
func (ls *LayerStru) SendPrjn(idx int) emer.Prjn { return ls.SndPrjns[idx] }

func (ls *LayerStru) Idx4DFrom2D(x, y int) ([]int, bool) {
	lshp := ls.Shape()
	nux := lshp.Dim(3)
	nuy := lshp.Dim(2)
	ux := x % nux
	uy := y % nuy
	px := x / nux
	py := y / nuy
	idx := []int{py, px, uy, ux}
	if !lshp.IdxIsValid(idx) {
		return nil, false
	}
	return idx, true
}

func (ls *LayerStru) SetRelPos(rel relpos.Rel) {
	ls.Rel = rel
	if ls.Rel.Scale == 0 {
		ls.Rel.Defaults()
	}
}

func (ls *LayerStru) Size() mat32.Vec2 {
	if ls.Rel.Scale == 0 {
		ls.Rel.Defaults()
	}
	var sz mat32.Vec2
	switch {
	case ls.Is2D():
		sz = mat32.Vec2{float32(ls.Shp.Dim(1)), float32(ls.Shp.Dim(0))} // Y, X
	case ls.Is4D():
		// note: pool spacing is handled internally in display and does not affect overall size
		sz = mat32.Vec2{float32(ls.Shp.Dim(1) * ls.Shp.Dim(3)), float32(ls.Shp.Dim(0) * ls.Shp.Dim(2))} // Y, X
	default:
		sz = mat32.Vec2{float32(ls.Shp.Len()), 1}
	}
	return sz.MulScalar(ls.Rel.Scale)
}

// SetShape sets the layer shape and also uses default dim names
func (ls *LayerStru) SetShape(shape []int) {
	var dnms []string
	if len(shape) == 2 {
		dnms = emer.LayerDimNames2D
	} else if len(shape) == 4 {
		dnms = emer.LayerDimNames4D
	}
	ls.Shp.SetShape(shape, nil, dnms) // row major default
}

// NPools returns the number of unit sub-pools according to the shape parameters.
// Currently supported for a 4D shape, where the unit pools are the first 2 Y,X dims
// and then the units within the pools are the 2nd 2 Y,X dims
func (ls *LayerStru) NPools() int {
	if ls.Shp.NumDims() != 4 {
		return 0
	}
	return ls.Shp.Dim(0) * ls.Shp.Dim(1)
}

// RecipToSendPrjn finds the reciprocal projection relative to the given sending projection
// found within the SendPrjns of this layer.  This is then a recv prjn within this layer:
//  S=A -> R=B recip: R=A <- S=B -- ly = A -- we are the sender of srj and recv of rpj.
// returns false if not found.
func (ls *LayerStru) RecipToSendPrjn(spj emer.Prjn) (emer.Prjn, bool) {
	for _, rpj := range ls.RcvPrjns {
		if rpj.SendLay() == spj.RecvLay() {
			return rpj, true
		}
	}
	return nil, false
}

// Config configures the basic properties of the layer
func (ls *LayerStru) Config(shape []int, typ emer.LayerType) {
	ls.SetShape(shape)
	ls.Typ = typ
}

// ApplyParams applies given parameter style Sheet to this layer and its recv projections.
// Calls UpdateParams on anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ls *LayerStru) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	app, err := pars.Apply(ls.LeabraLay, setMsg) // essential to go through LeabraPrj
	if app {
		ls.LeabraLay.UpdateParams()
		applied = true
	}
	if err != nil {
		rerr = err
	}
	for _, pj := range ls.RcvPrjns {
		app, err = pj.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
}

// NonDefaultParams returns a listing of all parameters in the Layer that
// are not at their default values -- useful for setting param styles etc.
func (ls *LayerStru) NonDefaultParams() string {
	nds := giv.StructNonDefFieldsStr(ls.LeabraLay, ls.Nm)
	for _, pj := range ls.RcvPrjns {
		pnd := pj.NonDefaultParams()
		nds += pnd
	}
	return nds
}
