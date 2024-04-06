// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/giv"
	"cogentcore.org/core/mat32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/etable/v2/etensor"
)

// LayerBase manages the structural elements of the layer, which are common
// to any Layer type
type LayerBase struct {

	// we need a pointer to ourselves as an LeabraLayer (which subsumes emer.Layer), which can always be used to extract the true underlying type of object when layer is embedded in other structs -- function receivers do not have this ability so this is necessary.
	LeabraLay LeabraLayer `copy:"-" json:"-" xml:"-" view:"-"`

	// our parent network, in case we need to use it to find other layers etc -- set when added by network
	Network emer.Network `copy:"-" json:"-" xml:"-" view:"-"`

	// Name of the layer -- this must be unique within the network, which has a map for quick lookup and layers are typically accessed directly by name
	Nm string

	// Class is for applying parameter styles, can be space separated multple tags
	Cls string

	// inactivate this layer -- allows for easy experimentation
	Off bool

	// shape of the layer -- can be 2D for basic layers and 4D for layers with sub-groups (hypercolumns) -- order is outer-to-inner (row major) so Y then X for 2D and for 4D: Y-X unit pools then Y-X neurons within pools
	Shp etensor.Shape

	// type of layer -- Hidden, Input, Target, Compare, or extended type in specialized algorithms -- matches against .Class parameter styles (e.g., .Hidden etc)
	Typ emer.LayerType

	// the thread number (go routine) to use in updating this layer. The user is responsible for allocating layers to threads, trying to maintain an even distribution across layers and establishing good break-points.
	Thr int

	// Spatial relationship to other layer, determines positioning
	Rel relpos.Rel `view:"inline"`

	// position of lower-left-hand corner of layer in 3D space, computed from Rel.  Layers are in X-Y width - height planes, stacked vertically in Z axis.
	Ps mat32.Vec3

	// a 0..n-1 index of the position of the layer within list of layers in the network. For Leabra networks, it only has significance in determining who gets which weights for enforcing initial weight symmetry -- higher layers get weights from lower layers.
	Index int

	// indexes of representative units in the layer, for computationally expensive stats or displays
	RepIxs []int

	// shape of representative units in the layer -- if RepIxs is empty or .Shp is nil, use overall layer shape
	RepShp etensor.Shape

	// list of receiving projections into this layer from other layers
	RcvPrjns LeabraPrjns

	// list of sending projections from this layer to other layers
	SndPrjns LeabraPrjns
}

// emer.Layer interface methods

// InitName MUST be called to initialize the layer's pointer to itself as an emer.Layer
// which enables the proper interface methods to be called.  Also sets the name, and
// the parent network that this layer belongs to (which layers may want to retain).
func (ls *LayerBase) InitName(lay emer.Layer, name string, net emer.Network) {
	ls.LeabraLay = lay.(LeabraLayer)
	ls.Nm = name
	ls.Network = net
}

func (ls *LayerBase) Name() string        { return ls.Nm }
func (ls *LayerBase) SetName(nm string)   { ls.Nm = nm }
func (ls *LayerBase) Label() string       { return ls.Nm }
func (ls *LayerBase) Class() string       { return ls.Typ.String() + " " + ls.Cls }
func (ls *LayerBase) SetClass(cls string) { ls.Cls = cls }
func (ly *LayerBase) AddClass(cls string) { ly.Cls = params.AddClass(ly.Cls, cls) }

func (ls *LayerBase) TypeName() string           { return "Layer" } // type category, for params..
func (ls *LayerBase) Type() emer.LayerType       { return ls.Typ }
func (ls *LayerBase) SetType(typ emer.LayerType) { ls.Typ = typ }
func (ls *LayerBase) IsOff() bool                { return ls.Off }
func (ls *LayerBase) SetOff(off bool)            { ls.Off = off }
func (ls *LayerBase) Shape() *etensor.Shape      { return &ls.Shp }
func (ls *LayerBase) Is2D() bool                 { return ls.Shp.NumDims() == 2 }
func (ls *LayerBase) Is4D() bool                 { return ls.Shp.NumDims() == 4 }
func (ls *LayerBase) Thread() int                { return ls.Thr }
func (ls *LayerBase) SetThread(thr int)          { ls.Thr = thr }
func (ls *LayerBase) RelPos() relpos.Rel         { return ls.Rel }
func (ls *LayerBase) Pos() mat32.Vec3            { return ls.Ps }
func (ls *LayerBase) SetPos(pos mat32.Vec3)      { ls.Ps = pos }
func (ls *LayerBase) Index() int                 { return ls.Index }
func (ls *LayerBase) SetIndex(idx int)           { ls.Index = idx }
func (ls *LayerBase) RecvPrjns() *LeabraPrjns    { return &ls.RcvPrjns }
func (ls *LayerBase) NRecvPrjns() int            { return len(ls.RcvPrjns) }
func (ls *LayerBase) RecvPrjn(idx int) emer.Prjn { return ls.RcvPrjns[idx] }
func (ls *LayerBase) SendPrjns() *LeabraPrjns    { return &ls.SndPrjns }
func (ls *LayerBase) NSendPrjns() int            { return len(ls.SndPrjns) }
func (ls *LayerBase) SendPrjn(idx int) emer.Prjn { return ls.SndPrjns[idx] }
func (ls *LayerBase) RepIndexes() []int          { return ls.RepIxs }

func (ly *LayerBase) SendNameTry(sender string) (emer.Prjn, error) {
	return emer.SendNameTry(ly.LeabraLay, sender)
}
func (ly *LayerBase) SendName(sender string) emer.Prjn {
	pj, _ := emer.SendNameTry(ly.LeabraLay, sender)
	return pj
}
func (ly *LayerBase) SendNameTypeTry(sender, typ string) (emer.Prjn, error) {
	return emer.SendNameTypeTry(ly.LeabraLay, sender, typ)
}
func (ly *LayerBase) RecvNameTry(receiver string) (emer.Prjn, error) {
	return emer.RecvNameTry(ly.LeabraLay, receiver)
}
func (ly *LayerBase) RecvName(receiver string) emer.Prjn {
	pj, _ := emer.RecvNameTry(ly.LeabraLay, receiver)
	return pj
}
func (ly *LayerBase) RecvNameTypeTry(receiver, typ string) (emer.Prjn, error) {
	return emer.RecvNameTypeTry(ly.LeabraLay, receiver, typ)
}

func (ls *LayerBase) Index4DFrom2D(x, y int) ([]int, bool) {
	lshp := ls.Shape()
	nux := lshp.Dim(3)
	nuy := lshp.Dim(2)
	ux := x % nux
	uy := y % nuy
	px := x / nux
	py := y / nuy
	idx := []int{py, px, uy, ux}
	if !lshp.IndexIsValid(idx) {
		return nil, false
	}
	return idx, true
}

func (ls *LayerBase) SetRelPos(rel relpos.Rel) {
	ls.Rel = rel
	if ls.Rel.Scale == 0 {
		ls.Rel.Defaults()
	}
}

func (ls *LayerBase) Size() mat32.Vec2 {
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
func (ls *LayerBase) SetShape(shape []int) {
	var dnms []string
	if len(shape) == 2 {
		dnms = emer.LayerDimNames2D
	} else if len(shape) == 4 {
		dnms = emer.LayerDimNames4D
	}
	ls.Shp.SetShape(shape, nil, dnms) // row major default
}

// SetRepIndexesShape sets the RepIndexes, and RepShape and as list of dimension sizes
func (ls *LayerBase) SetRepIndexesShape(idxs, shape []int) {
	ls.RepIxs = idxs
	var dnms []string
	if len(shape) == 2 {
		dnms = emer.LayerDimNames2D
	} else if len(shape) == 4 {
		dnms = emer.LayerDimNames4D
	}
	ls.RepShp.SetShape(shape, nil, dnms) // row major default
}

// RepShape returns the shape to use for representative units
func (ls *LayerBase) RepShape() *etensor.Shape {
	sz := len(ls.RepIxs)
	if sz == 0 {
		return &ls.Shp
	}
	if ls.RepShp.Len() < sz {
		ls.RepShp.SetShape([]int{sz}, nil, nil) // row major default
	}
	return &ls.RepShp
}

// NPools returns the number of unit sub-pools according to the shape parameters.
// Currently supported for a 4D shape, where the unit pools are the first 2 Y,X dims
// and then the units within the pools are the 2nd 2 Y,X dims
func (ls *LayerBase) NPools() int {
	if ls.Shp.NumDims() != 4 {
		return 0
	}
	return ls.Shp.Dim(0) * ls.Shp.Dim(1)
}

// RecipToSendPrjn finds the reciprocal projection relative to the given sending projection
// found within the SendPrjns of this layer.  This is then a recv prjn within this layer:
//
//	S=A -> R=B recip: R=A <- S=B -- ly = A -- we are the sender of srj and recv of rpj.
//
// returns false if not found.
func (ls *LayerBase) RecipToSendPrjn(spj emer.Prjn) (emer.Prjn, bool) {
	for _, rpj := range ls.RcvPrjns {
		if rpj.SendLay() == spj.RecvLay() {
			return rpj, true
		}
	}
	return nil, false
}

// Config configures the basic properties of the layer
func (ls *LayerBase) Config(shape []int, typ emer.LayerType) {
	ls.SetShape(shape)
	ls.Typ = typ
}

// ApplyParams applies given parameter style Sheet to this layer and its recv projections.
// Calls UpdateParams on anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ls *LayerBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
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
func (ls *LayerBase) NonDefaultParams() string {
	nds := giv.StructNonDefFieldsStr(ls.LeabraLay, ls.Nm)
	for _, pj := range ls.RcvPrjns {
		pnd := pj.NonDefaultParams()
		nds += pnd
	}
	return nds
}
