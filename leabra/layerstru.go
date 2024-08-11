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

//////////////////////////////////////////////////////////////////////////////////////
//  LayerType

// LayerType is the type of the layer: Input, Hidden, Target, Compare.
// Class parameter styles automatically key off of these types.
// Specialized algorithms can extend this to other types, but these types encompass
// most standard neural network models.
type LayerType int32 //enums:enum

// The layer types
const (
	// Hidden is an internal representational layer that does not receive direct input / targets
	Hidden LayerType = iota

	// Input is a layer that receives direct external input in its Ext inputs
	Input

	// Target is a layer that receives direct external target inputs used for driving plus-phase learning
	Target

	// Compare is a layer that receives external comparison inputs, which drive statistics but
	// do NOT drive activation or learning directly
	Compare
)

// leabra.LayerStru manages the structural elements of the layer, which are common
// to any Layer type
type LayerStru struct {

	// [view: -] we need a pointer to ourselves as an LeabraLayer (which subsumes emer.Layer), which can always be used to extract the true underlying type of object when layer is embedded in other structs -- function receivers do not have this ability so this is necessary.
	LeabraLay LeabraLayer `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an LeabraLayer (which subsumes emer.Layer), which can always be used to extract the true underlying type of object when layer is embedded in other structs -- function receivers do not have this ability so this is necessary."`

	// [view: -] our parent network, in case we need to use it to find other layers etc -- set when added by network
	Network emer.Network `copy:"-" json:"-" xml:"-" view:"-" desc:"our parent network, in case we need to use it to find other layers etc -- set when added by network"`

	// Name of the layer -- this must be unique within the network, which has a map for quick lookup and layers are typically accessed directly by name
	Nm string `desc:"Name of the layer -- this must be unique within the network, which has a map for quick lookup and layers are typically accessed directly by name"`

	// Class is for applying parameter styles, can be space separated multple tags
	Cls string `desc:"Class is for applying parameter styles, can be space separated multple tags"`

	// inactivate this layer -- allows for easy experimentation
	Off bool `desc:"inactivate this layer -- allows for easy experimentation"`

	// shape of the layer -- can be 2D for basic layers and 4D for layers with sub-groups (hypercolumns) -- order is outer-to-inner (row major) so Y then X for 2D and for 4D: Y-X unit pools then Y-X neurons within pools
	Shp etensor.Shape `desc:"shape of the layer -- can be 2D for basic layers and 4D for layers with sub-groups (hypercolumns) -- order is outer-to-inner (row major) so Y then X for 2D and for 4D: Y-X unit pools then Y-X neurons within pools"`

	// type of layer -- Hidden, Input, Target, Compare, or extended type in specialized algorithms -- matches against .Class parameter styles (e.g., .Hidden etc)
	Typ emer.LayerType `desc:"type of layer -- Hidden, Input, Target, Compare, or extended type in specialized algorithms -- matches against .Class parameter styles (e.g., .Hidden etc)"`

	// the thread number (go routine) to use in updating this layer. The user is responsible for allocating layers to threads, trying to maintain an even distribution across layers and establishing good break-points.
	Thr int `desc:"the thread number (go routine) to use in updating this layer. The user is responsible for allocating layers to threads, trying to maintain an even distribution across layers and establishing good break-points."`

	// [view: inline] Spatial relationship to other layer, determines positioning
	Rel relpos.Rel `view:"inline" desc:"Spatial relationship to other layer, determines positioning"`

	// position of lower-left-hand corner of layer in 3D space, computed from Rel.  Layers are in X-Y width - height planes, stacked vertically in Z axis.
	Ps mat32.Vec3 `desc:"position of lower-left-hand corner of layer in 3D space, computed from Rel.  Layers are in X-Y width - height planes, stacked vertically in Z axis."`

	// a 0..n-1 index of the position of the layer within list of layers in the network. For Leabra networks, it only has significance in determining who gets which weights for enforcing initial weight symmetry -- higher layers get weights from lower layers.
	Idx int `desc:"a 0..n-1 index of the position of the layer within list of layers in the network. For Leabra networks, it only has significance in determining who gets which weights for enforcing initial weight symmetry -- higher layers get weights from lower layers."`

	// indexes of representative units in the layer, for computationally expensive stats or displays
	RepIxs []int `desc:"indexes of representative units in the layer, for computationally expensive stats or displays"`

	// shape of representative units in the layer -- if RepIxs is empty or .Shp is nil, use overall layer shape
	RepShp etensor.Shape `desc:"shape of representative units in the layer -- if RepIxs is empty or .Shp is nil, use overall layer shape"`

	// list of receiving pathways into this layer from other layers
	RecvPaths LeabraPaths `desc:"list of receiving pathways into this layer from other layers"`

	// list of sending pathways from this layer to other layers
	SndPaths LeabraPaths `desc:"list of sending pathways from this layer to other layers"`
}

// emer.Layer interface methods

// InitName MUST be called to initialize the layer's pointer to itself as an emer.Layer
// which enables the proper interface methods to be called.  Also sets the name, and
// the parent network that this layer belongs to (which layers may want to retain).
func (ls *LayerStru) InitName(lay emer.Layer, name string, net emer.Network) {
	ls.LeabraLay = lay.(LeabraLayer)
	ls.Name = name
	ls.Network = net
}

func (ls *LayerStru) TypeName() string           { return ls.Type.String() }
func (ls *LayerStru) RecvPaths() *LeabraPaths    { return &ls.RecvPaths }
func (ls *LayerStru) NumRecvPaths() int          { return len(ls.RecvPaths) }
func (ls *LayerStru) RecvPath(idx int) emer.Path { return ls.RecvPaths[idx] }
func (ls *LayerStru) SendPaths() *LeabraPaths    { return &ls.SendPaths }
func (ls *LayerStru) NumSendPaths() int          { return len(ls.SendPaths) }
func (ls *LayerStru) SendPath(idx int) emer.Path { return ls.SendPaths[idx] }

func (ly *LayerStru) SendNameTry(sender string) (emer.Path, error) {
	return emer.SendNameTry(ly.LeabraLay, sender)
}
func (ly *LayerStru) SendName(sender string) emer.Path {
	pj, _ := emer.SendNameTry(ly.LeabraLay, sender)
	return pj
}
func (ly *LayerStru) SendNameTypeTry(sender, typ string) (emer.Path, error) {
	return emer.SendNameTypeTry(ly.LeabraLay, sender, typ)
}
func (ly *LayerStru) RecvNameTry(receiver string) (emer.Path, error) {
	return emer.RecvNameTry(ly.LeabraLay, receiver)
}
func (ly *LayerStru) RecvName(receiver string) emer.Path {
	pj, _ := emer.RecvNameTry(ly.LeabraLay, receiver)
	return pj
}
func (ly *LayerStru) RecvNameTypeTry(receiver, typ string) (emer.Path, error) {
	return emer.RecvNameTypeTry(ly.LeabraLay, receiver, typ)
}

func (ls *LayerStru) Idx4DFrom2D(x, y int) ([]int, bool) {
	lshp := ls.Shape
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

// SetRepIdxsShape sets the RepIdxs, and RepShape and as list of dimension sizes
func (ls *LayerStru) SetRepIdxsShape(idxs, shape []int) {
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
func (ls *LayerStru) RepShape() *etensor.Shape {
	sz := len(ls.RepIxs)
	if sz == 0 {
		return &ls.Shp
	}
	if ls.RepShp.Len() < sz {
		ls.RepShp.SetShape([]int{sz}, nil, nil) // row major default
	}
	return &ls.RepShp
}

// RecipToSendPath finds the reciprocal pathway relative to the given sending pathway
// found within the SendPaths of this layer.  This is then a recv path within this layer:
//
//	S=A -> R=B recip: R=A <- S=B -- ly = A -- we are the sender of srj and recv of rpj.
//
// returns false if not found.
func (ls *LayerStru) RecipToSendPath(spj emer.Path) (emer.Path, bool) {
	for _, rpj := range ls.RecvPaths {
		if rpj.SendLayer() == spj.RecvLayer() {
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

// ApplyParams applies given parameter style Sheet to this layer and its recv pathways.
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
	for _, pj := range ls.RecvPaths {
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
	nds := giv.StructNonDefFieldsStr(ls.LeabraLay, ls.Name)
	for _, pj := range ls.RecvPaths {
		pnd := pj.NonDefaultParams()
		nds += pnd
	}
	return nds
}
