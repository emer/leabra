// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

//go:generate core generate

import (
	"errors"
	"fmt"
	"log"

	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/paths"
)

// leabra.Network implements the Leabra algorithm, managing the Layers.
type Network struct {
	emer.NetworkBase

	// list of layers
	Layers []*Layer

	// number of parallel threads (go routines) to use.
	NThreads int `edit:"-"`

	// how frequently to update the weight balance average
	// weight factor -- relatively expensive.
	WtBalInterval int `def:"10"`

	// counter for how long it has been since last WtBal.
	WtBalCtr int `edit:"-"`
}

func (nt *Network) NumLayers() int               { return len(nt.Layers) }
func (nt *Network) EmerLayer(idx int) emer.Layer { return nt.Layers[idx] }
func (nt *Network) MaxParallelData() int         { return 1 }
func (nt *Network) NParallelData() int           { return 1 }

// NewNetwork returns a new leabra Network
func NewNetwork(name string) *Network {
	net := &Network{}
	emer.InitNetwork(net, name)
	net.NThreads = 1
	return net
}

// LayerByName returns a layer by looking it up by name in the layer map
// (nil if not found).
func (nt *Network) LayerByName(name string) *Layer {
	ely, _ := nt.EmerLayerByName(name)
	return ely.(*Layer)
}

// LayersByType returns a list of layer names by given layer type(s).
func (nt *Network) LayersByType(layType ...LayerTypes) []string {
	var nms []string
	for _, tp := range layType {
		nm := tp.String()
		nms = append(nms, nm)
	}
	return nt.LayersByClass(nms...)
}

// KeyLayerParams returns a listing for all layers in the network,
// of the most important layer-level params (specific to each algorithm).
func (nt *Network) KeyLayerParams() string {
	return "" // todo: implement!
}

// KeyPathParams returns a listing for all Recv pathways in the network,
// of the most important pathway-level params (specific to each algorithm).
func (nt *Network) KeyPathParams() string {
	return nt.AllWtScales()
}

// AllWtScales returns a listing of all WtScale parameters in the Network
// in all Layers, Recv pathways.  These are among the most important
// and numerous of parameters (in larger networks) -- this helps keep
// track of what they all are set to.
func (nt *Network) AllWtScales() string {
	str := ""
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		str += "\nLayer: " + ly.Name + "\n"
		for _, pt := range ly.RecvPaths {
			if pt.Off {
				continue
			}
			str += fmt.Sprintf("\t%23s\t\tAbs:\t%g\tRel:\t%g\n", pt.Name, pt.WtScale.Abs, pt.WtScale.Rel)
		}
	}
	return str
}

// Defaults sets all the default parameters for all layers and pathways
func (nt *Network) Defaults() {
	nt.WtBalInterval = 10
	nt.WtBalCtr = 0
	for li, ly := range nt.Layers {
		ly.Defaults()
		ly.Index = li
	}
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and pathways
func (nt *Network) UpdateParams() {
	for _, ly := range nt.Layers {
		ly.UpdateParams()
	}
}

// UnitVarNames returns a list of variable names available on the units in this network.
// Not all layers need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) UnitVarNames() []string {
	return NeuronVars
}

// UnitVarProps returns properties for variables
func (nt *Network) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// SynVarNames returns the names of all the variables on the synapses in this network.
// Not all pathways need to support all variables, but must safely return 0's for
// unsupported ones.  The order of this list determines NetView variable display order.
// This is typically a global list so do not modify!
func (nt *Network) SynVarNames() []string {
	return SynapseVars
}

// SynVarProps returns properties for variables
func (nt *Network) SynVarProps() map[string]string {
	return SynapseVarProps
}

// AddLayerInit is implementation routine that takes a given layer and
// adds it to the network, and initializes and configures it properly.
func (nt *Network) AddLayerInit(ly *Layer, name string, shape []int, typ LayerTypes) {
	if nt.EmerNetwork == nil {
		log.Printf("Network EmerNetwork is nil: MUST call emer.InitNetwork on network, passing a pointer to the network to initialize properly!")
		return
	}
	emer.InitLayer(ly, name)
	ly.SetShape(shape)
	ly.Type = typ
	nt.Layers = append(nt.Layers, ly)
	nt.UpdateLayerMaps()
}

// AddLayer adds a new layer with given name and shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential -- see
// AddLayer2D and 4D for convenience methods for those.  4D layers enable
// pool (unit-group) level inhibition in Leabra networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each unit
// group having 4 rows (Y) of 5 (X) units.
func (nt *Network) AddLayer(name string, shape []int, typ LayerTypes) *Layer {
	ly := &Layer{} // essential to use EmerNet interface here!
	nt.AddLayerInit(ly, name, shape, typ)
	return ly
}

// AddLayer2D adds a new layer with given name and 2D shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential.
func (nt *Network) AddLayer2D(name string, shapeY, shapeX int, typ LayerTypes) *Layer {
	return nt.AddLayer(name, []int{shapeY, shapeX}, typ)
}

// AddLayer4D adds a new layer with given name and 4D shape to the network.
// 4D layers enable pool (unit-group) level inhibition in Leabra networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each pool
// having 4 rows (Y) of 5 (X) neurons.
func (nt *Network) AddLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, typ LayerTypes) *Layer {
	return nt.AddLayer(name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, typ)
}

// ConnectLayerNames establishes a pathway between two layers, referenced by name
// adding to the recv and send pathway lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *Network) ConnectLayerNames(send, recv string, pat paths.Pattern, typ PathTypes) (rlay, slay *Layer, pt *Path, err error) {
	rlay = nt.LayerByName(recv)
	if rlay == nil {
		return
	}
	slay = nt.LayerByName(send)
	if slay == nil {
		return
	}
	pt = nt.ConnectLayers(slay, rlay, pat, typ)
	return
}

// ConnectLayers establishes a pathway between two layers,
// adding to the recv and send pathway lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) ConnectLayers(send, recv *Layer, pat paths.Pattern, typ PathTypes) *Path {
	pt := &Path{}
	emer.InitPath(pt)
	pt.Connect(send, recv, pat, typ)
	recv.RecvPaths = append(recv.RecvPaths, pt)
	send.SendPaths = append(send.SendPaths, pt)
	return pt
}

// BidirConnectLayerNames establishes bidirectional pathways between two layers,
// referenced by name, with low = the lower layer that sends a Forward pathway
// to the high layer, and receives a Back pathway in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *Network) BidirConnectLayerNames(low, high string, pat paths.Pattern) (lowlay, highlay *Layer, fwdpj, backpj *Path, err error) {
	lowlay = nt.LayerByName(low)
	if lowlay == nil {
		return
	}
	highlay = nt.LayerByName(high)
	if highlay == nil {
		return
	}
	fwdpj = nt.ConnectLayers(lowlay, highlay, pat, ForwardPath)
	backpj = nt.ConnectLayers(highlay, lowlay, pat, BackPath)
	return
}

// BidirConnectLayers establishes bidirectional pathways between two layers,
// with low = lower layer that sends a Forward pathway to the high layer,
// and receives a Back pathway in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) BidirConnectLayers(low, high *Layer, pat paths.Pattern) (fwdpj, backpj *Path) {
	fwdpj = nt.ConnectLayers(low, high, pat, ForwardPath)
	backpj = nt.ConnectLayers(high, low, pat, BackPath)
	return
}

// BidirConnectLayersPy establishes bidirectional pathways between two layers,
// with low = lower layer that sends a Forward pathway to the high layer,
// and receives a Back pathway in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
// Py = python version with no return vals.
func (nt *Network) BidirConnectLayersPy(low, high *Layer, pat paths.Pattern) {
	nt.ConnectLayers(low, high, pat, ForwardPath)
	nt.ConnectLayers(high, low, pat, BackPath)
}

// LateralConnectLayer establishes a self-pathway within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) LateralConnectLayer(lay *Layer, pat paths.Pattern) *Path {
	pt := &Path{}
	return nt.LateralConnectLayerPath(lay, pat, pt)
}

// LateralConnectLayerPath makes lateral self-pathway using given pathway.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) LateralConnectLayerPath(lay *Layer, pat paths.Pattern, pt *Path) *Path {
	// pt.Init(pt)
	pt.Connect(lay, lay, pat, LateralPath)
	lay.RecvPaths = append(lay.RecvPaths, pt)
	lay.SendPaths = append(lay.SendPaths, pt)
	return pt
}

// Build constructs the layer and pathway state based on the layer shapes
// and patterns of interconnectivity
func (nt *Network) Build() error {
	nt.MakeLayerMaps()
	var errs []error
	for li, ly := range nt.Layers {
		ly.Index = li
		if ly.Off {
			continue
		}
		err := ly.Build()
		if err != nil {
			errs = append(errs, err)
		}
	}
	nt.LayoutLayers()
	return errors.Join(errs...)
}

// VarRange returns the min / max values for given variable
// todo: support r. s. pathway values
func (nt *Network) VarRange(varNm string) (min, max float32, err error) {
	first := true
	for _, ly := range nt.Layers {
		lmin, lmax, lerr := ly.VarRange(varNm)
		if lerr != nil {
			err = lerr
			return
		}
		if first {
			min = lmin
			max = lmax
			continue
		}
		if lmin < min {
			min = lmin
		}
		if lmax > max {
			max = lmax
		}
	}
	return
}
