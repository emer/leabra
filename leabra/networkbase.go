// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/path"
	"github.com/emer/emergent/v2/timer"
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

	// timers for each major function (step of processing).
	FunTimes map[string]*timer.Time `display:"-"`
}

func (nt *NetworkBase) NumLayers() int               { return len(nt.Layers) }
func (nt *NetworkBase) EmerLayer(idx int) emer.Layer { return nt.Layers[idx] }
func (nt *NetworkBase) MaxParallelData() int         { return 1 }
func (nt *NetworkBase) NParallelData() int           { return 1 }

// NewNetwork returns a new leabra Network
func NewNetwork(name string) *Network {
	net := &Network{}
	emer.InitNetwork(net, name)
	return net
}

// LayerByName returns a layer by looking it up by name in the layer map
// (nil if not found).
func (nt *Network) LayerByName(name string) *Layer {
	ely, _ := nt.EmerLayerByName(name)
	return ely.(*Layer)
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
	for _, lyi := range nt.Layers {
		if lyi.Off {
			continue
		}
		ly := lyi.(LeabraLayer).AsLeabra()
		str += "\nLayer: " + ly.Name() + "\n"
		rpjn := ly.RecvPaths
		for _, p := range rpjn {
			if p.Off {
				continue
			}
			pj := p.AsLeabra()
			str += fmt.Sprintf("\t%23s\t\tAbs:\t%g\tRel:\t%g\n", pj.Name(), pj.WtScale.Abs, pj.WtScale.Rel)
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
		ly.SetIndex(li)
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
	ly.Config(shape, typ)
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
func (nt *Network) AddLayer(name string, shape []int, typ emer.LayerType) emer.Layer {
	ly := nt.EmerNet.(LeabraNetwork).NewLayer() // essential to use EmerNet interface here!
	nt.AddLayerInit(ly, name, shape, typ)
	return ly
}

// AddLayer2D adds a new layer with given name and 2D shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential.
func (nt *Network) AddLayer2D(name string, shapeY, shapeX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{shapeY, shapeX}, typ)
}

// AddLayer4D adds a new layer with given name and 4D shape to the network.
// 4D layers enable pool (unit-group) level inhibition in Leabra networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each pool
// having 4 rows (Y) of 5 (X) neurons.
func (nt *Network) AddLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, typ)
}

// ConnectLayerNames establishes a pathway between two layers, referenced by name
// adding to the recv and send pathway lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *Network) ConnectLayerNames(send, recv string, pat path.Pattern, typ emer.PathType) (rlay, slay emer.Layer, pj emer.Path, err error) {
	rlay, err = nt.LayerByNameTry(recv)
	if err != nil {
		return
	}
	slay, err = nt.LayerByNameTry(send)
	if err != nil {
		return
	}
	pj = nt.ConnectLayers(slay, rlay, pat, typ)
	return
}

// ConnectLayers establishes a pathway between two layers,
// adding to the recv and send pathway lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) ConnectLayers(send, recv emer.Layer, pat path.Pattern, typ emer.PathType) emer.Path {
	pj := nt.EmerNet.(LeabraNetwork).NewPath() // essential to use EmerNet interface here!
	return nt.ConnectLayersPath(send, recv, pat, typ, pj)
}

// ConnectLayersPath makes connection using given pathway between two layers,
// adding given path to the recv and send pathway lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) ConnectLayersPath(send, recv emer.Layer, pat path.Pattern, typ emer.PathType, pj emer.Path) emer.Path {
	pj.Init(pj)
	pj.AsLeabra().Connect(send, recv, pat, typ)
	recv.(LeabraLayer).RecvPaths().Add(pj.(LeabraPath))
	send.(LeabraLayer).SendPaths().Add(pj.(LeabraPath))
	return pj
}

// BidirConnectLayerNames establishes bidirectional pathways between two layers,
// referenced by name, with low = the lower layer that sends a Forward pathway
// to the high layer, and receives a Back pathway in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *Network) BidirConnectLayerNames(low, high string, pat path.Pattern) (lowlay, highlay emer.Layer, fwdpj, backpj emer.Path, err error) {
	lowlay, err = nt.LayerByNameTry(low)
	if err != nil {
		return
	}
	highlay, err = nt.LayerByNameTry(high)
	if err != nil {
		return
	}
	fwdpj = nt.ConnectLayers(lowlay, highlay, pat, emer.Forward)
	backpj = nt.ConnectLayers(highlay, lowlay, pat, BackPath)
	return
}

// BidirConnectLayers establishes bidirectional pathways between two layers,
// with low = lower layer that sends a Forward pathway to the high layer,
// and receives a Back pathway in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) BidirConnectLayers(low, high emer.Layer, pat path.Pattern) (fwdpj, backpj emer.Path) {
	fwdpj = nt.ConnectLayers(low, high, pat, emer.Forward)
	backpj = nt.ConnectLayers(high, low, pat, BackPath)
	return
}

// BidirConnectLayersPy establishes bidirectional pathways between two layers,
// with low = lower layer that sends a Forward pathway to the high layer,
// and receives a Back pathway in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
// Py = python version with no return vals.
func (nt *Network) BidirConnectLayersPy(low, high emer.Layer, pat path.Pattern) {
	nt.ConnectLayers(low, high, pat, emer.Forward)
	nt.ConnectLayers(high, low, pat, BackPath)
}

// LateralConnectLayer establishes a self-pathway within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) LateralConnectLayer(lay emer.Layer, pat path.Pattern) emer.Path {
	pj := nt.EmerNet.(LeabraNetwork).NewPath() // essential to use EmerNet interface here!
	return nt.LateralConnectLayerPath(lay, pat, pj)
}

// LateralConnectLayerPath makes lateral self-pathway using given pathway.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) LateralConnectLayerPath(lay emer.Layer, pat path.Pattern, pj emer.Path) emer.Path {
	pj.Init(pj)
	pj.AsLeabra().Connect(lay, lay, pat, emer.Lateral)
	lay.(LeabraLayer).RecvPaths().Add(pj.(LeabraPath))
	lay.(LeabraLayer).SendPaths().Add(pj.(LeabraPath))
	return pj
}

// Build constructs the layer and pathway state based on the layer shapes
// and patterns of interconnectivity
func (nt *Network) Build() error {
	nt.StopThreads() // any existing..
	nt.LayClassMap = make(map[string][]string)
	emsg := ""
	for li, ly := range nt.Layers {
		ly.SetIndex(li)
		if ly.Off {
			continue
		}
		err := ly.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
		cls := strings.Split(ly.Class(), " ")
		for _, cl := range cls {
			ll := nt.LayClassMap[cl]
			ll = append(ll, ly.Name())
			nt.LayClassMap[cl] = ll
		}
	}
	nt.Layout()
	nt.BuildThreads()
	nt.StartThreads()
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
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

// TimerReport reports the amount of time spent in each function, and in each thread
func (nt *Network) TimerReport() {
	fmt.Printf("TimerReport: %v, NThreads: %v\n", nt.Nm, nt.NThreads)
	fmt.Printf("\t%13s \t%7s\t%7s\n", "Function Name", "Secs", "Pct")
	nfn := len(nt.FunTimes)
	fnms := make([]string, nfn)
	idx := 0
	for k := range nt.FunTimes {
		fnms[idx] = k
		idx++
	}
	sort.StringSlice(fnms).Sort()
	pcts := make([]float64, nfn)
	tot := 0.0
	for i, fn := range fnms {
		pcts[i] = nt.FunTimes[fn].TotalSecs()
		tot += pcts[i]
	}
	for i, fn := range fnms {
		fmt.Printf("\t%13s \t%7.3f\t%7.1f\n", fn, pcts[i], 100*(pcts[i]/tot))
	}
	fmt.Printf("\t%13s \t%7.3f\n", "Total", tot)

	if nt.NThreads <= 1 {
		return
	}
	fmt.Printf("\n\tThr\tSecs\tPct\n")
	pcts = make([]float64, nt.NThreads)
	tot = 0.0
	for th := 0; th < nt.NThreads; th++ {
		pcts[th] = nt.ThrTimes[th].TotalSecs()
		tot += pcts[th]
	}
	for th := 0; th < nt.NThreads; th++ {
		fmt.Printf("\t%v \t%7.3f\t%7.1f\n", th, pcts[th], 100*(pcts[th]/tot))
	}
}

// FunTimerStart starts function timer for given function name -- ensures creation of timer
func (nt *Network) FunTimerStart(fun string) {
	ft, ok := nt.FunTimes[fun]
	if !ok {
		ft = &timer.Time{}
		nt.FunTimes[fun] = ft
	}
	ft.Start()
}

// FunTimerStop stops function timer -- timer must already exist
func (nt *Network) FunTimerStop(fun string) {
	ft := nt.FunTimes[fun]
	ft.Stop()
}
