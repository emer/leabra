// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"cogentcore.org/core/core"
	"cogentcore.org/core/gox/indent"
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/emergent/v2/relpos"
	"github.com/emer/emergent/v2/timer"
	"github.com/emer/emergent/v2/weights"
)

// LayFunChan is a channel that runs LeabraLayer functions
type LayFunChan chan func(ly LeabraLayer)

// leabra.NetworkBase holds the basic structural components of a network (layers)
type NetworkBase struct {

	// we need a pointer to ourselves as an emer.Network, which can always be used to extract the true underlying type of object when network is embedded in other structs -- function receivers do not have this ability so this is necessary.
	EmerNet emer.Network `copy:"-" json:"-" xml:"-" view:"-"`

	// overall name of network -- helps discriminate if there are multiple
	Nm string

	// list of layers
	Layers emer.Layers

	// filename of last weights file loaded or saved
	WtsFile string

	// map of name to layers -- layer names must be unique
	LayMap map[string]emer.Layer `view:"-"`

	// map of layer classes -- made during Build
	LayClassMap map[string][]string `view:"-"`

	// minimum display position in network
	MinPos math32.Vec3 `view:"-"`

	// maximum display position in network
	MaxPos math32.Vec3 `view:"-"`

	// optional metadata that is saved in network weights files -- e.g., can indicate number of epochs that were trained, or any other information about this network that would be useful to save
	MetaData map[string]string

	// number of parallel threads (go routines) to use -- this is computed directly from the Layers which you must explicitly allocate to different threads -- updated during Build of network
	NThreads int `inactive:"+"`

	// if set, runtime.LockOSThread() is called on the compute threads, which can be faster on large networks on some architectures -- experimentation is recommended
	LockThreads bool

	// layers per thread -- outer group is threads and inner is layers operated on by that thread -- based on user-assigned threads, initialized during Build
	ThrLay [][]emer.Layer `view:"-" inactive:"+"`

	// layer function channels, per thread
	ThrChans []LayFunChan `view:"-"`

	// timers for each thread, so you can see how evenly the workload is being distributed
	ThrTimes []timer.Time `view:"-"`

	// timers for each major function (step of processing)
	FunTimes map[string]*timer.Time `view:"-"`

	// network-level wait group for synchronizing threaded layer calls
	WaitGp sync.WaitGroup `view:"-"`
}

// InitName MUST be called to initialize the network's pointer to itself as an emer.Network
// which enables the proper interface methods to be called.  Also sets the name.
func (nt *NetworkBase) InitName(net emer.Network, name string) {
	nt.EmerNet = net
	nt.Nm = name
}

// emer.Network interface methods:
func (nt *NetworkBase) Name() string                   { return nt.Nm }
func (nt *NetworkBase) Label() string                  { return nt.Nm }
func (nt *NetworkBase) NLayers() int                   { return len(nt.Layers) }
func (nt *NetworkBase) Layer(idx int) emer.Layer       { return nt.Layers[idx] }
func (nt *NetworkBase) Bounds() (min, max math32.Vec3) { min = nt.MinPos; max = nt.MaxPos; return }
func (nt *NetworkBase) MaxParallelData() int           { return 1 }
func (nt *NetworkBase) NParallelData() int             { return 1 }

// LayerByName returns a layer by looking it up by name in the layer map (nil if not found).
// Will create the layer map if it is nil or a different size than layers slice,
// but otherwise needs to be updated manually.
func (nt *NetworkBase) LayerByName(name string) emer.Layer {
	if nt.LayMap == nil || len(nt.LayMap) != len(nt.Layers) {
		nt.MakeLayMap()
	}
	ly := nt.LayMap[name]
	return ly
}

// LayerByNameTry returns a layer by looking it up by name -- emits a log error message
// if layer is not found
func (nt *NetworkBase) LayerByNameTry(name string) (emer.Layer, error) {
	ly := nt.LayerByName(name)
	if ly == nil {
		err := fmt.Errorf("Layer named: %v not found in Network: %v", name, nt.Nm)
		log.Println(err)
		return ly, err
	}
	return ly, nil
}

// MakeLayMap updates layer map based on current layers
func (nt *NetworkBase) MakeLayMap() {
	nt.LayMap = make(map[string]emer.Layer, len(nt.Layers))
	for _, ly := range nt.Layers {
		nt.LayMap[ly.Name()] = ly
	}
}

// PrjnByNameTry returns a Prjn by looking it up by name in the list of projections
// (nil if not found).
func (nt *NetworkBase) PrjnByNameTry(name string) (emer.Prjn, error) {
	for _, ly := range nt.Layers {
		for i := range ly.NRecvPrjns() {
			pj := ly.RecvPrjn(i)
			if name == pj.Name() {
				return pj, nil
			}
		}
	}
	return nil, fmt.Errorf("could not find prjn with name %q", name)
}

// LayersByClass returns a list of layer names by given class(es).
// Lists are compiled when network Build() function called.
// The layer Type is always included as a Class, along with any other
// space-separated strings specified in Class for parameter styling, etc.
// If no classes are passed, all layer names in order are returned.
func (nt *NetworkBase) LayersByClass(classes ...string) []string {
	var nms []string
	hasName := map[string]bool{}
	if len(classes) == 0 {
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			nm := ly.Name()
			if !hasName[nm] {
				hasName[nm] = true
				nms = append(nms, nm)
			}
		}
		return nms
	}
	for _, lc := range classes {
		ns := nt.LayClassMap[lc]
		for _, nm := range ns {
			if !hasName[nm] {
				hasName[nm] = true
				nms = append(nms, nm)
			}
		}
	}
	return nms
}

// BuildThreads constructs the layer thread allocation based on Thread setting in the layers
func (nt *NetworkBase) BuildThreads() {
	nthr := 0
	for _, lyi := range nt.Layers {
		if lyi.IsOff() {
			continue
		}
		ly := lyi.(LeabraLayer).AsLeabra()
		nthr = max(nthr, ly.Thr)
	}
	nt.NThreads = nthr + 1
	nt.ThrLay = make([][]emer.Layer, nt.NThreads)
	nt.ThrChans = make([]LayFunChan, nt.NThreads)
	nt.ThrTimes = make([]timer.Time, nt.NThreads)
	nt.FunTimes = make(map[string]*timer.Time)
	for _, lyi := range nt.Layers {
		if lyi.IsOff() {
			continue
		}
		ly := lyi.(LeabraLayer).AsLeabra()
		th := ly.Thr
		nt.ThrLay[th] = append(nt.ThrLay[th], ly)
	}
	for th := 0; th < nt.NThreads; th++ {
		if len(nt.ThrLay[th]) == 0 {
			log.Printf("Network BuildThreads: Network %v has no layers for thread: %v\n", nt.Nm, th)
		}
		nt.ThrChans[th] = make(LayFunChan)
	}
}

// StdVertLayout arranges layers in a standard vertical (z axis stack) layout, by setting
// the Rel settings
func (nt *NetworkBase) StdVertLayout() {
	lstnm := ""
	for li, ly := range nt.Layers {
		if li == 0 {
			ly.SetRelPos(relpos.Rel{Rel: relpos.NoRel})
			lstnm = ly.Name()
		} else {
			ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstnm, XAlign: relpos.Middle, YAlign: relpos.Front})
		}
	}
}

// Layout computes the 3D layout of layers based on their relative position settings
func (nt *NetworkBase) Layout() {
	for itr := 0; itr < 5; itr++ {
		var lstly emer.Layer
		for _, ly := range nt.Layers {
			rp := ly.RelPos()
			var oly emer.Layer
			if lstly != nil && rp.Rel == relpos.NoRel {
				oly = lstly
				ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstly.Name(), XAlign: relpos.Middle, YAlign: relpos.Front})
			} else {
				if rp.Other != "" {
					var err error
					oly, err = nt.LayerByNameTry(rp.Other)
					if err != nil {
						log.Println(err)
						continue
					}
				} else if lstly != nil {
					oly = lstly
					ly.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: lstly.Name(), XAlign: relpos.Middle, YAlign: relpos.Front})
				}
			}
			if oly != nil {
				ly.SetPos(rp.Pos(oly.Pos(), oly.Size(), ly.Size()))
			}
			lstly = ly
		}
	}
	nt.BoundsUpdate()
}

// BoundsUpdate updates the Min / Max display bounds for 3D display
func (nt *NetworkBase) BoundsUpdate() {
	mn := math32.V3Scalar(math32.Infinity)
	mx := math32.Vec3{}
	for _, ly := range nt.Layers {
		ps := ly.Pos()
		sz := ly.Size()
		ru := ps
		ru.X += sz.X
		ru.Y += sz.Y
		mn.SetMax(ps)
		mx.SetMax(ru)
	}
	nt.MaxPos = mn
	nt.MaxPos = mx
}

// ApplyParams applies given parameter style Sheet to layers and prjns in this network.
// Calls UpdateParams to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (nt *NetworkBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	for _, ly := range nt.Layers {
		app, err := ly.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
}

// NonDefaultParams returns a listing of all parameters in the Network that
// are not at their default values -- useful for setting param styles etc.
func (nt *NetworkBase) NonDefaultParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.NonDefaultParams()
		nds += nd
	}
	return nds
}

// AllParams returns a listing of all parameters in the Network.
func (nt *NetworkBase) AllParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.AllParams()
		nds += nd
	}
	return nds
}

// KeyLayerParams returns a listing for all layers in the network,
// of the most important layer-level params (specific to each algorithm).
func (nt *NetworkBase) KeyLayerParams() string {
	return "" // todo: implement!
}

// KeyPrjnParams returns a listing for all Recv projections in the network,
// of the most important projection-level params (specific to each algorithm).
func (nt *NetworkBase) KeyPrjnParams() string {
	return nt.AllWtScales()
}

// AllWtScales returns a listing of all WtScale parameters in the Network
// in all Layers, Recv projections.  These are among the most important
// and numerous of parameters (in larger networks) -- this helps keep
// track of what they all are set to.
func (nt *NetworkBase) AllWtScales() string {
	str := ""
	for _, lyi := range nt.Layers {
		if lyi.IsOff() {
			continue
		}
		ly := lyi.(LeabraLayer).AsLeabra()
		str += "\nLayer: " + ly.Name() + "\n"
		rpjn := ly.RcvPrjns
		for _, p := range rpjn {
			if p.IsOff() {
				continue
			}
			pj := p.(LeabraPrjn).AsLeabra()
			str += fmt.Sprintf("\t%23s\t\tAbs:\t%g\tRel:\t%g\n", pj.Name(), pj.WtScale.Abs, pj.WtScale.Rel)
		}
	}
	return str
}

// AddLayerInit is implementation routine that takes a given layer and
// adds it to the network, and initializes and configures it properly.
func (nt *NetworkBase) AddLayerInit(ly emer.Layer, name string, shape []int, typ emer.LayerType) {
	if nt.EmerNet == nil {
		log.Printf("Network EmerNet is nil -- you MUST call InitName on network, passing a pointer to the network to initialize properly!")
		return
	}
	ly.InitName(ly, name, nt.EmerNet)
	ly.Config(shape, typ)
	nt.Layers = append(nt.Layers, ly)
	nt.MakeLayMap()
}

// AddLayer adds a new layer with given name and shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential -- see
// AddLayer2D and 4D for convenience methods for those.  4D layers enable
// pool (unit-group) level inhibition in Leabra networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each unit
// group having 4 rows (Y) of 5 (X) units.
func (nt *NetworkBase) AddLayer(name string, shape []int, typ emer.LayerType) emer.Layer {
	ly := nt.EmerNet.(LeabraNetwork).NewLayer() // essential to use EmerNet interface here!
	nt.AddLayerInit(ly, name, shape, typ)
	return ly
}

// AddLayer2D adds a new layer with given name and 2D shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential.
func (nt *NetworkBase) AddLayer2D(name string, shapeY, shapeX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{shapeY, shapeX}, typ)
}

// AddLayer4D adds a new layer with given name and 4D shape to the network.
// 4D layers enable pool (unit-group) level inhibition in Leabra networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each pool
// having 4 rows (Y) of 5 (X) neurons.
func (nt *NetworkBase) AddLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, typ)
}

// ConnectLayerNames establishes a projection between two layers, referenced by name
// adding to the recv and send projection lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) ConnectLayerNames(send, recv string, pat prjn.Pattern, typ emer.PrjnType) (rlay, slay emer.Layer, pj emer.Prjn, err error) {
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

// ConnectLayers establishes a projection between two layers,
// adding to the recv and send projection lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) ConnectLayers(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType) emer.Prjn {
	pj := nt.EmerNet.(LeabraNetwork).NewPrjn() // essential to use EmerNet interface here!
	return nt.ConnectLayersPrjn(send, recv, pat, typ, pj)
}

// ConnectLayersPrjn makes connection using given projection between two layers,
// adding given prjn to the recv and send projection lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) ConnectLayersPrjn(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.(LeabraPrjn).AsLeabra().Connect(send, recv, pat, typ)
	recv.(LeabraLayer).RecvPrjns().Add(pj.(LeabraPrjn))
	send.(LeabraLayer).SendPrjns().Add(pj.(LeabraPrjn))
	return pj
}

// BidirConnectLayerNames establishes bidirectional projections between two layers,
// referenced by name, with low = the lower layer that sends a Forward projection
// to the high layer, and receives a Back projection in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkBase) BidirConnectLayerNames(low, high string, pat prjn.Pattern) (lowlay, highlay emer.Layer, fwdpj, backpj emer.Prjn, err error) {
	lowlay, err = nt.LayerByNameTry(low)
	if err != nil {
		return
	}
	highlay, err = nt.LayerByNameTry(high)
	if err != nil {
		return
	}
	fwdpj = nt.ConnectLayers(lowlay, highlay, pat, emer.Forward)
	backpj = nt.ConnectLayers(highlay, lowlay, pat, emer.Back)
	return
}

// BidirConnectLayers establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) BidirConnectLayers(low, high emer.Layer, pat prjn.Pattern) (fwdpj, backpj emer.Prjn) {
	fwdpj = nt.ConnectLayers(low, high, pat, emer.Forward)
	backpj = nt.ConnectLayers(high, low, pat, emer.Back)
	return
}

// BidirConnectLayersPy establishes bidirectional projections between two layers,
// with low = lower layer that sends a Forward projection to the high layer,
// and receives a Back projection in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
// Py = python version with no return vals.
func (nt *NetworkBase) BidirConnectLayersPy(low, high emer.Layer, pat prjn.Pattern) {
	nt.ConnectLayers(low, high, pat, emer.Forward)
	nt.ConnectLayers(high, low, pat, emer.Back)
}

// LateralConnectLayer establishes a self-projection within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayer(lay emer.Layer, pat prjn.Pattern) emer.Prjn {
	pj := nt.EmerNet.(LeabraNetwork).NewPrjn() // essential to use EmerNet interface here!
	return nt.LateralConnectLayerPrjn(lay, pat, pj)
}

// LateralConnectLayerPrjn makes lateral self-projection using given projection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkBase) LateralConnectLayerPrjn(lay emer.Layer, pat prjn.Pattern, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.(LeabraPrjn).AsLeabra().Connect(lay, lay, pat, emer.Lateral)
	lay.(LeabraLayer).RecvPrjns().Add(pj.(LeabraPrjn))
	lay.(LeabraLayer).SendPrjns().Add(pj.(LeabraPrjn))
	return pj
}

// Build constructs the layer and projection state based on the layer shapes
// and patterns of interconnectivity
func (nt *NetworkBase) Build() error {
	nt.StopThreads() // any existing..
	nt.LayClassMap = make(map[string][]string)
	emsg := ""
	for li, ly := range nt.Layers {
		ly.SetIndex(li)
		if ly.IsOff() {
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

//////////////////////////////////////////////////////////////////////////////////////
//  Weights File

// SaveWtsJSON saves network weights (and any other state that adapts with learning)
// to a JSON-formatted file.  If filename has .gz extension, then file is gzip compressed.
func (nt *NetworkBase) SaveWtsJSON(filename core.Filename) error {
	fp, err := os.Create(string(filename))
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return err
	}
	ext := filepath.Ext(string(filename))
	if ext == ".gz" {
		gzr := gzip.NewWriter(fp)
		err = nt.WriteWtsJSON(gzr)
		gzr.Close()
	} else {
		bw := bufio.NewWriter(fp)
		err = nt.WriteWtsJSON(bw)
		bw.Flush()
	}
	return err
}

// OpenWtsJSON opens network weights (and any other state that adapts with learning)
// from a JSON-formatted file.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *NetworkBase) OpenWtsJSON(filename core.Filename) error {
	fp, err := os.Open(string(filename))
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return err
	}
	ext := filepath.Ext(string(filename))
	if ext == ".gz" {
		gzr, err := gzip.NewReader(fp)
		defer gzr.Close()
		if err != nil {
			log.Println(err)
			return err
		}
		return nt.ReadWtsJSON(gzr)
	} else {
		return nt.ReadWtsJSON(bufio.NewReader(fp))
	}
}

// todo: proper error handling here!

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (nt *NetworkBase) WriteWtsJSON(w io.Writer) error {
	depth := 0
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Network\": %q,\n", nt.Nm))) // note: can't use \n in `` so need "
	w.Write(indent.TabBytes(depth))
	onls := make([]emer.Layer, 0, len(nt.Layers))
	for _, ly := range nt.Layers {
		if !ly.IsOff() {
			onls = append(onls, ly)
		}
	}
	nl := len(onls)
	if nl == 0 {
		w.Write([]byte("\"Layers\": null\n"))
	} else {
		w.Write([]byte("\"Layers\": [\n"))
		depth++
		for li, ly := range onls {
			ly.WriteWtsJSON(w, depth)
			if li == nl-1 {
				w.Write([]byte("\n"))
			} else {
				w.Write([]byte(",\n"))
			}
		}
		depth--
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("]\n"))
	}
	depth--
	w.Write(indent.TabBytes(depth))
	_, err := w.Write([]byte("}\n"))
	return err
}

// ReadWtsJSON reads network weights from the receiver-side perspective
// in a JSON text format.  Reads entire file into a temporary weights.Weights
// structure that is then passed to Layers etc using SetWts method.
func (nt *NetworkBase) ReadWtsJSON(r io.Reader) error {
	nw, err := weights.NetReadJSON(r)
	if err != nil {
		return err // note: already logged
	}
	err = nt.SetWts(nw)
	if err != nil {
		log.Println(err)
	}
	return err
}

// SetWts sets the weights for this network from weights.Network decoded values
func (nt *NetworkBase) SetWts(nw *weights.Network) error {
	var err error
	if nw.Network != "" {
		nt.Nm = nw.Network
	}
	if nw.MetaData != nil {
		if nt.MetaData == nil {
			nt.MetaData = nw.MetaData
		} else {
			for mk, mv := range nw.MetaData {
				nt.MetaData[mk] = mv
			}
		}
	}
	for li := range nw.Layers {
		lw := &nw.Layers[li]
		ly, er := nt.LayerByNameTry(lw.Layer)
		if er != nil {
			err = er
			continue
		}
		ly.SetWts(lw)
	}
	return err
}

// OpenWtsCpp opens network weights (and any other state that adapts with learning)
// from old C++ emergent format.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *NetworkBase) OpenWtsCpp(filename core.Filename) error {
	fp, err := os.Open(string(filename))
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return err
	}
	ext := filepath.Ext(string(filename))
	if ext == ".gz" {
		gzr, err := gzip.NewReader(fp)
		defer gzr.Close()
		if err != nil {
			log.Println(err)
			return err
		}
		return nt.ReadWtsCpp(gzr)
	} else {
		return nt.ReadWtsCpp(fp)
	}
}

// ReadWtsCpp reads the weights from old C++ emergent format.
// Reads entire file into a temporary weights.Weights
// structure that is then passed to Layers etc using SetWts method.
func (nt *NetworkBase) ReadWtsCpp(r io.Reader) error {
	nw, err := weights.NetReadCpp(r)
	if err != nil {
		return err // note: already logged
	}
	err = nt.SetWts(nw)
	if err != nil {
		log.Println(err)
	}
	return err
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (nt *NetworkBase) VarRange(varNm string) (min, max float32, err error) {
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

//////////////////////////////////////////////////////////////////////////////////////
//  Threading infrastructure

// StartThreads starts up the computation threads, which monitor the channels for work
func (nt *NetworkBase) StartThreads() {
	fmt.Printf("NThreads: %d\tgo max procs: %d\tnum cpu:%d\n", nt.NThreads, runtime.GOMAXPROCS(0), runtime.NumCPU())
	for th := 0; th < nt.NThreads; th++ {
		go nt.ThrWorker(th) // start the worker thread for this channel
	}
}

// StopThreads stops the computation threads
func (nt *NetworkBase) StopThreads() {
	for th := 0; th < nt.NThreads; th++ {
		close(nt.ThrChans[th])
	}
}

// ThrWorker is the worker function run by the worker threads
func (nt *NetworkBase) ThrWorker(tt int) {
	if nt.LockThreads {
		runtime.LockOSThread()
	}
	for fun := range nt.ThrChans[tt] {
		thly := nt.ThrLay[tt]
		nt.ThrTimes[tt].Start()
		for _, ly := range thly {
			if ly.IsOff() {
				continue
			}
			fun(ly.(LeabraLayer))
		}
		nt.ThrTimes[tt].Stop()
		nt.WaitGp.Done()
	}
	if nt.LockThreads {
		runtime.UnlockOSThread()
	}
}

// ThrLayFun calls function on layer, using threaded (go routine worker) computation if NThreads > 1
// and otherwise just iterates over layers in the current thread.
func (nt *NetworkBase) ThrLayFun(fun func(ly LeabraLayer), funame string) {
	nt.FunTimerStart(funame)
	if nt.NThreads <= 1 {
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			fun(ly.(LeabraLayer))
		}
	} else {
		for th := 0; th < nt.NThreads; th++ {
			nt.WaitGp.Add(1)
			nt.ThrChans[th] <- fun
		}
		nt.WaitGp.Wait()
	}
	nt.FunTimerStop(funame)
}

// TimerReport reports the amount of time spent in each function, and in each thread
func (nt *NetworkBase) TimerReport() {
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

// ThrTimerReset resets the per-thread timers
func (nt *NetworkBase) ThrTimerReset() {
	for th := 0; th < nt.NThreads; th++ {
		nt.ThrTimes[th].Reset()
	}
}

// FunTimerStart starts function timer for given function name -- ensures creation of timer
func (nt *NetworkBase) FunTimerStart(fun string) {
	ft, ok := nt.FunTimes[fun]
	if !ok {
		ft = &timer.Time{}
		nt.FunTimes[fun] = ft
	}
	ft.Start()
}

// FunTimerStop stops function timer -- timer must already exist
func (nt *NetworkBase) FunTimerStop(fun string) {
	ft := nt.FunTimes[fun]
	ft.Stop()
}
