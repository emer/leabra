// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/timer"
	"github.com/emer/emergent/weights"
	"github.com/goki/gi/gi"
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ints"
	"github.com/goki/mat32"
)

// LayFunChan is a channel that runs LeabraLayer functions
type LayFunChan chan func(ly LeabraLayer)

// leabra.NetworkStru holds the basic structural components of a network (layers)
type NetworkStru struct {
	EmerNet  emer.Network          `copy:"-" json:"-" xml:"-" view:"-" desc:"we need a pointer to ourselves as an emer.Network, which can always be used to extract the true underlying type of object when network is embedded in other structs -- function receivers do not have this ability so this is necessary."`
	Nm       string                `desc:"overall name of network -- helps discriminate if there are multiple"`
	Layers   emer.Layers           `desc:"list of layers"`
	WtsFile  string                `desc:"filename of last weights file loaded or saved"`
	LayMap   map[string]emer.Layer `view:"-" desc:"map of name to layers -- layer names must be unique"`
	MinPos   mat32.Vec3            `view:"-" desc:"minimum display position in network"`
	MaxPos   mat32.Vec3            `view:"-" desc:"maximum display position in network"`
	MetaData map[string]string     `desc:"optional metadata that is saved in network weights files -- e.g., can indicate number of epochs that were trained, or any other information about this network that would be useful to save"`

	NThreads int                    `inactive:"+" desc:"number of parallel threads (go routines) to use -- this is computed directly from the Layers which you must explicitly allocate to different threads -- updated during Build of network"`
	ThrLay   [][]emer.Layer         `view:"-" inactive:"+" desc:"layers per thread -- outer group is threads and inner is layers operated on by that thread -- based on user-assigned threads, initialized during Build"`
	ThrChans []LayFunChan           `view:"-" desc:"layer function channels, per thread"`
	ThrTimes []timer.Time           `view:"-" desc:"timers for each thread, so you can see how evenly the workload is being distributed"`
	FunTimes map[string]*timer.Time `view:"-" desc:"timers for each major function (step of processing)"`
	WaitGp   sync.WaitGroup         `view:"-" desc:"network-level wait group for synchronizing threaded layer calls"`
}

// InitName MUST be called to initialize the network's pointer to itself as an emer.Network
// which enables the proper interface methods to be called.  Also sets the name.
func (nt *NetworkStru) InitName(net emer.Network, name string) {
	nt.EmerNet = net
	nt.Nm = name
}

// emer.Network interface methods:
func (nt *NetworkStru) Name() string                  { return nt.Nm }
func (nt *NetworkStru) Label() string                 { return nt.Nm }
func (nt *NetworkStru) NLayers() int                  { return len(nt.Layers) }
func (nt *NetworkStru) Layer(idx int) emer.Layer      { return nt.Layers[idx] }
func (nt *NetworkStru) Bounds() (min, max mat32.Vec3) { min = nt.MinPos; max = nt.MaxPos; return }

// LayerByName returns a layer by looking it up by name in the layer map (nil if not found).
// Will create the layer map if it is nil or a different size than layers slice,
// but otherwise needs to be updated manually.
func (nt *NetworkStru) LayerByName(name string) emer.Layer {
	if nt.LayMap == nil || len(nt.LayMap) != len(nt.Layers) {
		nt.MakeLayMap()
	}
	ly := nt.LayMap[name]
	return ly
}

// LayerByNameTry returns a layer by looking it up by name -- emits a log error message
// if layer is not found
func (nt *NetworkStru) LayerByNameTry(name string) (emer.Layer, error) {
	ly := nt.LayerByName(name)
	if ly == nil {
		err := fmt.Errorf("Layer named: %v not found in Network: %v\n", name, nt.Nm)
		log.Println(err)
		return ly, err
	}
	return ly, nil
}

// MakeLayMap updates layer map based on current layers
func (nt *NetworkStru) MakeLayMap() {
	nt.LayMap = make(map[string]emer.Layer, len(nt.Layers))
	for _, ly := range nt.Layers {
		nt.LayMap[ly.Name()] = ly
	}
}

// BuildThreads constructs the layer thread allocation based on Thread setting in the layers
func (nt *NetworkStru) BuildThreads() {
	nthr := 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		nthr = ints.MaxInt(nthr, ly.Thread())
	}
	nt.NThreads = nthr + 1
	nt.ThrLay = make([][]emer.Layer, nt.NThreads)
	nt.ThrChans = make([]LayFunChan, nt.NThreads)
	nt.ThrTimes = make([]timer.Time, nt.NThreads)
	nt.FunTimes = make(map[string]*timer.Time)
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		th := ly.Thread()
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
func (nt *NetworkStru) StdVertLayout() {
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
func (nt *NetworkStru) Layout() {
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
	nt.BoundsUpdt()
}

// BoundsUpdt updates the Min / Max display bounds for 3D display
func (nt *NetworkStru) BoundsUpdt() {
	mn := mat32.NewVec3Scalar(mat32.Infinity)
	mx := mat32.Vec3Zero
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
func (nt *NetworkStru) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
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
func (nt *NetworkStru) NonDefaultParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.NonDefaultParams()
		nds += nd
	}
	return nds
}

// AllParams returns a listing of all parameters in the Network.
func (nt *NetworkStru) AllParams() string {
	nds := ""
	for _, ly := range nt.Layers {
		nd := ly.AllParams()
		nds += nd
	}
	return nds
}

// AddLayerInit is implementation routine that takes a given layer and
// adds it to the network, and initializes and configures it properly.
func (nt *NetworkStru) AddLayerInit(ly emer.Layer, name string, shape []int, typ emer.LayerType) {
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
func (nt *NetworkStru) AddLayer(name string, shape []int, typ emer.LayerType) emer.Layer {
	ly := nt.EmerNet.NewLayer() // essential to use EmerNet interface here!
	nt.AddLayerInit(ly, name, shape, typ)
	return ly
}

// AddLayer2D adds a new layer with given name and 2D shape to the network.
// 2D and 4D layer shapes are generally preferred but not essential.
func (nt *NetworkStru) AddLayer2D(name string, shapeY, shapeX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{shapeY, shapeX}, typ)
}

// AddLayer4D adds a new layer with given name and 4D shape to the network.
// 4D layers enable pool (unit-group) level inhibition in Leabra networks, for example.
// shape is in row-major format with outer-most dimensions first:
// e.g., 4D 3, 2, 4, 5 = 3 rows (Y) of 2 cols (X) of pools, with each pool
// having 4 rows (Y) of 5 (X) neurons.
func (nt *NetworkStru) AddLayer4D(name string, nPoolsY, nPoolsX, nNeurY, nNeurX int, typ emer.LayerType) emer.Layer {
	return nt.AddLayer(name, []int{nPoolsY, nPoolsX, nNeurY, nNeurX}, typ)
}

// ConnectLayerNames establishes a projection between two layers, referenced by name
// adding to the recv and send projection lists on each side of the connection.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkStru) ConnectLayerNames(send, recv string, pat prjn.Pattern, typ emer.PrjnType) (rlay, slay emer.Layer, pj emer.Prjn, err error) {
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
func (nt *NetworkStru) ConnectLayers(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType) emer.Prjn {
	pj := nt.EmerNet.NewPrjn() // essential to use EmerNet interface here!
	return nt.ConnectLayersPrjn(send, recv, pat, typ, pj)
}

// ConnectLayersPrjn makes connection using given projection between two layers,
// adding given prjn to the recv and send projection lists on each side of the connection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkStru) ConnectLayersPrjn(send, recv emer.Layer, pat prjn.Pattern, typ emer.PrjnType, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.Connect(send, recv, pat, typ)
	recv.RecvPrjns().Add(pj)
	send.SendPrjns().Add(pj)
	return pj
}

// BidirConnectLayerNames establishes bidirectional projections between two layers,
// referenced by name, with low = the lower layer that sends a Forward projection
// to the high layer, and receives a Back projection in the opposite direction.
// Returns error if not successful.
// Does not yet actually connect the units within the layers -- that requires Build.
func (nt *NetworkStru) BidirConnectLayerNames(low, high string, pat prjn.Pattern) (lowlay, highlay emer.Layer, fwdpj, backpj emer.Prjn, err error) {
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
func (nt *NetworkStru) BidirConnectLayers(low, high emer.Layer, pat prjn.Pattern) (fwdpj, backpj emer.Prjn) {
	fwdpj = nt.ConnectLayers(low, high, pat, emer.Forward)
	backpj = nt.ConnectLayers(high, low, pat, emer.Back)
	return
}

// LateralConnectLayer establishes a self-projection within given layer.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkStru) LateralConnectLayer(lay emer.Layer, pat prjn.Pattern) emer.Prjn {
	pj := nt.EmerNet.NewPrjn() // essential to use EmerNet interface here!
	return nt.LateralConnectLayerPrjn(lay, pat, pj)
}

// LateralConnectLayerPrjn makes lateral self-projection using given projection.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *NetworkStru) LateralConnectLayerPrjn(lay emer.Layer, pat prjn.Pattern, pj emer.Prjn) emer.Prjn {
	pj.Init(pj)
	pj.Connect(lay, lay, pat, emer.Lateral)
	lay.RecvPrjns().Add(pj)
	lay.SendPrjns().Add(pj)
	return pj
}

// Build constructs the layer and projection state based on the layer shapes
// and patterns of interconnectivity
func (nt *NetworkStru) Build() error {
	nt.StopThreads() // any existing..
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
func (nt *NetworkStru) SaveWtsJSON(filename gi.FileName) error {
	fp, err := os.Create(string(filename))
	defer fp.Close()
	if err != nil {
		log.Println(err)
		return err
	}
	ext := filepath.Ext(string(filename))
	if ext == ".gz" {
		gzr := gzip.NewWriter(fp)
		defer gzr.Close()
		nt.WriteWtsJSON(gzr)
	} else {
		nt.WriteWtsJSON(fp)
	}
	return nil
}

// OpenWtsJSON opens network weights (and any other state that adapts with learning)
// from a JSON-formatted file.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *NetworkStru) OpenWtsJSON(filename gi.FileName) error {
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
		return nt.ReadWtsJSON(fp)
	}
}

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (nt *NetworkStru) WriteWtsJSON(w io.Writer) {
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
	w.Write([]byte("}\n"))
}

// ReadWtsJSON reads network weights from the receiver-side perspective
// in a JSON text format.  Reads entire file into a temporary weights.Weights
// structure that is then passed to Layers etc using SetWts method.
func (nt *NetworkStru) ReadWtsJSON(r io.Reader) error {
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
func (nt *NetworkStru) SetWts(nw *weights.Network) error {
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
func (nt *NetworkStru) OpenWtsCpp(filename gi.FileName) error {
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
func (nt *NetworkStru) ReadWtsCpp(r io.Reader) error {
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
func (nt *NetworkStru) VarRange(varNm string) (min, max float32, err error) {
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
func (nt *NetworkStru) StartThreads() {
	for th := 0; th < nt.NThreads; th++ {
		go nt.ThrWorker(th) // start the worker thread for this channel
	}
}

// StopThreads stops the computation threads
func (nt *NetworkStru) StopThreads() {
	for th := 0; th < nt.NThreads; th++ {
		close(nt.ThrChans[th])
	}
}

// ThrWorker is the worker function run by the worker threads
func (nt *NetworkStru) ThrWorker(tt int) {
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
}

// ThrLayFun calls function on layer, using threaded (go routine worker) computation if NThreads > 1
// and otherwise just iterates over layers in the current thread.
func (nt *NetworkStru) ThrLayFun(fun func(ly LeabraLayer), funame string) {
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
func (nt *NetworkStru) TimerReport() {
	fmt.Printf("TimerReport: %v, NThreads: %v\n", nt.Nm, nt.NThreads)
	fmt.Printf("\tFunction Name\tTotal Secs\tPct\n")
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
		fmt.Printf("\t%v \t%6.4g\t%6.4g\n", fn, pcts[i], 100*(pcts[i]/tot))
	}
	fmt.Printf("\tTotal   \t%6.4g\n", tot)

	if nt.NThreads <= 1 {
		return
	}
	fmt.Printf("\n\tThr\tTotal Secs\tPct\n")
	pcts = make([]float64, nt.NThreads)
	tot = 0.0
	for th := 0; th < nt.NThreads; th++ {
		pcts[th] = nt.ThrTimes[th].TotalSecs()
		tot += pcts[th]
	}
	for th := 0; th < nt.NThreads; th++ {
		fmt.Printf("\t%v \t%6.4g\t%6.4g\n", th, pcts[th], 100*(pcts[th]/tot))
	}
}

// ThrTimerReset resets the per-thread timers
func (nt *NetworkStru) ThrTimerReset() {
	for th := 0; th < nt.NThreads; th++ {
		nt.ThrTimes[th].Reset()
	}
}

// FunTimerStart starts function timer for given function name -- ensures creation of timer
func (nt *NetworkStru) FunTimerStart(fun string) {
	ft, ok := nt.FunTimes[fun]
	if !ok {
		ft = &timer.Time{}
		nt.FunTimes[fun] = ft
	}
	ft.Start()
}

// FunTimerStop stops function timer -- timer must already exist
func (nt *NetworkStru) FunTimerStop(fun string) {
	ft := nt.FunTimes[fun]
	ft.Stop()
}
