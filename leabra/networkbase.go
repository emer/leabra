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

	"cogentcore.org/core/core"
	"cogentcore.org/core/gox/indent"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/path"
	"github.com/emer/emergent/v2/timer"
	"github.com/emer/emergent/v2/weights"
)

// leabra.Network implements the Leabra algorithm, managing the Layers.
type Network struct {
	emer.NetworkBase

	// list of layers
	Layers []*Layer

	// number of parallel threads (go routines) to use.
	NThreads int `inactive:"+"`

	// timers for each major function (step of processing)
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
			pj := p.(LeabraPath).AsLeabra()
			str += fmt.Sprintf("\t%23s\t\tAbs:\t%g\tRel:\t%g\n", pj.Name(), pj.WtScale.Abs, pj.WtScale.Rel)
		}
	}
	return str
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
	pj.(LeabraPath).AsLeabra().Connect(send, recv, pat, typ)
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
	backpj = nt.ConnectLayers(highlay, lowlay, pat, emer.Back)
	return
}

// BidirConnectLayers establishes bidirectional pathways between two layers,
// with low = lower layer that sends a Forward pathway to the high layer,
// and receives a Back pathway in the opposite direction.
// Does not yet actually connect the units within the layers -- that
// requires Build.
func (nt *Network) BidirConnectLayers(low, high emer.Layer, pat path.Pattern) (fwdpj, backpj emer.Path) {
	fwdpj = nt.ConnectLayers(low, high, pat, emer.Forward)
	backpj = nt.ConnectLayers(high, low, pat, emer.Back)
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
	nt.ConnectLayers(high, low, pat, emer.Back)
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
	pj.(LeabraPath).AsLeabra().Connect(lay, lay, pat, emer.Lateral)
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

//////////////////////////////////////////////////////////////////////////////////////
//  Weights File

// SaveWeightsJSON saves network weights (and any other state that adapts with learning)
// to a JSON-formatted file.  If filename has .gz extension, then file is gzip compressed.
func (nt *Network) SaveWeightsJSON(filename core.Filename) error {
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

// OpenWeightsJSON opens network weights (and any other state that adapts with learning)
// from a JSON-formatted file.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *Network) OpenWeightsJSON(filename core.Filename) error {
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
func (nt *Network) WriteWtsJSON(w io.Writer) error {
	depth := 0
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Network\": %q,\n", nt.Nm))) // note: can't use \n in `` so need "
	w.Write(indent.TabBytes(depth))
	onls := make([]emer.Layer, 0, len(nt.Layers))
	for _, ly := range nt.Layers {
		if !ly.Off {
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
func (nt *Network) ReadWtsJSON(r io.Reader) error {
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
func (nt *Network) SetWts(nw *weights.Network) error {
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

// OpenWeightsCpp opens network weights (and any other state that adapts with learning)
// from old C++ emergent format.  If filename has .gz extension, then file is gzip uncompressed.
func (nt *Network) OpenWeightsCpp(filename core.Filename) error {
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
func (nt *Network) ReadWtsCpp(r io.Reader) error {
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

//////////////////////////////////////////////////////////////////////////////////////
//  Threading infrastructure

// StartThreads starts up the computation threads, which monitor the channels for work
func (nt *Network) StartThreads() {
	fmt.Printf("NThreads: %d\tgo max procs: %d\tnum cpu:%d\n", nt.NThreads, runtime.GOMAXPROCS(0), runtime.NumCPU())
	for th := 0; th < nt.NThreads; th++ {
		go nt.ThrWorker(th) // start the worker thread for this channel
	}
}

// StopThreads stops the computation threads
func (nt *Network) StopThreads() {
	for th := 0; th < nt.NThreads; th++ {
		close(nt.ThrChans[th])
	}
}

// ThrWorker is the worker function run by the worker threads
func (nt *Network) ThrWorker(tt int) {
	if nt.LockThreads {
		runtime.LockOSThread()
	}
	for fun := range nt.ThrChans[tt] {
		thly := nt.ThrLay[tt]
		nt.ThrTimes[tt].Start()
		for _, ly := range thly {
			if ly.Off {
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
func (nt *Network) ThrLayFun(fun func(ly LeabraLayer), funame string) {
	nt.FunTimerStart(funame)
	if nt.NThreads <= 1 {
		for _, ly := range nt.Layers {
			if ly.Off {
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

// ThrTimerReset resets the per-thread timers
func (nt *Network) ThrTimerReset() {
	for th := 0; th < nt.NThreads; th++ {
		nt.ThrTimes[th].Reset()
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
