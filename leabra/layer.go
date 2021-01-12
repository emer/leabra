// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/bitflag"
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ints"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// leabra.Layer has parameters for running a basic rate-coded Leabra layer
type Layer struct {
	LayerStru
	Act     ActParams       `view:"add-fields" desc:"Activation parameters and methods for computing activations"`
	Inhib   InhibParams     `view:"add-fields" desc:"Inhibition parameters and methods for computing layer-level inhibition"`
	Learn   LearnNeurParams `view:"add-fields" desc:"Learning parameters and methods that operate at the neuron level"`
	Neurons []Neuron        `desc:"slice of neurons for this layer -- flat list of len = Shp.Len(). You must iterate over index and use pointer to modify values."`
	Pools   []Pool          `desc:"inhibition and other pooled, aggregate state variables -- flat list has at least of 1 for layer, and one for each sub-pool (unit group) if shape supports that (4D).  You must iterate over index and use pointer to modify values."`
	CosDiff CosDiffStats    `desc:"cosine difference between ActM, ActP stats"`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

// AsLeabra returns this layer as a leabra.Layer -- all derived layers must redefine
// this to return the base Layer type, so that the LeabraLayer interface does not
// need to include accessors to all the basic stuff
func (ly *Layer) AsLeabra() *Layer {
	return ly
}

func (ly *Layer) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Inhib.Layer.On = true
	for _, pj := range ly.RcvPrjns {
		pj.Defaults()
	}
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *Layer) UpdateParams() {
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()
	for _, pj := range ly.RcvPrjns {
		pj.UpdateParams()
	}
}

// JsonToParams reformates json output to suitable params display output
func JsonToParams(b []byte) string {
	br := strings.Replace(string(b), `"`, ``, -1)
	br = strings.Replace(br, ",\n", "", -1)
	br = strings.Replace(br, "{\n", "{", -1)
	br = strings.Replace(br, "} ", "}\n  ", -1)
	br = strings.Replace(br, "\n }", " }", -1)
	br = strings.Replace(br, "\n  }\n", " }", -1)
	return br[1:] + "\n"
}

// AllParams returns a listing of all parameters in the Layer
func (ly *Layer) AllParams() string {
	str := "/////////////////////////////////////////////////\nLayer: " + ly.Nm + "\n"
	b, _ := json.MarshalIndent(&ly.Act, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)
	for _, pj := range ly.RcvPrjns {
		pstr := pj.AllParams()
		str += pstr
	}
	return str
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *Layer) UnitVarNames() []string {
	return NeuronVars
}

// UnitVarProps returns properties for variables
func (ly *Layer) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to *this layer's* UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	return NeuronVarIdxByName(varNm)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return len(NeuronVars)
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	if varIdx < 0 || varIdx >= ly.UnitVarNum() {
		return math32.NaN()
	}
	nrn := &ly.Neurons[idx]
	return nrn.VarByIndex(varIdx)
}

// UnitVals fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *Layer) UnitVals(vals *[]float32, varNm string) error {
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	vidx, err := ly.LeabraLay.UnitVarIdx(varNm)
	if err != nil {
		nan := math32.NaN()
		for i := range ly.Neurons {
			(*vals)[i] = nan
		}
		return err
	}
	for i := range ly.Neurons {
		(*vals)[i] = ly.LeabraLay.UnitVal1D(vidx, i)
	}
	return nil
}

// UnitValsTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *Layer) UnitValsTensor(tsr etensor.Tensor, varNm string) error {
	if tsr == nil {
		err := fmt.Errorf("leabra.UnitValsTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	tsr.SetShape(ly.Shp.Shp, ly.Shp.Strd, ly.Shp.Nms)
	vidx, err := ly.LeabraLay.UnitVarIdx(varNm)
	if err != nil {
		nan := math.NaN()
		for i := range ly.Neurons {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i := range ly.Neurons {
		v := ly.LeabraLay.UnitVal1D(vidx, i)
		if math32.IsNaN(v) {
			tsr.SetFloat1D(1, math.NaN())
		} else {
			tsr.SetFloat1D(i, float64(v))
		}
	}
	return nil
}

// UnitVal returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *Layer) UnitVal(varNm string, idx []int) float32 {
	vidx, err := ly.LeabraLay.UnitVarIdx(varNm)
	if err != nil {
		return math32.NaN()
	}
	fidx := ly.Shp.Offset(idx)
	return ly.LeabraLay.UnitVal1D(vidx, fidx)
}

// RecvPrjnVals fills in values of given synapse variable name,
// for projection into given sending layer and neuron 1D index,
// for all receiving neurons in this layer,
// into given float32 slice (only resized if not big enough).
// prjnType is the string representation of the prjn type -- used if non-empty,
// useful when there are multiple projections between two layers.
// Returns error on invalid var name.
// If the receiving neuron is not connected to the given sending layer or neuron
// then the value is set to math32.NaN().
// Returns error on invalid var name or lack of recv prjn (vals always set to nan on prjn err).
func (ly *Layer) RecvPrjnVals(vals *[]float32, varNm string, sendLay emer.Layer, sendIdx1D int, prjnType string) error {
	var err error
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := math32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if sendLay == nil {
		return fmt.Errorf("sending layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = sendLay.SendPrjns().RecvNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
		}
	} else {
		pj, err = sendLay.SendPrjns().RecvNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	if pj.IsOff() {
		return fmt.Errorf("projection is off")
	}
	for ri := 0; ri < nn; ri++ {
		(*vals)[ri] = pj.SynVal(varNm, sendIdx1D, ri) // this will work with any variable -- slower, but necessary
	}
	return nil
}

// SendPrjnVals fills in values of given synapse variable name,
// for projection into given receiving layer and neuron 1D index,
// for all sending neurons in this layer,
// into given float32 slice (only resized if not big enough).
// prjnType is the string representation of the prjn type -- used if non-empty,
// useful when there are multiple projections between two layers.
// Returns error on invalid var name.
// If the sending neuron is not connected to the given receiving layer or neuron
// then the value is set to math32.NaN().
// Returns error on invalid var name or lack of recv prjn (vals always set to nan on prjn err).
func (ly *Layer) SendPrjnVals(vals *[]float32, varNm string, recvLay emer.Layer, recvIdx1D int, prjnType string) error {
	var err error
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	nan := math32.NaN()
	for i := 0; i < nn; i++ {
		(*vals)[i] = nan
	}
	if recvLay == nil {
		return fmt.Errorf("receiving layer is nil")
	}
	var pj emer.Prjn
	if prjnType != "" {
		pj, err = recvLay.RecvPrjns().SendNameTypeTry(ly.Nm, prjnType)
		if pj == nil {
			pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
		}
	} else {
		pj, err = recvLay.RecvPrjns().SendNameTry(ly.Nm)
	}
	if pj == nil {
		return err
	}
	if pj.IsOff() {
		return fmt.Errorf("projection is off")
	}
	for si := 0; si < nn; si++ {
		(*vals)[si] = pj.SynVal(varNm, si, recvIdx1D)
	}
	return nil
}

// Pool returns pool at given index
func (ly *Layer) Pool(idx int) *Pool {
	return &(ly.Pools[idx])
}

// PoolTry returns pool at given index, returns error if index is out of range
func (ly *Layer) PoolTry(idx int) (*Pool, error) {
	np := len(ly.Pools)
	if idx < 0 || idx >= np {
		return nil, fmt.Errorf("Layer Pool index: %v out of range, N = %v", idx, np)
	}
	return &(ly.Pools[idx]), nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Build

// BuildSubPools initializes neuron start / end indexes for sub-pools
func (ly *Layer) BuildSubPools() {
	if !ly.Is4D() {
		return
	}
	sh := ly.Shp.Shapes()
	spy := sh[0]
	spx := sh[1]
	pi := 1
	for py := 0; py < spy; py++ {
		for px := 0; px < spx; px++ {
			soff := ly.Shp.Offset([]int{py, px, 0, 0})
			eoff := ly.Shp.Offset([]int{py, px, sh[2] - 1, sh[3] - 1}) + 1
			pl := &ly.Pools[pi]
			pl.StIdx = soff
			pl.EdIdx = eoff
			for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
				nrn := &ly.Neurons[ni]
				nrn.SubPool = int32(pi)
			}
			pi++
		}
	}
}

// BuildPools builds the inhibitory pools structures -- nu = number of units in layer
func (ly *Layer) BuildPools(nu int) error {
	np := 1 + ly.NPools()
	ly.Pools = make([]Pool, np)
	lpl := &ly.Pools[0]
	lpl.StIdx = 0
	lpl.EdIdx = nu
	if np > 1 {
		ly.BuildSubPools()
	}
	return nil
}

// BuildPrjns builds the projections, recv-side
func (ly *Layer) BuildPrjns() error {
	emsg := ""
	for _, pj := range ly.RcvPrjns {
		if pj.IsOff() {
			continue
		}
		err := pj.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
	}
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// Build constructs the layer state, including calling Build on the projections
func (ly *Layer) Build() error {
	nu := ly.Shp.Len()
	if nu == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Nm)
	}
	ly.Neurons = make([]Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPrjns()
	return err
}

// WriteWtsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (ly *Layer) WriteWtsJSON(w io.Writer, depth int) {
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Layer\": %q,\n", ly.Nm)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"MetaData\": {\n")))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"ActMAvg\": \"%g\",\n", ly.Pools[0].ActAvg.ActMAvg)))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"ActPAvg\": \"%g\"\n", ly.Pools[0].ActAvg.ActPAvg)))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("},\n"))
	w.Write(indent.TabBytes(depth))
	onps := make(emer.Prjns, 0, len(ly.RcvPrjns))
	for _, pj := range ly.RcvPrjns {
		if !pj.IsOff() {
			onps = append(onps, pj)
		}
	}
	np := len(onps)
	if np == 0 {
		w.Write([]byte(fmt.Sprintf("\"Prjns\": null\n")))
	} else {
		w.Write([]byte(fmt.Sprintf("\"Prjns\": [\n")))
		depth++
		for pi, pj := range onps {
			pj.WriteWtsJSON(w, depth) // this leaves prjn unterminated
			if pi == np-1 {
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
	w.Write([]byte("}")) // note: leave unterminated as outer loop needs to add , or just \n depending
}

// ReadWtsJSON reads the weights from this layer from the receiver-side perspective
// in a JSON text format.  This is for a set of weights that were saved *for one layer only*
// and is not used for the network-level ReadWtsJSON, which reads into a separate
// structure -- see SetWts method.
func (ly *Layer) ReadWtsJSON(r io.Reader) error {
	lw, err := weights.LayReadJSON(r)
	if err != nil {
		return err // note: already logged
	}
	return ly.SetWts(lw)
}

// SetWts sets the weights for this layer from weights.Layer decoded values
func (ly *Layer) SetWts(lw *weights.Layer) error {
	if ly.IsOff() {
		return nil
	}
	if lw.MetaData != nil {
		if am, ok := lw.MetaData["ActMAvg"]; ok {
			pv, _ := strconv.ParseFloat(am, 32)
			ly.Pools[0].ActAvg.ActMAvg = float32(pv)
		}
		if ap, ok := lw.MetaData["ActPAvg"]; ok {
			pv, _ := strconv.ParseFloat(ap, 32)
			pl := &ly.Pools[0]
			pl.ActAvg.ActPAvg = float32(pv)
			ly.Inhib.ActAvg.EffFmAvg(&pl.ActAvg.ActPAvgEff, pl.ActAvg.ActPAvg)
		}
	}
	var err error
	rpjs := ly.RecvPrjns()
	if len(lw.Prjns) == len(*rpjs) { // this is essential if multiple prjns from same layer
		for pi := range lw.Prjns {
			pw := &lw.Prjns[pi]
			pj := (*rpjs)[pi]
			er := pj.SetWts(pw)
			if er != nil {
				err = er
			}
		}
	} else {
		for pi := range lw.Prjns {
			pw := &lw.Prjns[pi]
			pj := rpjs.SendName(pw.From)
			if pj != nil {
				er := pj.SetWts(pw)
				if er != nil {
					err = er
				}
			}
		}
	}
	return err
}

// VarRange returns the min / max values for given variable
// todo: support r. s. projection values
func (ly *Layer) VarRange(varNm string) (min, max float32, err error) {
	sz := len(ly.Neurons)
	if sz == 0 {
		return
	}
	vidx := 0
	vidx, err = NeuronVarIdxByName(varNm)
	if err != nil {
		return
	}

	v0 := ly.Neurons[0].VarByIndex(vidx)
	min = v0
	max = v0
	for i := 1; i < sz; i++ {
		vl := ly.Neurons[i].VarByIndex(vidx)
		if vl < min {
			min = vl
		}
		if vl > max {
			max = vl
		}
	}
	return
}

// note: all basic computation can be performed on layer-level and prjn level

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes the weight values in the network, i.e., resetting learning
// Also calls InitActs
func (ly *Layer) InitWts() {
	ly.LeabraLay.UpdateParams()
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).InitWts()
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.ActAvg.ActMAvg = ly.Inhib.ActAvg.Init
		pl.ActAvg.ActPAvg = ly.Inhib.ActAvg.Init
		pl.ActAvg.ActPAvgEff = ly.Inhib.ActAvg.EffInit()
	}
	ly.LeabraLay.InitActAvg()
	ly.LeabraLay.InitActs()
	ly.CosDiff.Init()
}

// InitActAvg initializes the running-average activation values that drive learning.
func (ly *Layer) InitActAvg() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Learn.InitActAvg(nrn)
	}
}

// InitActs fully initializes activation state -- only called automatically during InitWts
func (ly *Layer) InitActs() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Act.InitActs(nrn)
	}
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Init()
		pl.ActM.Init()
		pl.ActP.Init()
	}
}

// InitWtsSym initializes the weight symmetry -- higher layers copy weights from lower layers
func (ly *Layer) InitWtSym() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		plp := p.(LeabraPrjn)
		if !(plp.AsLeabra().WtInit.Sym) {
			continue
		}
		// key ordering constraint on which way weights are copied
		if p.RecvLay().Index() < p.SendLay().Index() {
			continue
		}
		rpj, has := ly.RecipToSendPrjn(p)
		if !has {
			continue
		}
		rlp := rpj.(LeabraPrjn)
		if !(rlp.AsLeabra().WtInit.Sym) {
			continue
		}
		plp.InitWtSym(rlp)
	}
}

// InitExt initializes external input state -- called prior to apply ext
func (ly *Layer) InitExt() {
	msk := bitflag.Mask32(int(NeurHasExt), int(NeurHasTarg), int(NeurHasCmpr))
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Ext = 0
		nrn.Targ = 0
		nrn.ClearMask(msk)
	}
}

// ApplyExtFlags gets the clear mask and set mask for updating neuron flags
// based on layer type, and whether input should be applied to Targ (else Ext)
func (ly *Layer) ApplyExtFlags() (clrmsk, setmsk int32, toTarg bool) {
	clrmsk = bitflag.Mask32(int(NeurHasExt), int(NeurHasTarg), int(NeurHasCmpr))
	toTarg = false
	if ly.Typ == emer.Target {
		setmsk = bitflag.Mask32(int(NeurHasTarg))
		toTarg = true
	} else if ly.Typ == emer.Compare {
		setmsk = bitflag.Mask32(int(NeurHasCmpr))
		toTarg = true
	} else {
		setmsk = bitflag.Mask32(int(NeurHasExt))
	}
	return
}

// ApplyExt applies external input in the form of an etensor.Float32.  If
// dimensionality of tensor matches that of layer, and is 2D or 4D, then each dimension
// is iterated separately, so any mismatch preserves dimensional structure.
// Otherwise, the flat 1D view of the tensor is used.
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt(ext etensor.Tensor) {
	switch {
	case ext.NumDims() == 2 && ly.Shp.NumDims() == 4: // special case
		ly.ApplyExt2Dto4D(ext)
	case ext.NumDims() != ly.Shp.NumDims() || !(ext.NumDims() == 2 || ext.NumDims() == 4):
		ly.ApplyExt1DTsr(ext)
	case ext.NumDims() == 2:
		ly.ApplyExt2D(ext)
	case ext.NumDims() == 4:
		ly.ApplyExt4D(ext)
	}
}

// ApplyExt2D applies 2D tensor external input
func (ly *Layer) ApplyExt2D(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	ymx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
	xmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			vl := float32(ext.FloatVal(idx))
			i := ly.Shp.Offset(idx)
			nrn := &ly.Neurons[i]
			if nrn.IsOff() {
				continue
			}
			if toTarg {
				nrn.Targ = vl
			} else {
				nrn.Ext = vl
			}
			nrn.ClearMask(clrmsk)
			nrn.SetMask(setmsk)
		}
	}
}

// ApplyExt2Dto4D applies 2D tensor external input to a 4D layer
func (ly *Layer) ApplyExt2Dto4D(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	lNy, lNx, _, _ := etensor.Prjn2DShape(&ly.Shp, false)

	ymx := ints.MinInt(ext.Dim(0), lNy)
	xmx := ints.MinInt(ext.Dim(1), lNx)
	for y := 0; y < ymx; y++ {
		for x := 0; x < xmx; x++ {
			idx := []int{y, x}
			vl := float32(ext.FloatVal(idx))
			ui := etensor.Prjn2DIdx(&ly.Shp, false, y, x)
			nrn := &ly.Neurons[ui]
			if nrn.IsOff() {
				continue
			}
			if toTarg {
				nrn.Targ = vl
			} else {
				nrn.Ext = vl
			}
			nrn.ClearMask(clrmsk)
			nrn.SetMask(setmsk)
		}
	}
}

// ApplyExt4D applies 4D tensor external input
func (ly *Layer) ApplyExt4D(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	ypmx := ints.MinInt(ext.Dim(0), ly.Shp.Dim(0))
	xpmx := ints.MinInt(ext.Dim(1), ly.Shp.Dim(1))
	ynmx := ints.MinInt(ext.Dim(2), ly.Shp.Dim(2))
	xnmx := ints.MinInt(ext.Dim(3), ly.Shp.Dim(3))
	for yp := 0; yp < ypmx; yp++ {
		for xp := 0; xp < xpmx; xp++ {
			for yn := 0; yn < ynmx; yn++ {
				for xn := 0; xn < xnmx; xn++ {
					idx := []int{yp, xp, yn, xn}
					vl := float32(ext.FloatVal(idx))
					i := ly.Shp.Offset(idx)
					nrn := &ly.Neurons[i]
					if nrn.IsOff() {
						continue
					}
					if toTarg {
						nrn.Targ = vl
					} else {
						nrn.Ext = vl
					}
					nrn.ClearMask(clrmsk)
					nrn.SetMask(setmsk)
				}
			}
		}
	}
}

// ApplyExt1DTsr applies external input using 1D flat interface into tensor.
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1DTsr(ext etensor.Tensor) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(ext.Len(), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := float32(ext.FloatVal1D(i))
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// ApplyExt1D applies external input in the form of a flat 1-dimensional slice of floats
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D(ext []float64) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := float32(ext[i])
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// ApplyExt1D32 applies external input in the form of a flat 1-dimensional slice of float32s.
// If the layer is a Target or Compare layer type, then it goes in Targ
// otherwise it goes in Ext
func (ly *Layer) ApplyExt1D32(ext []float32) {
	clrmsk, setmsk, toTarg := ly.ApplyExtFlags()
	mx := ints.MinInt(len(ext), len(ly.Neurons))
	for i := 0; i < mx; i++ {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		vl := ext[i]
		if toTarg {
			nrn.Targ = vl
		} else {
			nrn.Ext = vl
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (ly *Layer) UpdateExtFlags() {
	clrmsk, setmsk, _ := ly.ApplyExtFlags()
	for i := range ly.Neurons {
		nrn := &ly.Neurons[i]
		if nrn.IsOff() {
			continue
		}
		nrn.ClearMask(clrmsk)
		nrn.SetMask(setmsk)
	}
}

// AlphaCycInit handles all initialization at start of new input pattern, including computing
// input scaling from running average activation etc.
// should already have presented the external input to the network at this point.
func (ly *Layer) AlphaCycInit() {
	ly.LeabraLay.AvgLFmAvgM()
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		ly.Inhib.ActAvg.AvgFmAct(&pl.ActAvg.ActMAvg, pl.ActM.Avg)
		ly.Inhib.ActAvg.AvgFmAct(&pl.ActAvg.ActPAvg, pl.ActP.Avg)
		ly.Inhib.ActAvg.EffFmAvg(&pl.ActAvg.ActPAvgEff, pl.ActAvg.ActPAvg)
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.ActQ0 = nrn.ActP
	}
	ly.LeabraLay.GScaleFmAvgAct()
	if ly.Act.Noise.Type != NoNoise && ly.Act.Noise.Fixed && ly.Act.Noise.Dist != erand.Mean {
		ly.LeabraLay.GenNoise()
	}
	ly.LeabraLay.DecayState(ly.Act.Init.Decay)
	ly.LeabraLay.InitGInc()
	if ly.Act.Clamp.Hard && ly.Typ == emer.Input {
		ly.LeabraLay.HardClamp()
	}
}

// AvgLFmAvgM updates AvgL long-term running average activation that drives BCM Hebbian learning
func (ly *Layer) AvgLFmAvgM() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Learn.AvgLFmAvgM(nrn)
		if ly.Learn.AvgL.ErrMod {
			nrn.AvgLLrn *= ly.CosDiff.ModAvgLLrn
		}
	}
}

// GScaleFmAvgAct computes the scaling factor for synaptic input conductances G,
// based on sending layer average activation.
// This attempts to automatically adjust for overall differences in raw activity
// coming into the units to achieve a general target of around .5 to 1
// for the integrated Ge value.
func (ly *Layer) GScaleFmAvgAct() {
	totGeRel := float32(0)
	totGiRel := float32(0)
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(LeabraPrjn).AsLeabra()
		slay := p.SendLay().(LeabraLayer).AsLeabra()
		slpl := &slay.Pools[0]
		savg := slpl.ActAvg.ActPAvgEff
		snu := len(slay.Neurons)
		ncon := pj.RConNAvgMax.Avg
		pj.GScale = pj.WtScale.FullScale(savg, float32(snu), ncon)
		// reverting this change: if you want to eliminate a prjn, set the Off flag
		// if you want to negate it but keep the relative factor in the denominator
		// then set the scale to 0.
		// if pj.GScale == 0 {
		// 	continue
		// }
		if pj.Typ == emer.Inhib {
			totGiRel += pj.WtScale.Rel
		} else {
			totGeRel += pj.WtScale.Rel
		}
	}

	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(LeabraPrjn).AsLeabra()
		if pj.Typ == emer.Inhib {
			if totGiRel > 0 {
				pj.GScale /= totGiRel
			}
		} else {
			if totGeRel > 0 {
				pj.GScale /= totGeRel
			}
		}
	}
}

// GenNoise generates random noise for all neurons
func (ly *Layer) GenNoise() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Noise = float32(ly.Act.Noise.Gen(-1))
	}
}

// DecayState decays activation state by given proportion (default is on ly.Act.Init.Decay).
// This does *not* call InitGInc -- must call that separately at start of AlphaCyc
func (ly *Layer) DecayState(decay float32) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.DecayState(nrn, decay)
	}
	for pi := range ly.Pools { // decaying average act is essential for inhib
		pl := &ly.Pools[pi]
		pl.Inhib.Decay(decay)
	}
}

// DecayStatePool decays activation state by given proportion in given sub-pool index (0 based)
func (ly *Layer) DecayStatePool(pool int, decay float32) {
	pi := int32(pool + 1) // 1 based
	pl := &ly.Pools[pi]
	for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.DecayState(nrn, decay)
	}
	pl.Inhib.Decay(decay)
}

// HardClamp hard-clamps the activations in the layer -- called during AlphaCycInit for hard-clamped Input layers
func (ly *Layer) HardClamp() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.HardClamp(nrn)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// InitGinc initializes the Ge excitatory and Gi inhibitory conductance accumulation states
// including ActSent and G*Raw values.
// called at start of trial always, and can be called optionally
// when delta-based Ge computation needs to be updated (e.g., weights
// might have changed strength)
func (ly *Layer) InitGInc() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.InitGInc(nrn)
	}
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).InitGInc()
	}
}

// SendGDelta sends change in activation since last sent, to increment recv
// synaptic conductances G, if above thresholds
func (ly *Layer) SendGDelta(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.Act > ly.Act.OptThresh.Send {
			delta := nrn.Act - nrn.ActSent
			if math32.Abs(delta) > ly.Act.OptThresh.Delta {
				for _, sp := range ly.SndPrjns {
					if sp.IsOff() {
						continue
					}
					sp.(LeabraPrjn).SendGDelta(ni, delta)
				}
				nrn.ActSent = nrn.Act
			}
		} else if nrn.ActSent > ly.Act.OptThresh.Send {
			delta := -nrn.ActSent // un-send the last above-threshold activation to get back to 0
			for _, sp := range ly.SndPrjns {
				if sp.IsOff() {
					continue
				}
				sp.(LeabraPrjn).SendGDelta(ni, delta)
			}
			nrn.ActSent = 0
		}
	}
}

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *Layer) GFmInc(ltime *Time) {
	ly.RecvGInc(ltime)
	ly.GFmIncNeur(ltime)
}

// RecvGInc calls RecvGInc on receiving projections to collect Neuron-level G*Inc values.
// This is called by GFmInc overall method, but separated out for cases that need to
// do something different.
func (ly *Layer) RecvGInc(ltime *Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).RecvGInc()
	}
}

// GFmIncNeur is the neuron-level code for GFmInc that integrates overall Ge, Gi values
// from their G*Raw accumulators.
func (ly *Layer) GFmIncNeur(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		// note: each step broken out here so other variants can add extra terms to Raw
		ly.Act.GeFmRaw(nrn, nrn.GeRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
}

// AvgMaxGe computes the average and max Ge stats, used in inhibition
func (ly *Layer) AvgMaxGe(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Ge.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.Inhib.Ge.UpdateVal(nrn.Ge, ni)
		}
		pl.Inhib.Ge.CalcAvg()
	}
}

// InhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) InhibFmGeAct(ltime *Time) {
	lpl := &ly.Pools[0]
	ly.Inhib.Layer.Inhib(&lpl.Inhib)
	ly.PoolInhibFmGeAct(ltime)
	ly.InhibFmPool(ltime)
}

// PoolInhibFmGeAct computes inhibition Gi from Ge and Act averages within relevant Pools
func (ly *Layer) PoolInhibFmGeAct(ltime *Time) {
	np := len(ly.Pools)
	if np == 1 {
		return
	}
	lpl := &ly.Pools[0]
	lyInhib := ly.Inhib.Layer.On
	for pi := 1; pi < np; pi++ {
		pl := &ly.Pools[pi]
		ly.Inhib.Pool.Inhib(&pl.Inhib)
		if lyInhib {
			pl.Inhib.LayGi = lpl.Inhib.Gi
			pl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi) // pool is max of layer
		} else {
			lpl.Inhib.Gi = math32.Max(pl.Inhib.Gi, lpl.Inhib.Gi) // update layer from pool
		}
	}
	if !lyInhib {
		lpl.Inhib.GiOrig = lpl.Inhib.Gi // effective GiOrig
	}
}

// InhibFmPool computes inhibition Gi from Pool-level aggregated inhibition, including self and syn
func (ly *Layer) InhibFmPool(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		pl := &ly.Pools[nrn.SubPool]
		ly.Inhib.Self.Inhib(&nrn.GiSelf, nrn.Act)
		nrn.Gi = pl.Inhib.Gi + nrn.GiSelf + nrn.GiSyn
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *Layer) ActFmG(ltime *Time) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Act.VmFmG(nrn)
		ly.Act.ActFmG(nrn)
		ly.Learn.AvgsFmAct(nrn)
	}
}

// AvgMaxAct computes the average and max Act stats, used in inhibition
func (ly *Layer) AvgMaxAct(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		pl.Inhib.Act.Init()
		for ni := pl.StIdx; ni < pl.EdIdx; ni++ {
			nrn := &ly.Neurons[ni]
			if nrn.IsOff() {
				continue
			}
			pl.Inhib.Act.UpdateVal(nrn.Act, ni)
		}
		pl.Inhib.Act.CalcAvg()
	}
}

// CyclePost is called after the standard Cycle update, as a separate
// network layer loop.
// This is reserved for any kind of special ad-hoc types that
// need to do something special after Act is finally computed.
// For example, sending a neuromodulatory signal such as dopamine.
func (ly *Layer) CyclePost(ltime *Time) {
}

//////////////////////////////////////////////////////////////////////////////////////
//  Quarter

// QuarterFinal does updating after end of a quarter
func (ly *Layer) QuarterFinal(ltime *Time) {
	for pi := range ly.Pools {
		pl := &ly.Pools[pi]
		switch ltime.Quarter {
		case 2:
			pl.ActM = pl.Inhib.Act
		case 3:
			pl.ActP = pl.Inhib.Act
		}
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		switch ltime.Quarter {
		case 0:
			nrn.ActQ1 = nrn.Act
		case 1:
			nrn.ActQ2 = nrn.Act
		case 2:
			nrn.ActM = nrn.Act
			if nrn.HasFlag(NeurHasTarg) { // will be clamped in plus phase
				nrn.Ext = nrn.Targ
				nrn.SetFlag(NeurHasExt)
			}
		case 3:
			nrn.ActP = nrn.Act
			nrn.ActDif = nrn.ActP - nrn.ActM
			nrn.ActAvg += ly.Act.Dt.AvgDt * (nrn.Act - nrn.ActAvg)
		}
	}
	if ltime.Quarter == 3 {
		ly.LeabraLay.CosDiffFmActs()
	}
}

// CosDiffFmActs computes the cosine difference in activation state between minus and plus phases.
// this is also used for modulating the amount of BCM hebbian learning
func (ly *Layer) CosDiffFmActs() {
	lpl := &ly.Pools[0]
	avgM := lpl.ActM.Avg
	avgP := lpl.ActP.Avg
	cosv := float32(0)
	ssm := float32(0)
	ssp := float32(0)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ap := nrn.ActP - avgP // zero mean
		am := nrn.ActM - avgM
		cosv += ap * am
		ssm += am * am
		ssp += ap * ap
	}

	dist := math32.Sqrt(ssm * ssp)
	if dist != 0 {
		cosv /= dist
	}
	ly.CosDiff.Cos = cosv

	ly.Learn.CosDiff.AvgVarFmCos(&ly.CosDiff.Avg, &ly.CosDiff.Var, ly.CosDiff.Cos)

	if ly.LeabraLay.IsTarget() {
		ly.CosDiff.AvgLrn = 0 // no BCM for non-hidden layers
		ly.CosDiff.ModAvgLLrn = 0
	} else {
		ly.CosDiff.AvgLrn = 1 - ly.CosDiff.Avg
		ly.CosDiff.ModAvgLLrn = ly.Learn.AvgL.ErrModFmLayErr(ly.CosDiff.AvgLrn)
	}
}

// IsTarget returns true if this layer is a Target layer.
// By default, returns true for layers of Type == emer.Target
// Other Target layers include the TRCLayer in deep predictive learning.
// This is used for turning off BCM hebbian learning,
// in CosDiffFmActs to set the CosDiff.ModAvgLLrn value
// for error-modulated level of hebbian learning.
// It is also used in WtBal to not apply it to target layers.
// In both cases, Target layers are purely error-driven.
func (ly *Layer) IsTarget() bool {
	return ly.Typ == emer.Target
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learning

// DWt computes the weight change (learning) -- calls DWt method on sending projections
func (ly *Layer) DWt() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).DWt()
	}
}

// WtFmDWt updates the weights from delta-weight changes -- on the sending projections
func (ly *Layer) WtFmDWt() {
	for _, p := range ly.SndPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).WtFmDWt()
	}
}

// WtBalFmWt computes the Weight Balance factors based on average recv weights
func (ly *Layer) WtBalFmWt() {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		p.(LeabraPrjn).WtBalFmWt()
	}
}

// LrateMult sets the new Lrate parameter for Prjns to LrateInit * mult.
// Useful for implementing learning rate schedules.
func (ly *Layer) LrateMult(mult float32) {
	for _, p := range ly.RcvPrjns {
		// if p.IsOff() { // keep all sync'd
		// 	continue
		// }
		p.(LeabraPrjn).LrateMult(mult)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Threading / Reports

// CostEst returns the estimated computational cost associated with this layer,
// separated by neuron-level and synapse-level, in arbitrary units where
// cost per synapse is 1.  Neuron-level computation is more expensive but
// there are typically many fewer neurons, so in larger networks, synaptic
// costs tend to dominate.  Neuron cost is estimated from TimerReport output
// for large networks.
func (ly *Layer) CostEst() (neur, syn, tot int) {
	perNeur := 300 // cost per neuron, relative to synapse which is 1
	neur = len(ly.Neurons) * perNeur
	syn = 0
	for _, pji := range ly.SndPrjns {
		pj := pji.(LeabraPrjn).AsLeabra()
		ns := len(pj.Syns)
		syn += ns
	}
	tot = neur + syn
	return
}

//////////////////////////////////////////////////////////////////////////////////////
//  Stats

// note: use float64 for stats as that is best for logging

// MSE returns the sum-squared-error and mean-squared-error
// over the layer, in terms of ActP - ActM (valid even on non-target layers FWIW).
// Uses the given tolerance per-unit to count an error at all
// (e.g., .5 = activity just has to be on the right side of .5).
func (ly *Layer) MSE(tol float32) (sse, mse float64) {
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0, 0
	}
	sse = 0.0
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		var d float32
		if ly.Typ == emer.Compare {
			d = nrn.Targ - nrn.ActM
		} else {
			d = nrn.ActP - nrn.ActM
		}
		if math32.Abs(d) < tol {
			continue
		}
		sse += float64(d * d)
	}
	return sse, sse / float64(nn)
}

// SSE returns the sum-squared-error over the layer, in terms of ActP - ActM
// (valid even on non-target layers FWIW).
// Uses the given tolerance per-unit to count an error at all
// (e.g., .5 = activity just has to be on the right side of .5).
// Use this in Python which only allows single return values.
func (ly *Layer) SSE(tol float32) float64 {
	sse, _ := ly.MSE(tol)
	return sse
}

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion

// UnLesionNeurons unlesions (clears the Off flag) for all neurons in the layer
func (ly *Layer) UnLesionNeurons() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.ClearFlag(NeurOff)
	}
}

// LesionNeurons lesions (sets the Off flag) for given proportion (0-1) of neurons in layer
// returns number of neurons lesioned.  Emits error if prop > 1 as indication that percent
// might have been passed
func (ly *Layer) LesionNeurons(prop float32) int {
	ly.UnLesionNeurons()
	if prop > 1 {
		log.Printf("LesionNeurons got a proportion > 1 -- must be 0-1 as *proportion* (not percent) of neurons to lesion: %v\n", prop)
		return 0
	}
	nn := len(ly.Neurons)
	if nn == 0 {
		return 0
	}
	p := rand.Perm(nn)
	nl := int(prop * float32(nn))
	for i := 0; i < nl; i++ {
		nrn := &ly.Neurons[p[i]]
		nrn.SetFlag(NeurOff)
	}
	return nl
}

//////////////////////////////////////////////////////////////////////////////////////
//  Layer props for gui

var LayerProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"Defaults", ki.Props{
			"icon": "reset",
			"desc": "return all parameters to their intial default values",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's weight values according to prjn parameters, for all *sending* projections out of this layer",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"LesionNeurons", ki.Props{
			"icon": "close",
			"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
			"Args": ki.PropSlice{
				{"Proportion", ki.Props{
					"desc": "proportion (0 -- 1) of neurons to lesion",
				}},
			},
		}},
		{"UnLesionNeurons", ki.Props{
			"icon": "reset",
			"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
		}},
	},
}
