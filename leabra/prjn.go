// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strconv"
	"strings"

	"github.com/chewxy/math32"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/weights"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/indent"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// leabra.Prjn is a basic Leabra projection with synaptic learning parameters
type Prjn struct {
	PrjnStru
	WtInit  WtInitParams   `view:"inline" desc:"initial random weight distribution"`
	WtScale WtScaleParams  `view:"inline" desc:"weight scaling parameters: modulates overall strength of projection, using both absolute and relative factors"`
	Learn   LearnSynParams `view:"add-fields" desc:"synaptic-level learning parameters"`
	Syns    []Synapse      `desc:"synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SConIdx array"`

	// misc state variables below:
	GScale float32         `desc:"scaling factor for integrating synaptic input conductances (G's) -- computed in AlphaCycInit, incorporates running-average activity levels"`
	GInc   []float32       `desc:"local per-recv unit increment accumulator for synaptic conductance from sending units -- goes to either GeInc or GiInc on neuron depending on projection type -- this will be thread-safe"`
	WbRecv []WtBalRecvPrjn `desc:"weight balance state variables for this projection, one per recv neuron"`
}

var KiT_Prjn = kit.Types.AddType(&Prjn{}, PrjnProps)

var PrjnProps = ki.Props{}

// AsLeabra returns this prjn as a leabra.Prjn -- all derived prjns must redefine
// this to return the base Prjn type, so that the LeabraPrjn interface does not
// need to include accessors to all the basic stuff.
func (pj *Prjn) AsLeabra() *Prjn {
	return pj
}

func (pj *Prjn) Defaults() {
	pj.WtInit.Defaults()
	pj.WtScale.Defaults()
	pj.Learn.Defaults()
	pj.GScale = 1
}

// UpdateParams updates all params given any changes that might have been made to individual values
func (pj *Prjn) UpdateParams() {
	pj.WtScale.Update()
	pj.Learn.Update()
	pj.Learn.LrateInit = pj.Learn.Lrate
}

// AllParams returns a listing of all parameters in the Layer
func (pj *Prjn) AllParams() string {
	str := "///////////////////////////////////////////////////\nPrjn: " + pj.Name() + "\n"
	b, _ := json.MarshalIndent(&pj.WtInit, "", " ")
	str += "WtInit: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.WtScale, "", " ")
	str += "WtScale: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.Learn, "", " ")
	str += "Learn: {\n " + strings.Replace(JsonToParams(b), " XCal: {", "\n  XCal: {", -1)
	return str
}

func (pj *Prjn) SynVarNames() []string {
	return SynapseVars
}

// SynVarProps returns properties for variables
func (pj *Prjn) SynVarProps() map[string]string {
	return SynapseVarProps
}

// SynVals sets values of given variable name for each synapse, using the natural ordering
// of the synapses (sender based for Leabra),
// into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (pj *Prjn) SynVals(vals *[]float32, varNm string) error {
	vidx, err := SynapseVarByName(varNm)
	if err != nil {
		return err
	}
	ns := len(pj.Syns)
	if *vals == nil || cap(*vals) < ns {
		*vals = make([]float32, ns)
	} else if len(*vals) < ns {
		*vals = (*vals)[0:ns]
	}
	for i := range pj.Syns {
		sy := &pj.Syns[i]
		(*vals)[i] = sy.VarByIndex(vidx)
	}
	return nil
}

// Syn returns the synapse between given send, recv unit indexes (1D, flat indexes).
// Returns nil for access errors (see SynTry for version that returns errors).
func (pj *Prjn) Syn(sidx, ridx int) *Synapse {
	nc := int(pj.RConN[ridx])
	st := int(pj.RConIdxSt[ridx])
	for ci := 0; ci < nc; ci++ {
		si := int(pj.RConIdx[st+ci])
		if si != sidx {
			continue
		}
		rsi := pj.RSynIdx[st+ci]
		sy := &pj.Syns[rsi]
		return sy
	}
	return nil
}

// SynTry returns the synapse between given send, recv unit indexes (1D, flat indexes).
// Returns error for access errors.
func (pj *Prjn) SynTry(sidx, ridx int) (*Synapse, error) {
	slay := pj.Send.(LeabraLayer).AsLeabra()
	rlay := pj.Recv.(LeabraLayer).AsLeabra()
	nr := len(rlay.Neurons)
	ns := len(slay.Neurons)
	if ridx >= nr {
		return nil, fmt.Errorf("Prjn.SynVal: recv unit index %v is > size of recv layer: %v", ridx, nr)
	}
	if sidx >= ns {
		return nil, fmt.Errorf("Prjn.SynVal: send unit index %v is > size of send layer: %v", sidx, ns)
	}
	nc := int(pj.RConN[ridx])
	st := int(pj.RConIdxSt[ridx])
	for ci := 0; ci < nc; ci++ {
		si := int(pj.RConIdx[st+ci])
		if si != sidx {
			continue
		}
		rsi := pj.RSynIdx[st+ci]
		sy := &pj.Syns[rsi]
		return sy, nil
	}
	return nil, fmt.Errorf("Prjn.SynTry: recv unit index %v does not recv from send unit index %v", ridx, sidx)
}

// SynVal returns value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes).
// Returns math32.NaN() for access errors (see SynValTry for error message)
func (pj *Prjn) SynVal(varNm string, sidx, ridx int) float32 {
	vidx, err := SynapseVarByName(varNm)
	if err != nil {
		return math32.NaN()
	}
	sy := pj.Syn(sidx, ridx)
	if sy == nil {
		return math32.NaN()
	}
	return sy.VarByIndex(vidx)
}

// SynValTry returns value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes).
// Returns error for access errors.
func (pj *Prjn) SynValTry(varNm string, sidx, ridx int) (float32, error) {
	vidx, err := SynapseVarByName(varNm)
	if err != nil {
		return 0, err
	}
	sy, err := pj.SynTry(sidx, ridx)
	if err != nil {
		return 0, err
	}
	return sy.VarByIndex(vidx), nil
}

// SetSynVal sets value of given variable name on the synapse
// between given send, recv unit indexes (1D, flat indexes)
// returns error for access errors.
func (pj *Prjn) SetSynVal(varNm string, sidx, ridx int, val float32) error {
	vidx, err := SynapseVarByName(varNm)
	if err != nil {
		return err
	}
	sy, err := pj.SynTry(sidx, ridx)
	if err != nil {
		return err
	}
	sy.SetVarByIndex(vidx, val)
	if varNm == "Wt" {
		pj.Learn.LWtFmWt(sy)
	}
	return nil
}

///////////////////////////////////////////////////////////////////////
//  Weights File

// WriteWtsJSON writes the weights from this projection from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (pj *Prjn) WriteWtsJSON(w io.Writer, depth int) {
	slay := pj.Send.(LeabraLayer).AsLeabra()
	rlay := pj.Recv.(LeabraLayer).AsLeabra()
	nr := len(rlay.Neurons)
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("{\n"))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"From\": %q,\n", slay.Name())))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"MetaData\": {\n")))
	depth++
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"GScale\": \"%g\"\n", pj.GScale)))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("},\n"))
	w.Write(indent.TabBytes(depth))
	w.Write([]byte(fmt.Sprintf("\"Rs\": [\n")))
	depth++
	for ri := 0; ri < nr; ri++ {
		nc := int(pj.RConN[ri])
		st := int(pj.RConIdxSt[ri])
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("{\n"))
		depth++
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"Ri\": %v,\n", ri)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte(fmt.Sprintf("\"N\": %v,\n", nc)))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Si\": [ "))
		for ci := 0; ci < nc; ci++ {
			si := pj.RConIdx[st+ci]
			w.Write([]byte(fmt.Sprintf("%v", si)))
			if ci == nc-1 {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("],\n"))
		w.Write(indent.TabBytes(depth))
		w.Write([]byte("\"Wt\": [ "))
		for ci := 0; ci < nc; ci++ {
			rsi := pj.RSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			w.Write([]byte(strconv.FormatFloat(float64(sy.Wt), 'g', weights.Prec, 32)))
			if ci == nc-1 {
				w.Write([]byte(" "))
			} else {
				w.Write([]byte(", "))
			}
		}
		w.Write([]byte("]\n"))
		depth--
		w.Write(indent.TabBytes(depth))
		if ri == nr-1 {
			w.Write([]byte("}\n"))
		} else {
			w.Write([]byte("},\n"))
		}
	}
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("]\n"))
	depth--
	w.Write(indent.TabBytes(depth))
	w.Write([]byte("}")) // note: leave unterminated as outer loop needs to add , or just \n depending
}

// ReadWtsJSON reads the weights from this projection from the receiver-side perspective
// in a JSON text format.  This is for a set of weights that were saved *for one prjn only*
// and is not used for the network-level ReadWtsJSON, which reads into a separate
// structure -- see SetWts method.
func (pj *Prjn) ReadWtsJSON(r io.Reader) error {
	pw, err := weights.PrjnReadJSON(r)
	if err != nil {
		return err // note: already logged
	}
	return pj.SetWts(pw)
}

// SetWts sets the weights for this projection from weights.Prjn decoded values
func (pj *Prjn) SetWts(pw *weights.Prjn) error {
	if pw.MetaData != nil {
		if gs, ok := pw.MetaData["GScale"]; ok {
			pv, _ := strconv.ParseFloat(gs, 32)
			pj.GScale = float32(pv)
		}
	}
	var err error
	for i := range pw.Rs {
		pr := &pw.Rs[i]
		for si := range pr.Si {
			er := pj.SetSynVal("Wt", pr.Si[si], pr.Ri, pr.Wt[si]) // updates lin wt
			if er != nil {
				err = er
			}
		}
	}
	return err
}

// Build constructs the full connectivity among the layers as specified in this projection.
// Calls PrjnStru.BuildStru and then allocates the synaptic values in Syns accordingly.
func (pj *Prjn) Build() error {
	if err := pj.BuildStru(); err != nil {
		return err
	}
	pj.Syns = make([]Synapse, len(pj.SConIdx))
	rsh := pj.Recv.Shape()
	//	ssh := pj.Send.Shape()
	rlen := rsh.Len()
	pj.GInc = make([]float32, rlen)
	pj.WbRecv = make([]WtBalRecvPrjn, rlen)
	return nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// SetScalesRPool initializes synaptic Scale values using given tensor
// of values which has unique values for each recv neuron within a given pool
// i.e., recv layer must have 4D pool structure.
func (pj *Prjn) SetScalesRPool(scales etensor.Tensor) {
	rNuY := scales.Dim(0)
	rNuX := scales.Dim(1)
	rNu := rNuY * rNuX
	rfsz := scales.Len() / rNu

	rsh := pj.Recv.Shape()
	if rsh.NumDims() != 4 {
		log.Printf("leabra.Prjn.SetScalesRPool: recv layer must have 4D shape")
		return
	}
	rNpY := rsh.Dim(0)
	rNpX := rsh.Dim(1)

	for rpy := 0; rpy < rNpY; rpy++ {
		for rpx := 0; rpx < rNpX; rpx++ {
			for ruy := 0; ruy < rNuY; ruy++ {
				for rux := 0; rux < rNuX; rux++ {
					ri := rsh.Offset([]int{rpy, rpx, ruy, rux})
					scst := (ruy*rNuX + rux) * rfsz
					nc := int(pj.RConN[ri])
					st := int(pj.RConIdxSt[ri])
					for ci := 0; ci < nc; ci++ {
						// si := int(pj.RConIdx[st+ci]) // could verify coords etc
						rsi := pj.RSynIdx[st+ci]
						sy := &pj.Syns[rsi]
						sc := scales.FloatVal1D(scst + ci)
						sy.Scale = float32(sc)
					}
				}
			}
		}
	}
}

// SetWtsFunc initializes synaptic Wt value using given function
// based on receiving and sending unit indexes.
func (pj *Prjn) SetWtsFunc(wtFun func(si, ri int, send, recv *etensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		nc := int(pj.RConN[ri])
		st := int(pj.RConIdxSt[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pj.RConIdx[st+ci])
			wt := wtFun(si, ri, ssh, rsh)
			rsi := pj.RSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			sy.Wt = wt
			pj.Learn.LWtFmWt(sy)
		}
	}
}

// SetScalesFunc initializes synaptic Scale values using given function
// based on receiving and sending unit indexes.
func (pj *Prjn) SetScalesFunc(scaleFun func(si, ri int, send, recv *etensor.Shape) float32) {
	rsh := pj.Recv.Shape()
	rn := rsh.Len()
	ssh := pj.Send.Shape()

	for ri := 0; ri < rn; ri++ {
		nc := int(pj.RConN[ri])
		st := int(pj.RConIdxSt[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pj.RConIdx[st+ci])
			sc := scaleFun(si, ri, ssh, rsh)
			rsi := pj.RSynIdx[st+ci]
			sy := &pj.Syns[rsi]
			sy.Scale = sc
		}
	}
}

// InitWtsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (pj *Prjn) InitWtsSyn(syn *Synapse) {
	if syn.Scale == 0 {
		syn.Scale = 1
	}
	syn.Wt = float32(pj.WtInit.Gen(-1))
	// enforce normalized weight range -- required for most uses and if not
	// then a new type of prjn should be used:
	if syn.Wt < 0 {
		syn.Wt = 0
	}
	if syn.Wt > 1 {
		syn.Wt = 1
	}
	syn.LWt = pj.Learn.WtSig.LinFmSigWt(syn.Wt)
	syn.Wt *= syn.Scale // note: scale comes after so LWt is always "pure" non-scaled value
	syn.DWt = 0
	syn.Norm = 0
	syn.Moment = 0
}

// InitWts initializes weight values according to Learn.WtInit params
func (pj *Prjn) InitWts() {
	for si := range pj.Syns {
		sy := &pj.Syns[si]
		pj.InitWtsSyn(sy)
	}
	for wi := range pj.WbRecv {
		wb := &pj.WbRecv[wi]
		wb.Init()
	}
	pj.LeabraPrj.InitGInc()
}

// InitWtSym initializes weight symmetry -- is given the reciprocal projection where
// the Send and Recv layers are reversed.
func (pj *Prjn) InitWtSym(rpjp LeabraPrjn) {
	rpj := rpjp.AsLeabra()
	slay := pj.Send.(LeabraLayer).AsLeabra()
	ns := len(slay.Neurons)
	for si := 0; si < ns; si++ {
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		for ci := 0; ci < nc; ci++ {
			sy := &pj.Syns[st+ci]
			ri := pj.SConIdx[st+ci]
			// now we need to find the reciprocal synapse on rpj!
			// look in ri for sending connections
			rsi := ri
			rsnc := int(rpj.SConN[rsi])
			rsst := int(rpj.SConIdxSt[rsi])
			for rci := 0; rci < rsnc; rci++ {
				rri := int(rpj.SConIdx[rsst+rci])
				if rri == si {
					rsy := &rpj.Syns[rsst+rci]
					rsy.Wt = sy.Wt
					rsy.LWt = sy.LWt
					rsy.Scale = sy.Scale
					// note: if we support SymFmTop then can have option to go other way
				}
			}
		}
	}
}

// IniteGInc initializes the per-projection GInc threadsafe increment -- not
// typically needed (called during InitWts only) but can be called when needed
func (pj *Prjn) InitGInc() {
	for ri := range pj.GInc {
		pj.GInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendGDelta sends the delta-activation from sending neuron index si,
// to integrate synaptic conductances on receivers
func (pj *Prjn) SendGDelta(si int, delta float32) {
	scdel := delta * pj.GScale
	nc := pj.SConN[si]
	st := pj.SConIdxSt[si]
	syns := pj.Syns[st : st+nc]
	scons := pj.SConIdx[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pj.GInc[ri] += scdel * syns[ci].Wt
	}
}

// RecvGInc increments the receiver's GeInc or GiInc from that of all the projections.
func (pj *Prjn) RecvGInc() {
	rlay := pj.Recv.(LeabraLayer).AsLeabra()
	if pj.Typ == emer.Inhib {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GiInc += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	} else {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GeInc += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) -- on sending projections
func (pj *Prjn) DWt() {
	if !pj.Learn.Learn {
		return
	}
	slay := pj.Send.(LeabraLayer).AsLeabra()
	rlay := pj.Recv.(LeabraLayer).AsLeabra()
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		if sn.AvgS < pj.Learn.XCal.LrnThr && sn.AvgM < pj.Learn.XCal.LrnThr {
			continue
		}
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			err, bcm := pj.Learn.CHLdWt(sn.AvgSLrn, sn.AvgM, rn.AvgSLrn, rn.AvgM, rn.AvgL)

			bcm *= pj.Learn.XCal.LongLrate(rn.AvgLLrn)
			err *= pj.Learn.XCal.MLrn
			dwt := bcm + err
			norm := float32(1)
			if pj.Learn.Norm.On {
				norm = pj.Learn.Norm.NormFmAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pj.Learn.Momentum.On {
				dwt = norm * pj.Learn.Momentum.MomentFmDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pj.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pj.Learn.Norm.On {
			maxNorm := float32(0)
			for ci := range syns {
				sy := &syns[ci]
				if sy.Norm > maxNorm {
					maxNorm = sy.Norm
				}
			}
			for ci := range syns {
				sy := &syns[ci]
				sy.Norm = maxNorm
			}
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *Prjn) WtFmDWt() {
	if !pj.Learn.Learn {
		return
	}
	if pj.Learn.WtBal.On {
		for si := range pj.Syns {
			sy := &pj.Syns[si]
			ri := pj.SConIdx[si]
			wb := &pj.WbRecv[ri]
			pj.Learn.WtFmDWt(wb.Inc, wb.Dec, &sy.DWt, &sy.Wt, &sy.LWt, sy.Scale)
		}
	} else {
		for si := range pj.Syns {
			sy := &pj.Syns[si]
			pj.Learn.WtFmDWt(1, 1, &sy.DWt, &sy.Wt, &sy.LWt, sy.Scale)
		}
	}
}

// WtBalFmWt computes the Weight Balance factors based on average recv weights
func (pj *Prjn) WtBalFmWt() {
	if !pj.Learn.Learn || !pj.Learn.WtBal.On {
		return
	}

	rlay := pj.Recv.(LeabraLayer).AsLeabra()
	if rlay.Typ == emer.Target {
		return
	}
	for ri := range rlay.Neurons {
		nc := int(pj.RConN[ri])
		if nc <= 1 {
			continue
		}
		rn := &rlay.Neurons[ri]
		if rn.HasFlag(NeurHasTarg) { // todo: ensure that Pulvinar has this set, or do something else
			continue
		}
		wb := &pj.WbRecv[ri]
		st := int(pj.RConIdxSt[ri])
		rsidxs := pj.RSynIdx[st : st+nc]
		sumWt := float32(0)
		sumN := 0
		for ci := range rsidxs {
			rsi := rsidxs[ci]
			sy := &pj.Syns[rsi]
			if sy.Wt >= pj.Learn.WtBal.AvgThr {
				sumWt += sy.Wt
				sumN++
			}
		}
		if sumN > 0 {
			sumWt /= float32(sumN)
		} else {
			sumWt = 0
		}
		wb.Avg = sumWt
		wb.Fact, wb.Inc, wb.Dec = pj.Learn.WtBal.WtBal(sumWt)
	}
}

// LrateMult sets the new Lrate parameter for Prjns to LrateInit * mult.
// Useful for implementing learning rate schedules.
func (pj *Prjn) LrateMult(mult float32) {
	pj.Learn.Lrate = pj.Learn.LrateInit * mult
}

///////////////////////////////////////////////////////////////////////
//  WtBalRecvPrjn

// WtBalRecvPrjn are state variables used in computing the WtBal weight balance function
// There is one of these for each Recv Neuron participating in the projection.
type WtBalRecvPrjn struct {
	Avg  float32 `desc:"average of effective weight values that exceed WtBal.AvgThr across given Recv Neuron's connections for given Prjn"`
	Fact float32 `desc:"overall weight balance factor that drives changes in WbInc vs. WbDec via a sigmoidal function -- this is the net strength of weight balance changes"`
	Inc  float32 `desc:"weight balance increment factor -- extra multiplier to add to weight increases to maintain overall weight balance"`
	Dec  float32 `desc:"weight balance decrement factor -- extra multiplier to add to weight decreases to maintain overall weight balance"`
}

func (wb *WtBalRecvPrjn) Init() {
	wb.Avg = 0
	wb.Fact = 0
	wb.Inc = 1
	wb.Dec = 1
}
