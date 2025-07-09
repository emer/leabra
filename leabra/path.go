// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/tensor"
)

// note: path.go contains algorithm methods; pathbase.go has infrastructure.

////////  Init methods

// SetScalesRPool initializes synaptic Scale values using given tensor
// of values which has unique values for each recv neuron within a given pool.
func (pt *Path) SetScalesRPool(scales tensor.Tensor) {
	rNuY := scales.DimSize(0)
	rNuX := scales.DimSize(1)
	rNu := rNuY * rNuX
	rfsz := scales.Len() / rNu

	rsh := pt.Recv.Shape
	rNpY := rsh.DimSize(0)
	rNpX := rsh.DimSize(1)
	r2d := false
	if rsh.NumDims() != 4 {
		r2d = true
		rNpY = 1
		rNpX = 1
	}

	for rpy := 0; rpy < rNpY; rpy++ {
		for rpx := 0; rpx < rNpX; rpx++ {
			for ruy := 0; ruy < rNuY; ruy++ {
				for rux := 0; rux < rNuX; rux++ {
					ri := 0
					if r2d {
						ri = rsh.IndexTo1D(ruy, rux)
					} else {
						ri = rsh.IndexTo1D(rpy, rpx, ruy, rux)
					}
					scst := (ruy*rNuX + rux) * rfsz
					nc := int(pt.RConN[ri])
					st := int(pt.RConIndexSt[ri])
					for ci := 0; ci < nc; ci++ {
						// si := int(pj.RConIndex[st+ci]) // could verify coords etc
						rsi := pt.RSynIndex[st+ci]
						sy := &pt.Syns[rsi]
						sc := scales.Float1D(scst + ci)
						sy.Scale = float32(sc)
					}
				}
			}
		}
	}
}

// SetWtsFunc initializes synaptic Wt value using given function
// based on receiving and sending unit indexes.
func (pt *Path) SetWtsFunc(wtFun func(si, ri int, send, recv *tensor.Shape) float32) {
	pp := &pt.Params
	rsh := &pt.Recv.Shape
	rn := rsh.Len()
	ssh := &pt.Send.Shape

	for ri := 0; ri < rn; ri++ {
		nc := int(pt.RConN[ri])
		st := int(pt.RConIndexSt[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pt.RConIndex[st+ci])
			wt := wtFun(si, ri, ssh, rsh)
			rsi := pt.RSynIndex[st+ci]
			sy := &pt.Syns[rsi]
			sy.Wt = wt * sy.Scale
			pp.Learn.LWtFromWt(sy)
		}
	}
}

// SetScalesFunc initializes synaptic Scale values using given function
// based on receiving and sending unit indexes.
func (pt *Path) SetScalesFunc(scaleFun func(si, ri int, send, recv *tensor.Shape) float32) {
	rsh := &pt.Recv.Shape
	rn := rsh.Len()
	ssh := &pt.Send.Shape

	for ri := 0; ri < rn; ri++ {
		nc := int(pt.RConN[ri])
		st := int(pt.RConIndexSt[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pt.RConIndex[st+ci])
			sc := scaleFun(si, ri, ssh, rsh)
			rsi := pt.RSynIndex[st+ci]
			sy := &pt.Syns[rsi]
			sy.Scale = sc
		}
	}
}

// InitWeightsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (pt *Path) InitWeightsSyn(syn *Synapse) {
	pp := &pt.Params
	if syn.Scale == 0 {
		syn.Scale = 1
	}
	syn.Wt = float32(pp.WtInit.Gen())
	// enforce normalized weight range -- required for most uses and if not
	// then a new type of path should be used:
	if syn.Wt < 0 {
		syn.Wt = 0
	}
	if syn.Wt > 1 {
		syn.Wt = 1
	}
	syn.LWt = pp.Learn.WtSig.LinFromSigWt(syn.Wt)
	syn.Wt *= syn.Scale // note: scale comes after so LWt is always "pure" non-scaled value
	syn.DWt = 0
	syn.Norm = 0
	syn.Moment = 0
}

// InitWeights initializes weight values according to Learn.WtInit params
func (pt *Path) InitWeights() {
	for si := range pt.Syns {
		sy := &pt.Syns[si]
		pt.InitWeightsSyn(sy)
	}
	for wi := range pt.WbRecv {
		wb := &pt.WbRecv[wi]
		wb.Init()
	}
	pt.InitGInc()
	pt.ClearTrace()
}

// InitWtSym initializes weight symmetry -- is given the reciprocal pathway where
// the Send and Recv layers are reversed.
func (pt *Path) InitWtSym(rpt *Path) {
	slay := pt.Send
	ns := int32(len(slay.Neurons))
	for si := int32(0); si < ns; si++ {
		nc := pt.SConN[si]
		st := pt.SConIndexSt[si]
		for ci := int32(0); ci < nc; ci++ {
			sy := &pt.Syns[st+ci]
			ri := pt.SConIndex[st+ci]
			// now we need to find the reciprocal synapse on rpt!
			// look in ri for sending connections
			rsi := ri
			if len(rpt.SConN) == 0 {
				continue
			}
			rsnc := rpt.SConN[rsi]
			if rsnc == 0 {
				continue
			}
			rsst := rpt.SConIndexSt[rsi]
			rist := rpt.SConIndex[rsst]        // starting index in recv path
			ried := rpt.SConIndex[rsst+rsnc-1] // ending index
			if si < rist || si > ried {        // fast reject -- paths are always in order!
				continue
			}
			// start at index proportional to si relative to rist
			up := int32(0)
			if ried > rist {
				up = int32(float32(rsnc) * float32(si-rist) / float32(ried-rist))
			}
			dn := up - 1

			for {
				doing := false
				if up < rsnc {
					doing = true
					rrii := rsst + up
					rri := rpt.SConIndex[rrii]
					if rri == si {
						rsy := &rpt.Syns[rrii]
						rsy.Wt = sy.Wt
						rsy.LWt = sy.LWt
						rsy.Scale = sy.Scale
						// note: if we support SymFromTop then can have option to go other way
						break
					}
					up++
				}
				if dn >= 0 {
					doing = true
					rrii := rsst + dn
					rri := rpt.SConIndex[rrii]
					if rri == si {
						rsy := &rpt.Syns[rrii]
						rsy.Wt = sy.Wt
						rsy.LWt = sy.LWt
						rsy.Scale = sy.Scale
						// note: if we support SymFromTop then can have option to go other way
						break
					}
					dn--
				}
				if !doing {
					break
				}
			}
		}
	}
}

// InitGInc initializes the per-pathway GInc threadsafe increment -- not
// typically needed (called during InitWeights only) but can be called when needed
func (pt *Path) InitGInc() {
	for ri := range pt.GInc {
		pt.GInc[ri] = 0
		pt.CtxtGeInc[ri] = 0
		pt.GeRaw[ri] = 0
	}
}

////////  Act methods

// SendGDelta sends the delta-activation from sending neuron index si,
// to integrate synaptic conductances on receivers
func (pt *Path) SendGDelta(si int, delta float32) {
	pp := &pt.Params
	if pp.Type == CTCtxtPath {
		return
	}
	scdel := delta * pt.GScale
	nc := pt.SConN[si]
	st := pt.SConIndexSt[si]
	syns := pt.Syns[st : st+nc]
	scons := pt.SConIndex[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pt.GInc[ri] += scdel * syns[ci].Wt
	}
}

// RecvGInc increments the receiver's GeRaw or GiRaw from that of all the pathways.
func (pt *Path) RecvGInc() {
	pp := &pt.Params
	rlay := pt.Recv
	switch pp.Type {
	case CTCtxtPath:
		// nop
	case InhibPath:
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GiRaw += pt.GInc[ri]
			pt.GInc[ri] = 0
		}
	case GPiThalPath:
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			ginc := pt.GInc[ri]
			pt.GeRaw[ri] += ginc
			rn.GeRaw += ginc
			pt.GInc[ri] = 0
		}
	default:
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GeRaw += pt.GInc[ri]
			pt.GInc[ri] = 0
		}
	}
}

////////  Learn methods

// DWt computes the weight change (learning) -- on sending pathways
func (pt *Path) DWt() {
	pp := &pt.Params
	if !pp.Learn.Learn {
		return
	}
	switch {
	case pp.Type == CHLPath && pp.CHL.On:
		pt.DWtCHL()
	case pp.Type == CTCtxtPath:
		pt.DWtCTCtxt()
	case pp.Type == EcCa1Path:
		pt.DWtEcCa1()
	case pp.Type == MatrixPath:
		pt.DWtMatrix()
	case pp.Type == RWPath:
		pt.DWtRW()
	case pp.Type == TDPredPath:
		pt.DWtTDPred()
	case pp.Type == DaHebbPath:
		pt.DWtDaHebb()
	default:
		pt.DWtStd()
	}
}

// DWt computes the weight change (learning) -- on sending pathways
func (pt *Path) DWtStd() {
	pp := &pt.Params
	slay := pt.Send
	rlay := pt.Recv
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		if sn.AvgS < pp.Learn.XCal.LrnThr && sn.AvgM < pp.Learn.XCal.LrnThr {
			continue
		}
		nc := int(pt.SConN[si])
		st := int(pt.SConIndexSt[si])
		syns := pt.Syns[st : st+nc]
		scons := pt.SConIndex[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			err, bcm := pp.Learn.CHLdWt(sn.AvgSLrn, sn.AvgM, rn.AvgSLrn, rn.AvgM, rn.AvgL)

			bcm *= pp.Learn.XCal.LongLrate(rn.AvgLLrn)
			err *= pp.Learn.XCal.MLrn
			dwt := bcm + err
			norm := float32(1)
			if pp.Learn.Norm.On {
				norm = pp.Learn.Norm.NormFromAbsDWt(&sy.Norm, math32.Abs(dwt))
			}
			if pp.Learn.Momentum.On {
				dwt = norm * pp.Learn.Momentum.MomentFromDWt(&sy.Moment, dwt)
			} else {
				dwt *= norm
			}
			sy.DWt += pp.Learn.Lrate * dwt
		}
		// aggregate max DWtNorm over sending synapses
		if pp.Learn.Norm.On {
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

// WtFromDWt updates the synaptic weight values from delta-weight changes -- on sending pathways
func (pt *Path) WtFromDWt() {
	pp := &pt.Params
	if !pp.Learn.Learn {
		return
	}
	switch pp.Type {
	case RWPath, TDPredPath:
		pt.WtFromDWtLinear()
		return
	}
	if pp.Learn.WtBal.On {
		for si := range pt.Syns {
			sy := &pt.Syns[si]
			ri := pt.SConIndex[si]
			wb := &pt.WbRecv[ri]
			pp.Learn.WtFromDWt(wb.Inc, wb.Dec, &sy.DWt, &sy.Wt, &sy.LWt, sy.Scale)
		}
	} else {
		for si := range pt.Syns {
			sy := &pt.Syns[si]
			pp.Learn.WtFromDWt(1, 1, &sy.DWt, &sy.Wt, &sy.LWt, sy.Scale)
		}
	}
}

// WtFromDWtLinear updates the synaptic weight values from delta-weight
// changes, with no constraints or limits
func (pt *Path) WtFromDWtLinear() {
	for si := range pt.Syns {
		sy := &pt.Syns[si]
		if sy.DWt != 0 {
			sy.Wt += sy.DWt // straight update, no limits or anything
			sy.LWt = sy.Wt
			sy.DWt = 0
		}
	}
}

// WtBalFromWt computes the Weight Balance factors based on average recv weights
func (pt *Path) WtBalFromWt() {
	pp := &pt.Params
	if !pp.Learn.Learn || !pp.Learn.WtBal.On {
		return
	}

	rlay := pt.Recv
	if !pp.Learn.WtBal.Targs && rlay.IsTarget() {
		return
	}
	for ri := range rlay.Neurons {
		nc := int(pt.RConN[ri])
		if nc < 1 {
			continue
		}
		wb := &pt.WbRecv[ri]
		st := int(pt.RConIndexSt[ri])
		rsidxs := pt.RSynIndex[st : st+nc]
		sumWt := float32(0)
		sumN := 0
		for ci := range rsidxs {
			rsi := rsidxs[ci]
			sy := &pt.Syns[rsi]
			if sy.Wt >= pp.Learn.WtBal.AvgThr {
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
		wb.Fact, wb.Inc, wb.Dec = pp.Learn.WtBal.WtBal(sumWt)
	}
}

// LrateMult sets the new Lrate parameter for Paths to LrateInit * mult.
// Useful for implementing learning rate schedules.
func (pt *Path) LrateMult(mult float32) {
	pp := &pt.Params
	pp.Learn.Lrate = pp.Learn.LrateInit * mult
}

////////  WtBalRecvPath

// WtBalRecvPath are state variables used in computing the WtBal weight balance function
// There is one of these for each Recv Neuron participating in the pathway.
type WtBalRecvPath struct {

	// average of effective weight values that exceed WtBal.AvgThr across given Recv Neuron's connections for given Path
	Avg float32

	// overall weight balance factor that drives changes in WbInc vs. WbDec via a sigmoidal function -- this is the net strength of weight balance changes
	Fact float32

	// weight balance increment factor -- extra multiplier to add to weight increases to maintain overall weight balance
	Inc float32

	// weight balance decrement factor -- extra multiplier to add to weight decreases to maintain overall weight balance
	Dec float32
}

func (wb *WtBalRecvPath) Init() {
	wb.Avg = 0
	wb.Fact = 0
	wb.Inc = 1
	wb.Dec = 1
}
