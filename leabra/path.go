// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
)

// note: path.go contains algorithm methods; pathbase.go has infrastructure.

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// SetScalesRPool initializes synaptic Scale values using given tensor
// of values which has unique values for each recv neuron within a given pool.
func (pj *Path) SetScalesRPool(scales tensor.Tensor) {
	rNuY := scales.Dim(0)
	rNuX := scales.Dim(1)
	rNu := rNuY * rNuX
	rfsz := scales.Len() / rNu

	rsh := pj.Recv.Shape
	rNpY := rsh.Dim(0)
	rNpX := rsh.Dim(1)
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
						ri = rsh.Offset([]int{ruy, rux})
					} else {
						ri = rsh.Offset([]int{rpy, rpx, ruy, rux})
					}
					scst := (ruy*rNuX + rux) * rfsz
					nc := int(pj.RConN[ri])
					st := int(pj.RConIndexSt[ri])
					for ci := 0; ci < nc; ci++ {
						// si := int(pj.RConIndex[st+ci]) // could verify coords etc
						rsi := pj.RSynIndex[st+ci]
						sy := &pj.Syns[rsi]
						sc := scales.FloatValue1D(scst + ci)
						sy.Scale = float32(sc)
					}
				}
			}
		}
	}
}

// SetWtsFunc initializes synaptic Wt value using given function
// based on receiving and sending unit indexes.
func (pj *Path) SetWtsFunc(wtFun func(si, ri int, send, recv *tensor.Shape) float32) {
	rsh := pj.Recv.Shape
	rn := rsh.Len()
	ssh := pj.Send.Shape

	for ri := 0; ri < rn; ri++ {
		nc := int(pj.RConN[ri])
		st := int(pj.RConIndexSt[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pj.RConIndex[st+ci])
			wt := wtFun(si, ri, ssh, rsh)
			rsi := pj.RSynIndex[st+ci]
			sy := &pj.Syns[rsi]
			sy.Wt = wt * sy.Scale
			pj.Learn.LWtFmWt(sy)
		}
	}
}

// SetScalesFunc initializes synaptic Scale values using given function
// based on receiving and sending unit indexes.
func (pj *Path) SetScalesFunc(scaleFun func(si, ri int, send, recv *tensor.Shape) float32) {
	rsh := pj.Recv.Shape
	rn := rsh.Len()
	ssh := pj.Send.Shape

	for ri := 0; ri < rn; ri++ {
		nc := int(pj.RConN[ri])
		st := int(pj.RConIndexSt[ri])
		for ci := 0; ci < nc; ci++ {
			si := int(pj.RConIndex[st+ci])
			sc := scaleFun(si, ri, ssh, rsh)
			rsi := pj.RSynIndex[st+ci]
			sy := &pj.Syns[rsi]
			sy.Scale = sc
		}
	}
}

// InitWeightsSyn initializes weight values based on WtInit randomness parameters
// for an individual synapse.
// It also updates the linear weight value based on the sigmoidal weight value.
func (pj *Path) InitWeightsSyn(syn *Synapse) {
	if syn.Scale == 0 {
		syn.Scale = 1
	}
	syn.Wt = float32(pj.WtInit.Gen(-1))
	// enforce normalized weight range -- required for most uses and if not
	// then a new type of path should be used:
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

// InitWeights initializes weight values according to Learn.WtInit params
func (pj *Path) InitWeights() {
	for si := range pj.Syns {
		sy := &pj.Syns[si]
		pj.InitWeightsSyn(sy)
	}
	for wi := range pj.WbRecv {
		wb := &pj.WbRecv[wi]
		wb.Init()
	}
	pj.LeabraPrj.InitGInc()
}

// InitWtSym initializes weight symmetry -- is given the reciprocal pathway where
// the Send and Recv layers are reversed.
func (pj *Path) InitWtSym(rpjp LeabraPath) {
	rpj := rpjp.AsLeabra()
	slay := pj.Send.(LeabraLayer).AsLeabra()
	ns := int32(len(slay.Neurons))
	for si := int32(0); si < ns; si++ {
		nc := pj.SConN[si]
		st := pj.SConIndexSt[si]
		for ci := int32(0); ci < nc; ci++ {
			sy := &pj.Syns[st+ci]
			ri := pj.SConIndex[st+ci]
			// now we need to find the reciprocal synapse on rpj!
			// look in ri for sending connections
			rsi := ri
			if len(rpj.SConN) == 0 {
				continue
			}
			rsnc := rpj.SConN[rsi]
			if rsnc == 0 {
				continue
			}
			rsst := rpj.SConIndexSt[rsi]
			rist := rpj.SConIndex[rsst]        // starting index in recv path
			ried := rpj.SConIndex[rsst+rsnc-1] // ending index
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
					rri := rpj.SConIndex[rrii]
					if rri == si {
						rsy := &rpj.Syns[rrii]
						rsy.Wt = sy.Wt
						rsy.LWt = sy.LWt
						rsy.Scale = sy.Scale
						// note: if we support SymFmTop then can have option to go other way
						break
					}
					up++
				}
				if dn >= 0 {
					doing = true
					rrii := rsst + dn
					rri := rpj.SConIndex[rrii]
					if rri == si {
						rsy := &rpj.Syns[rrii]
						rsy.Wt = sy.Wt
						rsy.LWt = sy.LWt
						rsy.Scale = sy.Scale
						// note: if we support SymFmTop then can have option to go other way
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
func (pj *Path) InitGInc() {
	for ri := range pj.GInc {
		pj.GInc[ri] = 0
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Act methods

// SendGDelta sends the delta-activation from sending neuron index si,
// to integrate synaptic conductances on receivers
func (pj *Path) SendGDelta(si int, delta float32) {
	scdel := delta * pj.GScale
	nc := pj.SConN[si]
	st := pj.SConIndexSt[si]
	syns := pj.Syns[st : st+nc]
	scons := pj.SConIndex[st : st+nc]
	for ci := range syns {
		ri := scons[ci]
		pj.GInc[ri] += scdel * syns[ci].Wt
	}
}

// RecvGInc increments the receiver's GeRaw or GiRaw from that of all the pathways.
func (pj *Path) RecvGInc() {
	rlay := pj.Recv.(LeabraLayer).AsLeabra()
	if pj.Type == InhibPath {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GiRaw += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	} else {
		for ri := range rlay.Neurons {
			rn := &rlay.Neurons[ri]
			rn.GeRaw += pj.GInc[ri]
			pj.GInc[ri] = 0
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) -- on sending pathways
func (pj *Path) DWt() {
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
		st := int(pj.SConIndexSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIndex[st : st+nc]
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

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending pathways
func (pj *Path) WtFmDWt() {
	if !pj.Learn.Learn {
		return
	}
	if pj.Learn.WtBal.On {
		for si := range pj.Syns {
			sy := &pj.Syns[si]
			ri := pj.SConIndex[si]
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
func (pj *Path) WtBalFmWt() {
	if !pj.Learn.Learn || !pj.Learn.WtBal.On {
		return
	}

	rlay := pj.Recv.(LeabraLayer).AsLeabra()
	if !pj.Learn.WtBal.Targs && rlay.LeabraLay.IsTarget() {
		return
	}
	for ri := range rlay.Neurons {
		nc := int(pj.RConN[ri])
		if nc < 1 {
			continue
		}
		wb := &pj.WbRecv[ri]
		st := int(pj.RConIndexSt[ri])
		rsidxs := pj.RSynIndex[st : st+nc]
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

// LrateMult sets the new Lrate parameter for Paths to LrateInit * mult.
// Useful for implementing learning rate schedules.
func (pj *Path) LrateMult(mult float32) {
	pj.Learn.Lrate = pj.Learn.LrateInit * mult
}

///////////////////////////////////////////////////////////////////////
//  WtBalRecvPath

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
