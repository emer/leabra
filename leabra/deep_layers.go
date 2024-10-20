// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"log"
	"math"

	"cogentcore.org/core/math32"
)

// BurstParams determine how the 5IB Burst activation is computed from
// standard Act activation values in SuperLayer. It is thresholded.
type BurstParams struct {

	// Quarter(s) when bursting occurs -- typically Q4 but can also be Q2 and Q4 for beta-frequency updating.  Note: this is a bitflag and must be accessed using its Set / Has etc routines, 32 bit versions.
	BurstQtr Quarters

	// Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds.
	ThrRel float32 `max:"1" default:"0.1,0.2,0.5"`

	// Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = Act).  Overall effective threshold is MAX of relative and absolute thresholds.
	ThrAbs float32 `min:"0" max:"1" default:"0.1,0.2,0.5"`
}

func (db *BurstParams) Defaults() {
	db.BurstQtr.SetFlag(true, Q4)
	db.ThrRel = 0.1
	db.ThrAbs = 0.1
}

func (db *BurstParams) Update() {
}

////////  Burst -- computed in CyclePost

// BurstPrv records Burst activity just prior to burst
func (ly *Layer) BurstPrv(ctx *Context) {
	if !ly.Burst.BurstQtr.HasNext(ctx.Quarter) {
		return
	}
	// if will be updating next quarter, save just prior
	// this logic works for all cases, but e.g., BurstPrv doesn't update
	// until end of minus phase for Q4 BurstQtr
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.BurstPrv = nrn.Burst
	}
}

// BurstFromAct updates Burst layer 5IB bursting value from current Act
// (superficial activation), subject to thresholding.
func (ly *Layer) BurstFromAct(ctx *Context) {
	if !ly.Burst.BurstQtr.HasFlag(ctx.Quarter) {
		return
	}
	lpl := &ly.Pools[0]
	actMax := lpl.Inhib.Act.Max
	actAvg := lpl.Inhib.Act.Avg
	thr := actAvg + ly.Burst.ThrRel*(actMax-actAvg)
	thr = math32.Max(thr, ly.Burst.ThrAbs)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		burst := float32(0)
		if nrn.Act > thr {
			burst = nrn.Act
		}
		nrn.Burst = burst
	}
}

// BurstAsAct records Burst activity as the activation directly.
func (ly *Layer) BurstAsAct(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Burst = nrn.Act
	}
}

//////// DeepCtxt -- once after Burst quarter

// SendCtxtGe sends Burst activation over CTCtxtPath pathways to integrate
// CtxtGe excitatory conductance on CT layers.
// This must be called at the end of the Burst quarter for this layer.
// Satisfies the CtxtSender interface.
func (ly *Layer) SendCtxtGe(ctx *Context) {
	if !ly.Burst.BurstQtr.HasFlag(ctx.Quarter) {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if nrn.Burst > ly.Act.OptThresh.Send {
			for _, sp := range ly.SendPaths {
				if sp.Off {
					continue
				}
				if sp.Type != CTCtxtPath {
					continue
				}
				sp.SendCtxtGe(ni, nrn.Burst)
			}
		}
	}
}

// CTGFromInc integrates new synaptic conductances from increments
// sent during last SendGDelta.
func (ly *Layer) CTGFromInc(ctx *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		geRaw := nrn.GeRaw + ly.Neurons[ni].CtxtGe
		ly.Act.GeFromRaw(nrn, geRaw)
		ly.Act.GiFromRaw(nrn, nrn.GiRaw)
	}
}

// CtxtFromGe integrates new CtxtGe excitatory conductance from pathways,
// and computes overall Ctxt value, only on Deep layers.
// This must be called at the end of the DeepBurst quarter for this layer,
// after SendCtxtGe.
func (ly *Layer) CtxtFromGe(ctx *Context) {
	if ly.Type != CTLayer {
		return
	}
	if !ly.Burst.BurstQtr.HasFlag(ctx.Quarter) {
		return
	}
	for ni := range ly.Neurons {
		ly.Neurons[ni].CtxtGe = 0
	}
	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		if pt.Type != CTCtxtPath {
			continue
		}
		pt.RecvCtxtGeInc()
	}
}

//////// Pulvinar

// Driver describes the source of driver inputs from cortex into Pulvinar.
type Driver struct {

	// driver layer
	Driver string

	// offset into Pulvinar pool
	Off int `inactive:"-"`
}

// Drivers are a list of drivers
type Drivers []*Driver

// Add adds new driver(s)
func (dr *Drivers) Add(laynms ...string) {
	for _, laynm := range laynms {
		d := &Driver{}
		d.Driver = laynm
		*dr = append(*dr, d)
	}
}

// PulvinarParams provides parameters for how the plus-phase (outcome) state
// of thalamic relay cell (e.g., Pulvinar) neurons is computed from the
// corresponding driver neuron Burst activation.
type PulvinarParams struct {

	// Turn off the driver inputs, in which case this layer behaves like a standard layer
	DriversOff bool `default:"false"`

	// Quarter(s) when bursting occurs -- typically Q4 but can also be Q2 and Q4 for beta-frequency updating.  Note: this is a bitflag and must be accessed using its Set / Has etc routines
	BurstQtr Quarters

	// multiplier on driver input strength, multiplies activation of driver layer
	DriveScale float32 `default:"0.3" min:"0.0"`

	// Level of Max driver layer activation at which the predictive non-burst inputs are fully inhibited.  Computationally, it is essential that driver inputs inhibit effect of predictive non-driver (CTLayer) inputs, so that the plus phase is not always just the minus phase plus something extra (the error will never go to zero then).  When max driver act input exceeds this value, predictive non-driver inputs are fully suppressed.  If there is only weak burst input however, then the predictive inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning.
	MaxInhib float32 `default:"0.6" min:"0.01"`

	// Do not treat the pools in this layer as topographically organized relative to driver inputs -- all drivers compress down to give same input to all pools
	NoTopo bool

	// proportion of average across driver pools that is combined with Max to provide some graded tie-breaker signal -- especially important for large pool downsampling, e.g., when doing NoTopo
	AvgMix float32 `min:"0" max:"1"`

	// Apply threshold to driver burst input for computing plus-phase activations -- above BinThr, then Act = BinOn, below = BinOff.  This is beneficial for layers with weaker graded activations, such as V1 or other perceptual inputs.
	Binarize bool

	// Threshold for binarizing in terms of sending Burst activation
	BinThr float32 `viewif:"Binarize"`

	// Resulting driver Ge value for units above threshold -- lower value around 0.3 or so seems best (DriveScale is NOT applied -- generally same range as that).
	BinOn float32 `default:"0.3" viewif:"Binarize"`

	// Resulting driver Ge value for units below threshold -- typically 0.
	BinOff float32 `default:"0" viewif:"Binarize"`
}

func (tp *PulvinarParams) Update() {
}

func (tp *PulvinarParams) Defaults() {
	tp.BurstQtr.SetFlag(true, Q4)
	tp.DriveScale = 0.3
	tp.MaxInhib = 0.6
	tp.Binarize = false
	tp.BinThr = 0.4
	tp.BinOn = 0.3
	tp.BinOff = 0
}

// DriveGe returns effective excitatory conductance to use for given driver
// input Burst activation
func (tp *PulvinarParams) DriveGe(act float32) float32 {
	if tp.Binarize {
		if act >= tp.BinThr {
			return tp.BinOn
		} else {
			return tp.BinOff
		}
	} else {
		return tp.DriveScale * act
	}
}

// GeFromMaxAvg returns the drive Ge value as function of max and average
func (tp *PulvinarParams) GeFromMaxAvg(max, avg float32) float32 {
	deff := (1-tp.AvgMix)*max + tp.AvgMix*avg
	return tp.DriveGe(deff)
}

// UnitsSize returns the dimension of the units,
// either within a pool for 4D, or layer for 2D..
func UnitsSize(ly *Layer) (x, y int) {
	if ly.Is4D() {
		y = ly.Shape.DimSize(2)
		x = ly.Shape.DimSize(3)
	} else {
		y = ly.Shape.DimSize(0)
		x = ly.Shape.DimSize(1)
	}
	return
}

// DriverLayer returns the driver layer for given Driver
func (ly *Layer) DriverLayer(drv *Driver) (*Layer, error) {
	tly := ly.Network.LayerByName(drv.Driver)
	if tly == nil {
		err := fmt.Errorf("PulvinarLayer %s: Driver Layer not found", ly.Name)
		log.Println(err)
		return nil, err
	}
	return tly, nil
}

// SetDriverOffs sets the driver offsets.
func (ly *Layer) SetDriverOffs() error {
	if ly.Type != PulvinarLayer {
		return nil
	}
	mx, my := UnitsSize(ly)
	mn := my * mx
	off := 0
	var err error
	for _, drv := range ly.Drivers {
		dl, err := ly.DriverLayer(drv)
		if err != nil {
			continue
		}
		drv.Off = off
		x, y := UnitsSize(dl)
		off += y * x
	}
	if off > mn {
		err = fmt.Errorf("PulvinarLayer %s: size of drivers: %d is greater than units: %d", ly.Name, off, mn)
		log.Println(err)
	}
	return err
}

func DriveAct(dni int, dly *Layer, issuper bool) float32 {
	act := float32(0)
	if issuper {
		act = dly.Neurons[dni].Burst
	} else {
		act = dly.Neurons[dni].Act
	}
	lmax := dly.Pools[0].Inhib.Act.Max // normalize by drive layer max act
	if lmax > 0.1 {                    // this puts all layers on equal footing for driving..
		return act / lmax
	}
	return act
}

// SetDriverNeuron sets the driver activation for given Neuron,
// based on given Ge driving value (use DriveFromMaxAvg) from driver layer (Burst or Act)
func (ly *Layer) SetDriverNeuron(tni int, drvGe, drvInhib float32) {
	if tni >= len(ly.Neurons) {
		return
	}
	nrn := &ly.Neurons[tni]
	if nrn.IsOff() {
		return
	}
	geRaw := (1-drvInhib)*nrn.GeRaw + drvGe
	ly.Act.GeFromRaw(nrn, geRaw)
	ly.Act.GiFromRaw(nrn, nrn.GiRaw)
}

// SetDriverActs sets the driver activations, integrating across all the driver layers
func (ly *Layer) SetDriverActs() {
	nux, nuy := UnitsSize(ly)
	nun := nux * nuy
	pyn := ly.Shape.DimSize(0)
	pxn := ly.Shape.DimSize(1)
	for _, drv := range ly.Drivers {
		dly, err := ly.DriverLayer(drv)
		if err != nil {
			continue
		}
		issuper := dly.Type == SuperLayer
		drvMax := dly.Pools[0].Inhib.Act.Max
		drvInhib := math32.Min(1, drvMax/ly.Pulvinar.MaxInhib)

		if dly.Is2D() {
			if ly.Is2D() {
				for dni := range dly.Neurons {
					tni := drv.Off + dni
					drvAct := DriveAct(dni, dly, issuper)
					ly.SetDriverNeuron(tni, ly.Pulvinar.GeFromMaxAvg(drvAct, drvAct), drvInhib)
				}
			} else { // copy flat to all pools -- not typical
				for dni := range dly.Neurons {
					drvAct := DriveAct(dni, dly, issuper)
					tni := drv.Off + dni
					for py := 0; py < pyn; py++ {
						for px := 0; px < pxn; px++ {
							pni := (py*pxn+px)*nun + tni
							ly.SetDriverNeuron(pni, ly.Pulvinar.GeFromMaxAvg(drvAct, drvAct), drvInhib)
						}
					}
				}
			}
		} else { // dly is 4D
			dpyn := dly.Shape.DimSize(0)
			dpxn := dly.Shape.DimSize(1)
			duxn, duyn := UnitsSize(dly)
			dnun := duxn * duyn
			if ly.Is2D() {
				for dni := 0; dni < dnun; dni++ {
					max := float32(0)
					avg := float32(0)
					avgn := 0
					for py := 0; py < dpyn; py++ {
						for px := 0; px < dpxn; px++ {
							pi := (py*dpxn + px)
							pni := pi*dnun + dni
							act := DriveAct(pni, dly, issuper)
							max = math32.Max(max, act)
							pmax := dly.Pools[1+pi].Inhib.Act.Max
							if pmax > 0.5 {
								avg += act
								avgn++
							}
						}
					}
					if avgn > 0 {
						avg /= float32(avgn)
					}
					tni := drv.Off + dni
					ly.SetDriverNeuron(tni, ly.Pulvinar.GeFromMaxAvg(max, avg), drvInhib)
				}
			} else if ly.Pulvinar.NoTopo { // ly is 4D
				for dni := 0; dni < dnun; dni++ {
					max := float32(0)
					avg := float32(0)
					avgn := 0
					for py := 0; py < dpyn; py++ {
						for px := 0; px < dpxn; px++ {
							pi := (py*dpxn + px)
							pni := pi*dnun + dni
							act := DriveAct(pni, dly, issuper)
							max = math32.Max(max, act)
							pmax := dly.Pools[1+pi].Inhib.Act.Max
							if pmax > 0.5 {
								avg += act
								avgn++
							}
						}
					}
					if avgn > 0 {
						avg /= float32(avgn)
					}
					drvGe := ly.Pulvinar.GeFromMaxAvg(max, avg)
					tni := drv.Off + dni
					for py := 0; py < pyn; py++ {
						for px := 0; px < pxn; px++ {
							pni := (py*pxn+px)*nun + tni
							ly.SetDriverNeuron(pni, drvGe, drvInhib)
						}
					}
				}
			} else { // ly is 4D
				pyr := float64(dpyn) / float64(pyn)
				pxr := float64(dpxn) / float64(pxn)
				for py := 0; py < pyn; py++ {
					sdpy := int(math.Round(float64(py) * pyr))
					edpy := int(math.Round(float64(py+1) * pyr))
					for px := 0; px < pxn; px++ {
						sdpx := int(math.Round(float64(px) * pxr))
						edpx := int(math.Round(float64(px+1) * pxr))
						pni := (py*pxn + px) * nun
						for dni := 0; dni < dnun; dni++ {
							max := float32(0)
							avg := float32(0)
							avgn := 0
							for dpy := sdpy; dpy < edpy; dpy++ {
								for dpx := sdpx; dpx < edpx; dpx++ {
									pi := (dpy*dpxn + dpx)
									dpni := pi*dnun + dni
									act := DriveAct(dpni, dly, issuper)
									max = math32.Max(max, act)
									pmax := dly.Pools[1+pi].Inhib.Act.Max
									if pmax > 0.5 {
										avg += act
										avgn++
									}
								}
							}
							if avgn > 0 {
								avg /= float32(avgn)
							}
							tni := pni + drv.Off + dni
							ly.SetDriverNeuron(tni, ly.Pulvinar.GeFromMaxAvg(max, avg), drvInhib)
						}
					}
				}
			}
		}
	}
}
