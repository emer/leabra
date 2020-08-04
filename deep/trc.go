// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"fmt"
	"log"

	"github.com/chewxy/math32"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
)

// TRCParams provides parameters for how the plus-phase (outcome) state of thalamic relay cell
// (e.g., Pulvinar) neurons is computed from the corresponding driver neuron Burst activation.
type TRCParams struct {
	BurstQtr   leabra.Quarters `desc:"Quarter(s) when bursting occurs -- typically Q4 but can also be Q2 and Q4 for beta-frequency updating.  Note: this is a bitflag and must be accessed using its Set / Has etc routines"`
	DriveScale float32         `def:"0.3" min:"0.0" desc:"multiplier on driver input strength, multiplies activation of driver layer"`
	MaxInhib   float32         `def:"0.6" min:"0.01" desc:"Level of pool Max driver activation at which the predictive non-burst inputs are fully inhibited (see InhibPool for option on what level of pooling this is computed over).  Computationally, it is essential that driver inputs inhibit effect of predictive non-driver (CTLayer) inputs, so that the plus phase is not always just the minus phase plus something extra (the error will never go to zero then).  When max driver act input exceeds this value, predictive non-driver inputs are fully suppressed.  If there is only weak burst input however, then the predictive inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning."`
	InhibPool  bool            `desc:"For the MaxInhib mechanism, if this is true then the Max driver activation value comes from the specific pool (if sub-pools exist in layer) -- otherwise it comes from the entire layer."`
	Binarize   bool            `desc:"Apply threshold to driver burst input for computing plus-phase activations -- above BinThr, then Act = BinOn, below = BinOff.  This is beneficial for layers with weaker graded activations, such as V1 or other perceptual inputs."`
	BinThr     float32         `viewif:"Binarize" desc:"Threshold for binarizing in terms of sending Burst activation"`
	BinOn      float32         `def:"0.3" viewif:"Binarize" desc:"Resulting driver Ge value for units above threshold -- lower value around 0.3 or so seems best (DriveScale is NOT applied -- generally same range as that)."`
	BinOff     float32         `def:"0" viewif:"Binarize" desc:"Resulting driver Ge value for units below threshold -- typically 0."`
}

func (tp *TRCParams) Update() {
}

func (tp *TRCParams) Defaults() {
	tp.BurstQtr.Set(int(leabra.Q4))
	tp.DriveScale = 0.3
	tp.MaxInhib = 0.6
	tp.InhibPool = false
	tp.Binarize = false
	tp.BinThr = 0.4
	tp.BinOn = 0.3
	tp.BinOff = 0
}

// DriveGe returns effective excitatory conductance to use for given driver input Burst activation
func (tp *TRCParams) DriveGe(act float32) float32 {
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

// TRCLayer is the thalamic relay cell layer for DeepLeabra.
// It has normal activity during the minus phase, as activated by non-driver inputs
// and is then driven by strong 5IB driver inputs in the plus phase, which are
// directly copied from a named DriverLay layer (not using a projection).
// The DriverLay MUST have the same shape as this TRC layer, including Pools!
type TRCLayer struct {
	leabra.Layer           // access as .Layer
	TRC          TRCParams `view:"inline" desc:"parameters for computing TRC plus-phase (outcome) activations based on Burst activation from corresponding driver neuron"`
	DriverLay    string    `desc:"name of SuperLayer that sends 5IB Burst driver inputs to this layer"`
}

var KiT_TRCLayer = kit.Types.AddType(&TRCLayer{}, LayerProps)

func (ly *TRCLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Act.Init.Decay = 0 // deep doesn't decay!
	ly.TRC.Defaults()
	ly.Typ = TRC
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving projections of this layer
func (ly *TRCLayer) UpdateParams() {
	ly.Layer.UpdateParams()
	ly.TRC.Update()
}

func (ly *TRCLayer) Class() string {
	return "TRC " + ly.Cls
}

// DriverLayer returns the driver SuperLayer based on DriverLay name
func (ly *TRCLayer) DriverLayer() (*leabra.Layer, error) {
	tly, err := ly.Network.LayerByNameTry(ly.DriverLay)
	if err != nil {
		err = fmt.Errorf("TRCLayer %s, DriverLay: %v", ly.Name(), err)
		log.Println(err)
		return nil, err
	}
	return tly.(leabra.LeabraLayer).AsLeabra(), nil
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GFmInc integrates new synaptic conductances from increments sent during last SendGDelta.
func (ly *TRCLayer) GFmInc(ltime *leabra.Time) {
	ly.RecvGInc(ltime)
	if !ly.TRC.BurstQtr.Has(ltime.Quarter) {
		ly.GFmIncNeur(ltime) // regular
		return
	}
	dly, err := ly.DriverLayer()
	if err != nil {
		ly.GFmIncNeur(ltime) // regular
		return
	}
	sly, issuper := dly.LeabraLay.(*SuperLayer)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		drvMax := float32(0)
		if ly.TRC.InhibPool {
			drvMax = dly.Pools[nrn.SubPool].Inhib.Act.Max
		} else {
			drvMax = dly.Pools[0].Inhib.Act.Max
		}
		drvAct := float32(0)
		if issuper {
			drvAct = sly.SuperNeurs[ni].Burst
		} else {
			drvAct = dly.Neurons[ni].Act
		}
		drvGe := ly.TRC.DriveGe(drvAct)

		drvInhib := math32.Min(1, drvMax/ly.TRC.MaxInhib)
		geRaw := (1-drvInhib)*nrn.GeRaw + drvGe
		ly.Act.GeFmRaw(nrn, geRaw)
		ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	}
}
