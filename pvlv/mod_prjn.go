package pvlv

import (
	"github.com/chewxy/math32"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
)

type IModPrjn interface {
	AsModPrjn() *ModHebbPrjn
}

func (pj *ModHebbPrjn) AsModPrjn() *ModHebbPrjn {
	return pj
}

type ISetScalePrjn interface {
	InitWtsScale()
}

type ModHebbPrjn struct {
	leabra.Prjn
	SetScale    bool    `desc:"only for Leabra algorithm: if initializing the weights, set the connection scaling parameter in addition to intializing the weights -- for specifically-supported specs, this will for example set a gaussian scaling parameter on top of random initial weights, instead of just setting the initial weights to a gaussian weighted value -- for other specs that do not support a custom init_wts function, this will set the scale values to what the random weights would otherwise be set to, and set the initial weight value to a constant (init_wt_val)"`
	SetScaleMin float32 `desc:"minimum scale value for SetScale projections"`
	SetScaleMax float32 `desc:"maximum scale value for SetScale projections"`
	InitWtVal   float32 `desc:"constant initial weight value for specs that do not support a custom init_wts function and have set_scale set: the scale values are set to what the random weights would otherwise be set to, and the initial weight value is set to this constant: the net actual weight value is scale * init_wt_val.."`
	DALRGain    float32 `desc:"gain multiplier on abs(DA) learning rate multiplier"`
	DALRBase    float32 `desc:"constant baseline amount of learning prior to abs(DA) factor -- should be near zero otherwise offsets in activation will drive learning in the absence of DA significance"`
	DALrnThr    float32 `desc:"minimum threshold for phasic abs(da) signals to count as non-zero;  useful to screen out spurious da signals due to tiny VSPatch-to-LHb signals on t2 & t4 timesteps that can accumulate over many trials - 0.02 seems to work okay"`
	ActDeltaThr float32 `desc:"minimum threshold for delta activation to count as non-zero;  useful to screen out spurious learning due to unintended delta activity - 0.02 seems to work okay for both acquisition and extinction guys"`
	ActLrnMod   bool    `desc:"if true, recv unit deep_lrn value modulates learning"`
	ActLrnThr   float32 `desc:"only ru->deep_lrn values > this get to learn - 0.05f seems to work okay"`
}

func (pj *ModHebbPrjn) InitWts() {
	if pj.SetScale {
		pj.SetScalesFunc(pj.GaussScale)
		pj.SetWtsFunc(func(_, _ int, _, _ *etensor.Shape) float32 {
			return pj.InitWtVal
		})
		for si := range pj.Syns {
			sy := &pj.Syns[si]
			sy.DWt = 0
			sy.Norm = 0
			sy.Moment = 0
		}
	} else {
		pj.Prjn.InitWts()
	}
}

// GaussScale returns gaussian weight value for given unit indexes in
// given send and recv layers according to Gaussian Sigma and MaxWt.
func (pj *ModHebbPrjn) GaussScale(_, _ int, _, _ *etensor.Shape) float32 {
	scale := float32(pj.WtInit.Gen(-1))
	scale = math32.Max(pj.SetScaleMin, scale)
	scale = math32.Min(pj.SetScaleMax, scale)
	return scale
}
