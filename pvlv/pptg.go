package pvlv

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"strconv"

	//"github.com/emer/leabra/pbwm"
	"github.com/goki/ki/kit"
)

type PPTgLayer struct {
	leabra.Layer
	Ge              float32
	GePrev          float32
	SendAct         float32
	DA              float32
	DNetGain        float32 `desc:"gain on input activation"`
	ActThreshold    float32 `desc:"activation threshold for passing through"`
	ClampActivation bool    `desc:"clamp activation directly, after applying gain"`
}

var KiT_PPTgLayer = kit.Types.AddType(&PPTgLayer{}, leabra.LayerProps)

func (ly *PPTgLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	return nil
}

func (ly *PPTgLayer) Defaults() {
	ly.Layer.Defaults()
}

func AddPPTgLayer(nt *Network, name string, nY, nX int) *PPTgLayer {
	rl := &PPTgLayer{}
	nt.AddLayerInit(rl, name, []int{nY, nX, 1, 1}, emer.Hidden)
	return rl
}

func (ly *PPTgLayer) InitActs() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.Act = 0
		nrn.ActSent = 0
	}
}

func (ly *PPTgLayer) GetDA() float32 {
	return ly.DA
}

func (ly *PPTgLayer) SetDA(da float32) {
	ly.DA = da
}

func (ly *PPTgLayer) QuarterFinal(ltime *leabra.Time) {
	ly.Layer.QuarterFinal(ltime)
	ly.Ge = ly.Neurons[0].Ge
	if ltime.PlusPhase {
		ly.GePrev = ly.Ge
	}
}

func (ly *PPTgLayer) GetMonitorVal(data []string) float64 {
	var val float32
	idx, _ := strconv.Atoi(data[1])
	switch data[0] {
	case "Act":
		val = ly.Neurons[idx].Act
	case "Ge":
		val = ly.Neurons[idx].Ge
	case "GePrev":
		val = ly.GePrev
	case "TotalAct":
		val = GlobalTotalActFn(ly)
	}
	return float64(val)
}

func (ly *PPTgLayer) ActFmG(ltime *leabra.Time) {
	nrn := &ly.Neurons[0]
	geSave := nrn.Ge
	nrn.Ge = ly.DNetGain * (nrn.Ge - ly.GePrev)
	if nrn.Ge < ly.ActThreshold {
		nrn.Ge = 0.0
	}
	ly.Ge = nrn.Ge
	ly.SendAct = nrn.Act // mainly for debugging
	nrn.Act = nrn.Ge
	nrn.ActLrn = nrn.Act
	nrn.ActDel = 0.0
	nrn.Ge = geSave
	ly.Learn.AvgsFmAct(nrn)
}
