package pvlv

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/leabra/leabra"
	"github.com/goki/ki/kit"
	"strings"
)

type Network struct {
	leabra.Network
}

var NetworkProps = leabra.NetworkProps
var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

type INetwork interface {
	AsLeabra() *leabra.Network
}

func (nt *Network) AsLeabra() *leabra.Network {
	return &nt.Network
}

func (nt *Network) InitActs() {
	for li := range nt.Layers {
		ly := nt.Layers[li]
		ly.(leabra.LeabraLayer).InitActs()
	}
}

func (nt *Network) Cycle(ltime *leabra.Time) {
	nt.CycleImpl(ltime)
	nt.EmerNet.(leabra.LeabraNetwork).CyclePostImpl(ltime) // always call this after std cycle..
}

func (nt *Network) CycleImpl(ltime *leabra.Time) {
	nt.QuarterInitPrvs(ltime)
	nt.SendGDelta(ltime) // also does integ
	nt.SendMods(ltime)
	nt.RecvModInc(ltime)
	nt.AvgMaxGe(ltime)
	nt.AvgMaxMod(ltime)
	nt.InhibFmGeAct(ltime)
	nt.ActFmG(ltime)
	nt.AvgMaxAct(ltime)
}

func (nt *Network) QuarterInitPrvs(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(*MSNLayer); ok {
			pl.QuarterInitPrvs(ltime)
		}
	}, "QuarterInitPrvs")
}

func (nt *Network) SendMods(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(ModSender); ok {
			pl.SendMods(ltime)
		}
	}, "SendMods")
}

func (nt *Network) RecvModInc(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(ModReceiver); ok {
			pl.ModsFmInc(ltime)
		}
	}, "RecvModInc")
}

func (nt *Network) ClearModActs(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if ml, ok := ly.(IModLayer); ok {
			ml.AsMod().ClearModActs()
		}
	}, "ClearModActs")
}

func (nt *Network) ClearMSNTraces(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if msnly, ok := ly.(IMSNLayer); ok {
			msnly.AsMSNLayer().ClearMSNTrace()
		}
	}, "ClearMSNTraces")
}

func (nt *Network) AvgMaxMod(ltime *leabra.Time) {
	nt.ThrLayFun(func(ly leabra.LeabraLayer) {
		if pl, ok := ly.(AvgMaxModLayer); ok {
			pl.AvgMaxMod(ltime)
		}
	}, "AvgMaxMod")
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return ModNeuronVarsAll
}

func (nt *Network) SynVarNames() []string {
	return SynapseVarsAll
}

// SynVarProps returns properties for variables
func (nt *Network) SynVarProps() map[string]string {
	return SynapseVarProps
}

// For special layer types

func (nt *Network) AddVTALayer(name string, val Valence) *VTALayer {
	ly := &VTALayer{Valence: val}
	nt.AddLayerInit(ly, name, []int{1, 1}, emer.Hidden)
	return ly
}

// AddMatrixLayer adds a MSNLayer of given size, with given name.
// nY = number of pools in Y dimension, nMaint + nOut are pools in X dimension,
// and each pool has nNeurY, nNeurX neurons.  da gives the DaReceptor type (D1R = Go, D2R = NoGo)
func (nt *Network) AddMSNLayer(name string, nY, nMaint, nOut, nNeurY, nNeurX int, cpmt StriatalCompartment, da DaRType) *MSNLayer {
	tX := nMaint + nOut
	stri := &MSNLayer{}
	nt.AddLayerInit(stri, name, []int{nY, tX, nNeurY, nNeurX}, emer.Hidden)
	stri.ModLayer.Init()
	stri.DaRType = da
	stri.Compartment = cpmt
	return stri
}

// Add a CentroLateral Amygdala layer with specified 2D geometry, acquisition/extinction, valence, and DA receptor type
func (nt *Network) AddCElAmygLayer(name string, nY, nX, nNeurY, nNeurX int,
	acqExt AcqExt, val Valence, dar DaRType) *CElAmygLayer {
	ly := CElAmygLayer{CElTyp: CElAmygLayerType{AcqExt: acqExt, Valence: val}}
	ly.DaRType = dar
	nt.AddLayerInit(&ly, name, []int{nY, nX, nNeurY, nNeurX}, emer.Hidden)
	ly.ModLayer.Init()
	class := "CEl" + acqExt.String() + strings.Title(strings.ToLower(val.String())) + dar.String()[0:2] + " CElAmyg"
	ly.SetClass(class)
	return &ly
}

func (nt *Network) AddBlAmygLayer(name string, nY, nX, nNeurY, nNeurX int, val Valence, dar DaRType, lTyp emer.LayerType) *BlAmygLayer {
	ly := BlAmygLayer{Valence: val}
	ly.DaRType = dar
	nt.AddLayerInit(&ly, name, []int{nY, nX, nNeurY, nNeurX}, lTyp)
	ly.ModLayer.Init()
	class := "BlAmyg" + strings.Title(strings.ToLower(val.String())) + dar.String()[0:2] + " BlAmyg"
	ly.SetClass(class)
	return &ly
}

func (nt *Network) ConnectLayersActMod(sender ModSender, rcvr ModReceiver, scale float32) {
	sender.(IModLayer).AsMod().AddModReceiver(rcvr, scale)
}
