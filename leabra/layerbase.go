// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"strconv"
	"strings"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/num"
	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/weights"
	"github.com/emer/etensor/tensor"
)

// Layer implements the Leabra algorithm at the layer level,
// managing neurons and pathways.
type Layer struct {
	emer.LayerBase

	// our parent network, in case we need to use it to
	// find other layers etc; set when added by network.
	Network *Network `copier:"-" json:"-" xml:"-" display:"-"`

	// type of layer.
	Type LayerTypes

	// list of receiving pathways into this layer from other layers.
	RecvPaths []*Path

	// list of sending pathways from this layer to other layers.
	SendPaths []*Path

	// Activation parameters and methods for computing activations.
	Act ActParams `display:"add-fields"`

	// Inhibition parameters and methods for computing layer-level inhibition.
	Inhib InhibParams `display:"add-fields"`

	// Learning parameters and methods that operate at the neuron level.
	Learn LearnNeurParams `display:"add-fields"`

	// Burst has parameters for computing Burst from act, in Superficial layers
	// (but also needed in Deep layers for deep self connections).
	Burst BurstParams `display:"inline"`

	// Pulvinar has parameters for computing Pulvinar plus-phase (outcome)
	// activations based on Burst activation from corresponding driver neuron.
	Pulvinar PulvinarParams `display:"inline"`

	// Drivers are names of SuperLayer(s) that sends 5IB Burst driver
	// inputs to this layer.
	Drivers Drivers

	// RW are Rescorla-Wagner RL learning parameters.
	RW RWParams `display:"inline"`

	// TD are Temporal Differences RL learning parameters.
	TD TDParams `display:"inline"`

	// Matrix BG gating parameters
	Matrix MatrixParams `display:"inline"`

	// PBWM has general PBWM parameters, including the shape
	// of overall Maint + Out gating system that this layer is part of.
	PBWM PBWMParams `display:"inline"`

	// GPiGate are gating parameters determining threshold for gating etc.
	GPiGate GPiGateParams `display:"inline"`

	// CIN cholinergic interneuron parameters.
	CIN CINParams `display:"inline"`

	// PFC Gating parameters
	PFCGate PFCGateParams `display:"inline"`

	// PFC Maintenance parameters
	PFCMaint PFCMaintParams `display:"inline"`

	// PFCDyns dynamic behavior parameters -- provides deterministic control over PFC maintenance dynamics -- the rows of PFC units (along Y axis) behave according to corresponding index of Dyns (inner loop is Super Y axis, outer is Dyn types) -- ensure Y dim has even multiple of len(Dyns)
	PFCDyns PFCDyns

	// slice of neurons for this layer, as a flat list of len = Shape.Len().
	// Must iterate over index and use pointer to modify values.
	Neurons []Neuron

	// inhibition and other pooled, aggregate state variables.
	// flat list has at least of 1 for layer, and one for each sub-pool
	// if shape supports that (4D).
	// Must iterate over index and use pointer to modify values.
	Pools []Pool

	// cosine difference between ActM, ActP stats.
	CosDiff CosDiffStats

	// NeuroMod is the neuromodulatory neurotransmitter state for this layer.
	NeuroMod NeuroMod `read-only:"+" display:"inline"`

	// SendTo is a list of layers that this layer sends special signals to,
	// which could be dopamine, gating signals, depending on the layer type.
	SendTo LayerNames
}

// emer.Layer interface methods

func (ly *Layer) StyleObject() any           { return ly }
func (ly *Layer) TypeName() string           { return ly.Type.String() }
func (ly *Layer) TypeNumber() int            { return int(ly.Type) }
func (ly *Layer) NumRecvPaths() int          { return len(ly.RecvPaths) }
func (ly *Layer) RecvPath(idx int) emer.Path { return ly.RecvPaths[idx] }
func (ly *Layer) NumSendPaths() int          { return len(ly.SendPaths) }
func (ly *Layer) SendPath(idx int) emer.Path { return ly.SendPaths[idx] }

func (ly *Layer) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Burst.Defaults()
	ly.Pulvinar.Defaults()
	ly.RW.Defaults()
	ly.TD.Defaults()
	ly.Matrix.Defaults()
	ly.PBWM.Defaults()
	ly.GPiGate.Defaults()
	ly.CIN.Defaults()
	ly.PFCGate.Defaults()
	ly.PFCMaint.Defaults()
	ly.Inhib.Layer.On = true
	for _, pt := range ly.RecvPaths {
		pt.Defaults()
	}
	ly.DefaultsForType()
}

// DefaultsForType sets the default parameter values for a given layer type.
func (ly *Layer) DefaultsForType() {
	switch ly.Type {
	case ClampDaLayer:
		ly.ClampDaDefaults()
	case MatrixLayer:
		ly.MatrixDefaults()
	case GPiThalLayer:
		ly.GPiThalDefaults()
	case CINLayer:
	case PFCLayer:
	case PFCDeepLayer:
		ly.PFCDeepDefaults()
	}
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving pathways of this layer
func (ly *Layer) UpdateParams() {
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()
	ly.Burst.Update()
	ly.Pulvinar.Update()
	ly.RW.Update()
	ly.TD.Update()
	ly.Matrix.Update()
	ly.PBWM.Update()
	ly.GPiGate.Update()
	ly.CIN.Update()
	ly.PFCGate.Update()
	ly.PFCMaint.Update()
	for _, pt := range ly.RecvPaths {
		pt.UpdateParams()
	}
}

func (ly *Layer) ShouldDisplay(field string) bool {
	isPBWM := ly.Type == MatrixLayer || ly.Type == GPiThalLayer || ly.Type == CINLayer || ly.Type == PFCLayer || ly.Type == PFCDeepLayer
	switch field {
	case "Burst":
		return ly.Type == SuperLayer || ly.Type == CTLayer
	case "Pulvinar", "Drivers":
		return ly.Type == PulvinarLayer
	case "RW":
		return ly.Type == RWPredLayer || ly.Type == RWDaLayer
	case "TD":
		return ly.Type == TDPredLayer || ly.Type == TDIntegLayer || ly.Type == TDDaLayer
	case "PBWM":
		return isPBWM
	case "SendTo":
		return ly.Type == GPiThalLayer || ly.Type == ClampDaLayer || ly.Type == RWDaLayer || ly.Type == TDDaLayer || ly.Type == CINLayer
	case "Matrix":
		return ly.Type == MatrixLayer
	case "GPiGate":
		return ly.Type == GPiThalLayer
	case "CIN":
		return ly.Type == CINLayer
	case "PFCGate", "PFCMaint":
		return ly.Type == PFCLayer || ly.Type == PFCDeepLayer
	case "PFCDyns":
		return ly.Type == PFCDeepLayer
	default:
		return true
	}
	return true
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
	str := "/////////////////////////////////////////////////\nLayer: " + ly.Name + "\n"
	b, _ := json.MarshalIndent(&ly.Act, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)
	for _, pt := range ly.RecvPaths {
		pstr := pt.AllParams()
		str += pstr
	}
	return str
}

// RecipToSendPath finds the reciprocal pathway relative to the given sending pathway
// found within the SendPaths of this layer.  This is then a recv path within this layer:
//
//	S=A -> R=B recip: R=A <- S=B -- ly = A -- we are the sender of srj and recv of rpj.
//
// returns false if not found.
func (ly *Layer) RecipToSendPath(spj *Path) (*Path, bool) {
	for _, rpj := range ly.RecvPaths {
		if rpj.Send == spj.Recv {
			return rpj, true
		}
	}
	return nil, false
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (ly *Layer) UnitVarNames() []string {
	return NeuronVars
}

// UnitVarProps returns properties for variables
func (ly *Layer) UnitVarProps() map[string]string {
	return NeuronVarProps
}

// UnitVarIndex returns the index of given variable within the Neuron,
// according to *this layer's* UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIndex(varNm string) (int, error) {
	return NeuronVarIndexByName(varNm)
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return len(NeuronVars)
}

// UnitValue1D returns value of given variable index on given unit,
// using 1-dimensional index. returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitValue1D(varIndex int, idx int, di int) float32 {
	if idx < 0 || idx >= len(ly.Neurons) {
		return math32.NaN()
	}
	if varIndex < 0 || varIndex >= ly.UnitVarNum() {
		return math32.NaN()
	}
	nrn := &ly.Neurons[idx]
	da := NeuronVarsMap["DA"]
	if varIndex >= da {
		switch varIndex - da {
		case 0:
			return ly.NeuroMod.DA
		case 1:
			return ly.NeuroMod.ACh
		case 2:
			return ly.NeuroMod.SE
		case 3:
			return ly.Pools[nrn.SubPool].Gate.Act
		case 4:
			return num.FromBool[float32](ly.Pools[nrn.SubPool].Gate.Now)
		case 5:
			return float32(ly.Pools[nrn.SubPool].Gate.Cnt)
		}
	}
	return nrn.VarByIndex(varIndex)
}

// UnitValues fills in values of given variable name on unit,
// for each unit in the layer, into given float32 slice (only resized if not big enough).
// Returns error on invalid var name.
func (ly *Layer) UnitValues(vals *[]float32, varNm string, di int) error {
	nn := len(ly.Neurons)
	if *vals == nil || cap(*vals) < nn {
		*vals = make([]float32, nn)
	} else if len(*vals) < nn {
		*vals = (*vals)[0:nn]
	}
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		nan := math32.NaN()
		for i := range ly.Neurons {
			(*vals)[i] = nan
		}
		return err
	}
	for i := range ly.Neurons {
		(*vals)[i] = ly.UnitValue1D(vidx, i, di)
	}
	return nil
}

// UnitValuesTensor returns values of given variable name on unit
// for each unit in the layer, as a float32 tensor in same shape as layer units.
func (ly *Layer) UnitValuesTensor(tsr tensor.Tensor, varNm string, di int) error {
	if tsr == nil {
		err := fmt.Errorf("leabra.UnitValuesTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	tsr.SetShape(ly.Shape.Sizes, ly.Shape.Names...)
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		nan := math.NaN()
		for i := range ly.Neurons {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i := range ly.Neurons {
		v := ly.UnitValue1D(vidx, i, di)
		if math32.IsNaN(v) {
			tsr.SetFloat1D(i, math.NaN())
		} else {
			tsr.SetFloat1D(i, float64(v))
		}
	}
	return nil
}

// UnitValuesSampleTensor fills in values of given variable name on unit
// for a smaller subset of sample units in the layer, into given tensor.
// This is used for computationally intensive stats or displays that work
// much better with a smaller number of units.
// The set of sample units are defined by SampleIndexes -- all units
// are used if no such subset has been defined.
// If tensor is not already big enough to hold the values, it is
// set to a 1D shape to hold all the values if subset is defined,
// otherwise it calls UnitValuesTensor and is identical to that.
// Returns error on invalid var name.
func (ly *Layer) UnitValuesSampleTensor(tsr tensor.Tensor, varNm string, di int) error {
	nu := len(ly.SampleIndexes)
	if nu == 0 {
		return ly.UnitValuesTensor(tsr, varNm, di)
	}
	if tsr == nil {
		err := fmt.Errorf("axon.UnitValuesSampleTensor: Tensor is nil")
		log.Println(err)
		return err
	}
	if tsr.Len() != nu {
		tsr.SetShape([]int{nu}, "Units")
	}
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		nan := math.NaN()
		for i, _ := range ly.SampleIndexes {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i, ui := range ly.SampleIndexes {
		v := ly.UnitValue1D(vidx, ui, di)
		if math32.IsNaN(v) {
			tsr.SetFloat1D(i, math.NaN())
		} else {
			tsr.SetFloat1D(i, float64(v))
		}
	}
	return nil
}

// UnitVal returns value of given variable name on given unit,
// using shape-based dimensional index
func (ly *Layer) UnitValue(varNm string, idx []int, di int) float32 {
	vidx, err := ly.UnitVarIndex(varNm)
	if err != nil {
		return math32.NaN()
	}
	fidx := ly.Shape.Offset(idx)
	return ly.UnitValue1D(vidx, fidx, di)
}

// RecvPathValues fills in values of given synapse variable name,
// for pathway into given sending layer and neuron 1D index,
// for all receiving neurons in this layer,
// into given float32 slice (only resized if not big enough).
// pathType is the string representation of the path type -- used if non-empty,
// useful when there are multiple pathways between two layers.
// Returns error on invalid var name.
// If the receiving neuron is not connected to the given sending layer or neuron
// then the value is set to math32.NaN().
// Returns error on invalid var name or lack of recv path
// (vals always set to nan on path err).
func (ly *Layer) RecvPathValues(vals *[]float32, varNm string, sendLay emer.Layer, sendIndex1D int, pathType string) error {
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
	slay := sendLay.AsEmer()
	var pt emer.Path
	if pathType != "" {
		pt, err = slay.SendPathByRecvNameType(ly.Name, pathType)
		if pt == nil {
			pt, err = slay.SendPathByRecvName(ly.Name)
		}
	} else {
		pt, err = slay.SendPathByRecvName(ly.Name)
	}
	if pt == nil {
		return err
	}
	if pt.AsEmer().Off {
		return fmt.Errorf("pathway is off")
	}
	for ri := 0; ri < nn; ri++ {
		(*vals)[ri] = pt.AsEmer().SynValue(varNm, sendIndex1D, ri) // this will work with any variable -- slower, but necessary
	}
	return nil
}

// SendPathValues fills in values of given synapse variable name,
// for pathway into given receiving layer and neuron 1D index,
// for all sending neurons in this layer,
// into given float32 slice (only resized if not big enough).
// pathType is the string representation of the path type -- used if non-empty,
// useful when there are multiple pathways between two layers.
// Returns error on invalid var name.
// If the sending neuron is not connected to the given receiving layer or neuron
// then the value is set to math32.NaN().
// Returns error on invalid var name or lack of recv path
// (vals always set to nan on path err).
func (ly *Layer) SendPathValues(vals *[]float32, varNm string, recvLay emer.Layer, recvIndex1D int, pathType string) error {
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
	rlay := recvLay.AsEmer()
	var pt emer.Path
	if pathType != "" {
		pt, err = rlay.RecvPathBySendNameType(ly.Name, pathType)
		if pt == nil {
			pt, err = rlay.RecvPathBySendName(ly.Name)
		}
	} else {
		pt, err = rlay.RecvPathBySendName(ly.Name)
	}
	if pt == nil {
		return err
	}
	if pt.AsEmer().Off {
		return fmt.Errorf("pathway is off")
	}
	for si := 0; si < nn; si++ {
		(*vals)[si] = pt.AsEmer().SynValue(varNm, si, recvIndex1D)
	}
	return nil
}

// Pool returns pool at given index
func (ly *Layer) Pool(idx int) *Pool {
	return &(ly.Pools[idx])
}

// AddSendTo adds given layer name(s) to list of those to send to.
func (ly *Layer) AddSendTo(laynm ...string) {
	ly.SendTo.Add(laynm...)
}

// AddAllSendToBut adds all layers in network except those in exclude list.
func (ly *Layer) AddAllSendToBut(excl ...string) {
	ly.SendTo.AddAllBut(ly.Network, excl...)
}

// ValidateSendTo ensures that SendTo layer names are valid.
func (ly *Layer) ValidateSendTo() error {
	return ly.SendTo.Validate(ly.Network)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Build

// BuildSubPools initializes neuron start / end indexes for sub-pools
func (ly *Layer) BuildSubPools() {
	if !ly.Is4D() {
		return
	}
	sh := ly.Shape.Sizes
	spy := sh[0]
	spx := sh[1]
	pi := 1
	for py := 0; py < spy; py++ {
		for px := 0; px < spx; px++ {
			soff := ly.Shape.Offset([]int{py, px, 0, 0})
			eoff := ly.Shape.Offset([]int{py, px, sh[2] - 1, sh[3] - 1}) + 1
			pl := &ly.Pools[pi]
			pl.StIndex = soff
			pl.EdIndex = eoff
			for ni := pl.StIndex; ni < pl.EdIndex; ni++ {
				nrn := &ly.Neurons[ni]
				nrn.SubPool = int32(pi)
			}
			pi++
		}
	}
}

// BuildPools builds the inhibitory pools structures -- nu = number of units in layer
func (ly *Layer) BuildPools(nu int) error {
	np := 1 + ly.NumPools()
	ly.Pools = make([]Pool, np)
	lpl := &ly.Pools[0]
	lpl.StIndex = 0
	lpl.EdIndex = nu
	if np > 1 {
		ly.BuildSubPools()
	}
	return nil
}

// BuildPaths builds the pathways, recv-side
func (ly *Layer) BuildPaths() error {
	emsg := ""
	for _, pt := range ly.RecvPaths {
		if pt.Off {
			continue
		}
		err := pt.Build()
		if err != nil {
			emsg += err.Error() + "\n"
		}
	}
	if emsg != "" {
		return errors.New(emsg)
	}
	return nil
}

// Build constructs the layer state, including calling Build on the pathways
func (ly *Layer) Build() error {
	nu := ly.Shape.Len()
	if nu == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Name)
	}
	ly.Neurons = make([]Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return errors.Log(err)
	}
	err = ly.BuildPaths()
	if err != nil {
		return errors.Log(err)
	}
	err = ly.ValidateSendTo()
	if err != nil {
		return errors.Log(err)
	}
	err = ly.CIN.RewLays.Validate(ly.Network)
	if err != nil {
		return errors.Log(err)
	}
	return nil
}

// WriteWeightsJSON writes the weights from this layer from the receiver-side perspective
// in a JSON text format.  We build in the indentation logic to make it much faster and
// more efficient.
func (ly *Layer) WriteWeightsJSON(w io.Writer, depth int) {
	ly.MetaData = make(map[string]string)
	ly.MetaData["ActMAvg"] = fmt.Sprintf("%g", ly.Pools[0].ActAvg.ActMAvg)
	ly.MetaData["ActPAvg"] = fmt.Sprintf("%g", ly.Pools[0].ActAvg.ActPAvg)
	ly.LayerBase.WriteWeightsJSONBase(w, depth)
}

// SetWeights sets the weights for this layer from weights.Layer decoded values
func (ly *Layer) SetWeights(lw *weights.Layer) error {
	if ly.Off {
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
			ly.Inhib.ActAvg.EffFromAvg(&pl.ActAvg.ActPAvgEff, pl.ActAvg.ActPAvg)
		}
	}
	var err error
	rpts := ly.RecvPaths
	if len(lw.Paths) == len(rpts) { // this is essential if multiple paths from same layer
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pt := (rpts)[pi]
			er := pt.SetWeights(pw)
			if er != nil {
				err = er
			}
		}
	} else {
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pt, err := ly.RecvPathBySendName(pw.From)
			if err == nil {
				er := pt.SetWeights(pw)
				if er != nil {
					err = er
				}
			}
		}
	}
	return err
}

// VarRange returns the min / max values for given variable
// todo: support r. s. pathway values
func (ly *Layer) VarRange(varNm string) (min, max float32, err error) {
	sz := len(ly.Neurons)
	if sz == 0 {
		return
	}
	vidx := 0
	vidx, err = NeuronVarIndexByName(varNm)
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
