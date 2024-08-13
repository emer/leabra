// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/views"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/relpos"
)

// Layer implements the Leabra algorithm at the layer level,
// managing neurons and pathways.
type Layer struct {
	emer.LayerBase

	// our parent network, in case we need to use it to
	// find other layers etc; set when added by network
	Network *Network `copier:"-" json:"-" xml:"-" display:"-"`

	// type of layer.
	Type LayerType

	// list of receiving pathways into this layer from other layers
	RecvPaths []*Path

	// list of sending pathways from this layer to other layers
	SendPaths []*Path

	// Activation parameters and methods for computing activations
	Act ActParams `display:"add-fields"`

	// Inhibition parameters and methods for computing layer-level inhibition
	Inhib InhibParams `display:"add-fields"`

	// Learning parameters and methods that operate at the neuron level
	Learn LearnNeurParams `display:"add-fields"`

	// slice of neurons for this layer, as a flat list of len = Shape.Len().
	// Must iterate over index and use pointer to modify values.
	Neurons []Neuron

	// inhibition and other pooled, aggregate state variables.
	// flat list has at least of 1 for layer, and one for each sub-pool
	// if shape supports that (4D).
	// Must iterate over index and use pointer to modify values.
	Pools []Pool

	// cosine difference between ActM, ActP stats
	CosDiff CosDiffStats
}

// emer.Layer interface methods

func (ls *LayerBase) TypeName() string           { return ly.Type.String() }
func (ls *LayerBase) NumRecvPaths() int          { return len(ls.RecvPaths) }
func (ls *LayerBase) RecvPath(idx int) emer.Path { return ls.RecvPaths[idx] }
func (ls *LayerBase) NumSendPaths() int          { return len(ls.SendPaths) }
func (ls *LayerBase) SendPath(idx int) emer.Path { return ls.SendPaths[idx] }

func (ly *Layer) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Inhib.Layer.On = true
	for _, pj := range ly.RecvPaths {
		pj.Defaults()
	}
}

// UpdateParams updates all params given any changes that might have been made to individual values
// including those in the receiving pathways of this layer
func (ly *Layer) UpdateParams() {
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()
	for _, pj := range ly.RecvPaths {
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
	str := "/////////////////////////////////////////////////\nLayer: " + ly.Name + "\n"
	b, _ := json.MarshalIndent(&ly.Act, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)
	for _, pj := range ly.RecvPaths {
		pstr := pj.AllParams()
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
func (ls *LayerBase) RecipToSendPath(spj emer.Path) (emer.Path, bool) {
	for _, rpj := range ls.RecvPaths {
		if rpj.Send == spj.Recv {
			return rpj, true
		}
	}
	return nil, false
}

// ApplyParams applies given parameter style Sheet to this layer and its recv pathways.
// Calls UpdateParams on anything set to ensure derived parameters are all updated.
// If setMsg is true, then a message is printed to confirm each parameter that is set.
// it always prints a message if a parameter fails to be set.
// returns true if any params were set, and error if there were any errors.
func (ls *LayerBase) ApplyParams(pars *params.Sheet, setMsg bool) (bool, error) {
	applied := false
	var rerr error
	app, err := pars.Apply(ls.LeabraLay, setMsg) // essential to go through LeabraPrj
	if app {
		ls.LeabraLay.UpdateParams()
		applied = true
	}
	if err != nil {
		rerr = err
	}
	for _, pj := range ls.RecvPaths {
		app, err = pj.ApplyParams(pars, setMsg)
		if app {
			applied = true
		}
		if err != nil {
			rerr = err
		}
	}
	return applied, rerr
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
	vidx, err := ly.LeabraLay.UnitVarIndex(varNm)
	if err != nil {
		nan := math32.NaN()
		for i := range ly.Neurons {
			(*vals)[i] = nan
		}
		return err
	}
	for i := range ly.Neurons {
		(*vals)[i] = ly.LeabraLay.UnitValue1D(vidx, i, di)
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
	tsr.SetShape(ly.Shape.Shp, ly.Shape.Strd, ly.Shape.Nms)
	vidx, err := ly.LeabraLay.UnitVarIndex(varNm)
	if err != nil {
		nan := math.NaN()
		for i := range ly.Neurons {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i := range ly.Neurons {
		v := ly.LeabraLay.UnitValue1D(vidx, i, di)
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
		tsr.SetShape([]int{nu}, nil, []string{"Units"})
	}
	vidx, err := ly.LeabraLay.UnitVarIndex(varNm)
	if err != nil {
		nan := math.NaN()
		for i, _ := range ly.SampleIndexes {
			tsr.SetFloat1D(i, nan)
		}
		return err
	}
	for i, ui := range ly.SampleIndexes {
		v := ly.LeabraLay.UnitValue1D(vidx, ui, di)
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
	vidx, err := ly.LeabraLay.UnitVarIndex(varNm)
	if err != nil {
		return math32.NaN()
	}
	fidx := ly.Shape.Offset(idx)
	return ly.LeabraLay.UnitValue1D(vidx, fidx, di)
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
	var pj emer.Path
	if pathType != "" {
		pj, err = sendLay.RecvNameTypeTry(ly.Name, pathType)
		if pj == nil {
			pj, err = sendLay.RecvNameTry(ly.Name)
		}
	} else {
		pj, err = sendLay.RecvNameTry(ly.Name)
	}
	if pj == nil {
		return err
	}
	if pj.Off {
		return fmt.Errorf("pathway is off")
	}
	for ri := 0; ri < nn; ri++ {
		(*vals)[ri] = pj.SynValue(varNm, sendIndex1D, ri) // this will work with any variable -- slower, but necessary
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
	var pj emer.Path
	if pathType != "" {
		pj, err = recvLay.SendNameTypeTry(ly.Name, pathType)
		if pj == nil {
			pj, err = recvLay.SendNameTry(ly.Name)
		}
	} else {
		pj, err = recvLay.SendNameTry(ly.Name)
	}
	if pj == nil {
		return err
	}
	if pj.Off {
		return fmt.Errorf("pathway is off")
	}
	for si := 0; si < nn; si++ {
		(*vals)[si] = pj.SynValue(varNm, si, recvIndex1D)
	}
	return nil
}

// Pool returns pool at given index
func (ly *Layer) Pool(idx int) *Pool {
	return &(ly.Pools[idx])
}

//////////////////////////////////////////////////////////////////////////////////////
//  Build

// BuildSubPools initializes neuron start / end indexes for sub-pools
func (ly *Layer) BuildSubPools() {
	if !ly.Is4D() {
		return
	}
	sh := ly.Shape
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
	np := 1 + ly.NPools()
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
	for _, pj := range ly.RecvPaths {
		if pj.Off {
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

// Build constructs the layer state, including calling Build on the pathways
func (ly *Layer) Build() error {
	nu := ly.Shape.Len()
	if nu == 0 {
		return fmt.Errorf("Build Layer %v: no units specified in Shape", ly.Name)
	}
	ly.Neurons = make([]Neuron, nu)
	err := ly.BuildPools(nu)
	if err != nil {
		return err
	}
	err = ly.BuildPaths()
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
	w.Write([]byte(fmt.Sprintf("\"Layer\": %q,\n", ly.Name)))
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
	onps := make(emer.Paths, 0, len(ly.RecvPaths))
	for _, pj := range ly.RecvPaths {
		if !pj.Off {
			onps = append(onps, pj)
		}
	}
	np := len(onps)
	if np == 0 {
		w.Write([]byte(fmt.Sprintf("\"Paths\": null\n")))
	} else {
		w.Write([]byte(fmt.Sprintf("\"Paths\": [\n")))
		depth++
		for pi, pj := range onps {
			pj.WriteWtsJSON(w, depth) // this leaves path unterminated
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
			ly.Inhib.ActAvg.EffFmAvg(&pl.ActAvg.ActPAvgEff, pl.ActAvg.ActPAvg)
		}
	}
	var err error
	rpjs := ly.RecvPaths()
	if len(lw.Paths) == len(*rpjs) { // this is essential if multiple paths from same layer
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pj := (*rpjs)[pi]
			er := pj.SetWts(pw)
			if er != nil {
				err = er
			}
		}
	} else {
		for pi := range lw.Paths {
			pw := &lw.Paths[pi]
			pj, err := ly.SendNameTry(pw.From)
			if err == nil {
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
