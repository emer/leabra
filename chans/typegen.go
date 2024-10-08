// Code generated by "core generate -add-types"; DO NOT EDIT.

package chans

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "github.com/emer/leabra/v2/chans.Chans", IDName: "chans", Doc: "Chans are ion channels used in computing point-neuron activation function", Directives: []types.Directive{{Tool: "go", Directive: "generate", Args: []string{"core", "generate", "-add-types"}}}, Fields: []types.Field{{Name: "E", Doc: "excitatory sodium (Na) AMPA channels activated by synaptic glutamate"}, {Name: "L", Doc: "constant leak (potassium, K+) channels -- determines resting potential (typically higher than resting potential of K)"}, {Name: "I", Doc: "inhibitory chloride (Cl-) channels activated by synaptic GABA"}, {Name: "K", Doc: "gated / active potassium channels -- typically hyperpolarizing relative to leak / rest"}}})
