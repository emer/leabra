// Code generated by "core generate -add-types"; DO NOT EDIT.

package knadapt

import (
	"cogentcore.org/core/types"
)

var _ = types.AddType(&types.Type{Name: "github.com/emer/leabra/v2/knadapt.Chan", IDName: "chan", Doc: "Chan describes one channel type of sodium-gated adaptation, with a specific\nset of rate constants.", Directives: []types.Directive{{Tool: "go", Directive: "generate", Args: []string{"core", "generate", "-add-types"}}}, Fields: []types.Field{{Name: "On", Doc: "if On, use this component of K-Na adaptation"}, {Name: "Rise", Doc: "Rise rate of fast time-scale adaptation as function of Na concentration -- directly multiplies -- 1/rise = tau for rise rate"}, {Name: "Max", Doc: "Maximum potential conductance of fast K channels -- divide nA biological value by 10 for the normalized units here"}, {Name: "Tau", Doc: "time constant in cycles for decay of adaptation, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life)"}, {Name: "Dt", Doc: "1/Tau rate constant"}}})

var _ = types.AddType(&types.Type{Name: "github.com/emer/leabra/v2/knadapt.Params", IDName: "params", Doc: "Params describes sodium-gated potassium channel adaptation mechanism.\nEvidence supports at least 3 different time constants:\nM-type (fast), Slick (medium), and Slack (slow)", Fields: []types.Field{{Name: "On", Doc: "if On, apply K-Na adaptation"}, {Name: "Rate", Doc: "extra multiplier for rate-coded activations on rise factors -- adjust to match discrete spiking"}, {Name: "Fast", Doc: "fast time-scale adaptation"}, {Name: "Med", Doc: "medium time-scale adaptation"}, {Name: "Slow", Doc: "slow time-scale adaptation"}}})
