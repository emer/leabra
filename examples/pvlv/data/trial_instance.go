package data

import (
	"github.com/emer/leabra/pvlv"
)

// A set of trial groups, sourced from an TrialParams list, instantiated according to the
// PercentOfTotal field in the source list.
// this is what we get after calling SetEpochTrialList
// Still not fully instantiated, US is still a probability
type TrialInstance struct {
	TrialName            string
	ValenceContext       pvlv.Valence
	USFlag               bool
	TestFlag             bool
	MixedUS              bool
	USProb               float64
	USMagnitude          float64
	AlphaTicksPerTrialGp int
	CS                   string
	CSTimeStart          int
	CSTimeEnd            int
	CS2TimeStart         int
	CS2TimeEnd           int
	USTimeStart          int
	USTimeEnd            int
	Context              string
	USType               string
}
