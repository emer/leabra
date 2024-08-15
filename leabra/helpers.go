// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/core"
	"github.com/emer/emergent/v2/ecmd"
)

////////////////////////////////////////////////////
// Misc

// ToggleLayersOff can be used to disable layers in a Network, for example if you are doing an ablation study.
func ToggleLayersOff(net *Network, layerNames []string, off bool) {
	for _, lnm := range layerNames {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			fmt.Printf("layer not found: %s\n", lnm)
			continue
		}
		lyi.Off = off
	}
}

/////////////////////////////////////////////
// Weights files

// WeightsFilename returns default current weights file name,
// using train run and epoch counters from looper
// and the RunName string identifying tag, parameters and starting run,
func WeightsFilename(net *Network, ctrString, runName string) string {
	return net.Name + "_" + runName + "_" + ctrString + ".wts.gz"
}

// SaveWeights saves network weights to filename with WeightsFilename information
// to identify the weights.
// only for 0 rank MPI if running mpi
// Returns the name of the file saved to, or empty if not saved.
func SaveWeights(net *Network, ctrString, runName string) string {
	if mpi.WorldRank() > 0 {
		return ""
	}
	fnm := WeightsFilename(net, ctrString, runName)
	fmt.Printf("Saving Weights to: %s\n", fnm)
	net.SaveWeightsJSON(core.Filename(fnm))
	return fnm
}

// SaveWeightsIfArgSet saves network weights if the "wts" arg has been set to true.
// uses WeightsFilename information to identify the weights.
// only for 0 rank MPI if running mpi
// Returns the name of the file saved to, or empty if not saved.
func SaveWeightsIfArgSet(net *Network, args *ecmd.Args, ctrString, runName string) string {
	if args.Bool("wts") {
		return SaveWeights(net, ctrString, runName)
	}
	return ""
}

// SaveWeightsIfConfigSet saves network weights if the given config
// bool value has been set to true.
// uses WeightsFilename information to identify the weights.
// only for 0 rank MPI if running mpi
// Returns the name of the file saved to, or empty if not saved.
func SaveWeightsIfConfigSet(net *Network, cfgWts bool, ctrString, runName string) string {
	if cfgWts {
		return SaveWeights(net, ctrString, runName)
	}
	return ""
}
