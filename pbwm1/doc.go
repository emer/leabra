// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package pbwm provides the prefrontal cortex basal ganglia working memory (PBWM)
model of the basal ganglia (BG) and prefrontal cortex (PFC) circuitry that
supports dynamic BG gating of PFC robust active maintenance.

This package builds on the deep package for defining thalamocortical circuits
involved in predictive learning -- the BG basically acts to gate these circuits.

It provides a basis for dopamine-modulated processing of all types, and is the
base package for the PVLV model package built on top of it.

There are multiple levels of functionality to allow for flexibility in
exploring new variants.

Each different Layer type defines and manages its own Neuron type, despite some
redundancy, so only one type is needed and it is exactly what that layer needs.
However, a Network must have a single consistent set of Neuron variables, which
is given by ModNeuronVars and NeuronVars enum.  In many cases, those "neuron"
variables are actually stored in the layer itself instead of on per-neuron level.

Naming rule: DA when a singleton, DaMod (lowercase a) when CamelCased with something else

##############
# Basic Level

* ModLayer has DA, ACh, SE -- can be modulated

* DaSrcLayer sends DA to a list of layers (does not use Prjns)
	+ AChSrcLayer, SeSrcLayer likewise for ACh and SE (serotonin)

* GateLayer has GateStates in 1-to-1 correspondence with Pools, to keep
   track of gating state -- source gating layers can send updates to other layers.

################
# PBWM specific

* MatrixLayer for dorsal striatum gating of DLPFC areas, separate D1R = Go, D2R = NoGo
	Each layer contains Maint and Out GateTypes, as function of outer 4D Pool X dimension
	(Maint on the left, Out on the right)

* GPiThalLayer receives from Matrix Go and GPe NoGo to compute final WTA gating, and
   broadcasts GateState info to its SendTo layers.  See Timing params for timing.

* PFCLayer for active maintenance -- uses DeepLeabra framework, with update timing according to
	deep.Layer DeepBurst.BurstQtr.  Gating is computed in quarter *before* updating in BurstQtr.
	At *end* of BurstQtr, Super Burst -> Deep Ctxt to drive maintenance via Ctxt in Deep.
*/
package pbwm
