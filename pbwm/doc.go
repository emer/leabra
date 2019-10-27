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

Naming rule: DA when a singleton, DaMod (lowercase a) when CamelCased with something else

# Basic Level

* ModLayer has DA, ACh, SE -- can be modulated

* DaSrcLayer sends DA to a list of layers (does not use Prjns)

* AChSrcLayer, SeSrcLayer likewise for ACh and SE (serotonin)

* msn.go has various elements for Medium Spiny Neuron principal cell of striatum -- core of BG

# PBWM specific

* MatrixLayer for dorsal striatum gating of DLPFC areas, separate D1R = Go, D2R = NoGo
	Each layer contains Maint and Out GateTypes, as function of outer 4D Pool X dimension
	(Maint on the left, Out on the right)

* PatchLayer for matrisomes within dorsal striatum modulating dopamine

* PFCLayer

*/
package pbwm
