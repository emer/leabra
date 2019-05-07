// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package leabra is the overall repository for all standard Leabra algorithm code
implemented in the Go language (golang) with Python wrappers.

This top-level of the repository has no functional code -- everything is organized
into the following sub-repositories:

* leabra: the core standard implementation with the minimal set of standard mechanisms
exclusively using rate-coded neurons -- there are too many differences with spiking,
so that is now separated out into a different package.

* deep: the DeepLeabra version which performs predictive
learning by attempting to predict the activation states over the Pulvinar nucleus
of the thalamus (in posterior sensory cortex), which are driven phasically every
100 msec by deep layer 5 intrinsic bursting (5IB) neurons that have strong focal
(essentially 1-to-1) connections onto the Pulvinar Thalamic Relay Cell (TRC)
neurons.

* examples: these actually compile into runnable programs and provide the starting
point for your own simulations.  examples/ra25 is the place to start for the most
basic standard template of a model that learns a small set of input / output
patterns in a classic supervised-learning manner.

* python: follow the instructions in the README.md file to build a python wrapper
that will allow you to fully control the models using Python.
*/
package leabra
