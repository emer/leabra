// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"log"

	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/evec"
)

///////////////////////////////////////////////////////////////////////////////////
//   EPools

// EPool are how to gather excitation across pools
type EPool struct {

	// layer name
	LayNm string

	// general scaling factor for how much excitation from this pool
	Wt float32
}

func (ep *EPool) Defaults() {
	ep.Wt = 1
}

// EPools is a list of pools
type EPools []*EPool

// Add adds a new epool connection
func (ep *EPools) Add(laynm string, wt float32) *EPool {
	np := &EPool{LayNm: laynm, Wt: wt}
	*ep = append(*ep, np)
	return np
}

// Validate ensures that LayNames layers are valid.
// ctxt is string for error message to provide context.
func (ep *EPools) Validate(net emer.Network, ctxt string) error {
	var lasterr error
	for _, p := range *ep {
		_, err := net.LayerByNameTry(p.LayNm)
		if err != nil {
			log.Printf("%s EPools.Validate: %v\n", ctxt, err)
			lasterr = err
		}
	}
	return lasterr
}

///////////////////////////////////////////////////////////////////////////////////
//   IPools

// IPool are how to gather inhibition across pools
type IPool struct {

	// layer name
	LayNm string

	// general scaling factor for how much overall inhibition from this pool contributes, in a non-pool-specific manner
	Wt float32

	// scaling factor for how much corresponding pools contribute in a pool-spcific manner, using offsets and averaging across pools as needed to match geometry
	PoolWt float32

	// offset into source, sending layer
	SOff evec.Vec2i

	// offset into our own receiving layer
	ROff evec.Vec2i
}

func (ip *IPool) Defaults() {
	ip.Wt = 1
}

// IPools is a list of pools
type IPools []*IPool

// Add adds a new ipool connection
func (ip *IPools) Add(laynm string, wt float32) *IPool {
	np := &IPool{LayNm: laynm, Wt: wt}
	*ip = append(*ip, np)
	return np
}

// Validate ensures that LayNames layers are valid.
// ctxt is string for error message to provide context.
func (ip *IPools) Validate(net emer.Network, ctxt string) error {
	var lasterr error
	for _, p := range *ip {
		_, err := net.LayerByNameTry(p.LayNm)
		if err != nil {
			log.Printf("%s IPools.Validate: %v\n", ctxt, err)
			lasterr = err
		}
	}
	return lasterr
}
