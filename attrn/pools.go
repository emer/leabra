// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attrn

import (
	"log"

	"github.com/emer/emergent/evec"
)

///////////////////////////////////////////////////////////////////////////////////
//   EPools

// EPoolParams are how to gather excitation across pools
type EPoolParams struct {
	LayNm string  `desc:"layer name"`
	Wt    float32 `desc:"general scaling factor for how much excitation from this pool"`
}

func (pp *EPoolParams) Defaults() {
	pp.Wt = 1
}

// EPools is a list of pools
type EPools []*EPoolParams

// Validate ensures that LayNames layers are valid.
// ctxt is string for error message to provide context.
func (pp *EPools) Validate(net Network, ctxt string) error {
	var lasterr error
	for _, p := range *pp {
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

// IPoolParams are how to gather inhibition across pools
type IPoolParams struct {
	LayNm  string     `desc:"layer name"`
	Wt     float32    `desc:"general scaling factor for how much overall inhibition from this pool contributes, in a non-pool-specific manner"`
	PoolWt float32    `desc:"scaling factor for how much corresponding pools contribute in a pool-spcific manner, using offsets and averaging across pools as needed to match geometry"`
	SOff   evec.Vec2i `desc:"offset into source, sending layer"`
	ROff   evec.Vec2i `desc:"offset into our own receiving layer"`
}

func (pp *IPoolParams) Defaults() {
	pp.Wt = 1
}

// IPools is a list of pools
type IPools []*IPoolParams

// Validate ensures that LayNames layers are valid.
// ctxt is string for error message to provide context.
func (pp *IPools) Validate(net Network, ctxt string) error {
	var lasterr error
	for _, p := range *pp {
		_, err := net.LayerByNameTry(p.LayNm)
		if err != nil {
			log.Printf("%s IPools.Validate: %v\n", ctxt, err)
			lasterr = err
		}
	}
	return lasterr
}
