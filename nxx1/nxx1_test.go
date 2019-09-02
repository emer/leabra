// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nxx1

import (
	"testing"

	"github.com/chewxy/math32"
)

// difTol is the numerical difference tolerance for comparing vs. target values
const difTol = float32(1.0e-10)

func TestXX1(t *testing.T) {
	xx1 := Params{}
	xx1.Defaults()

	tstx := []float32{-0.05, -0.04, -0.03, -0.02, -0.01, 0, .01, .02, .03, .04, .05, .1, .2, .3, .4, .5}
	cory := []float32{1.7735989e-14, 7.155215e-12, 2.8866178e-09, 1.1645374e-06, 0.00046864923, 0.094767615, 0.47916666, 0.65277773, 0.742268, 0.7967479, 0.8333333, 0.90909094, 0.95238096, 0.96774197, 0.9756098, 0.98039216}
	ny := make([]float32, len(tstx))

	for i := range tstx {
		ny[i] = xx1.NoisyXX1(tstx[i])
		dif := math32.Abs(ny[i] - cory[i])
		if dif > difTol { // allow for small numerical diffs
			t.Errorf("XX1 err: dix: %v, x: %v, y: %v, cor y: %v, dif: %v\n", i, tstx[i], ny[i], cory[i], dif)
		}
	}
	// fmt.Printf("ny vals: %v\n", ny)
}
