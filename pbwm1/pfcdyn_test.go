// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"fmt"
	"testing"
)

func TestPFCDyn(t *testing.T) {
	t.Skip()
	pd := PFCDyns{}
	pd.FullDyn(10)
	mx := float32(20)
	for di, dy := range pd {
		fmt.Printf("\nDyn: %v\t %v\n", di, dy.Desc)
		for t := float32(0); t < mx; t += 1 {
			fmt.Printf("\t%g\t%g\n", t, dy.Value(t))
		}
	}
}
