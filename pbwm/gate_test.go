// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pbwm

import (
	"testing"
)

func TestShape(t *testing.T) {
	gs := GateShape{}
	gs.Set(2, 3, 2)
	mtot := gs.Y * gs.MaintX
	ans := []int{0, 1, 2, 5, 6, 7}
	for i := 0; i < mtot; i++ {
		fi := gs.FullIndex1D(i, Maint)
		if fi != ans[i] {
			t.Errorf("Maint idx: %v: %v != %v\n", i, fi, ans[i])
		}
		// fmt.Printf("%v \t %v\n", i, fi)
	}
	otot := gs.Y * gs.OutX
	ans = []int{3, 4, 8, 9}
	for i := 0; i < otot; i++ {
		fi := gs.FullIndex1D(i, Out)
		if fi != ans[i] {
			t.Errorf("Out idx: %v: %v != %v\n", i, fi, ans[i])
		}
		// fmt.Printf("%v \t %v\n", i, fi)
	}
}
