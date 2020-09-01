// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package recordlist

import "github.com/goki/ki/kit"

// An interface for lists of records of unspecified type.
// The key method is Permute, which allows access to records in
// permuted order.
// This really needs generics.
type RecordList interface {
	New(interface{}) interface{}
	Init() interface{}
	SetIndex([]int)
	Permute()
	GetIndex() []int
	WriteNext(interface{})
	ReadNext() interface{}
	Peek() interface{}
	AtEnd() bool
	Reset()
	SetPos(int)
	Length() int
	Cur() int
}

type DataLoopOrder int

const (
	SEQUENTIAL DataLoopOrder = iota
	PERMUTED
	RANDOM
	DataLoopOrderN
)

var KiT_DataLoopOrder = kit.Enums.AddEnum(DataLoopOrderN, kit.NotBitFlag, nil)
