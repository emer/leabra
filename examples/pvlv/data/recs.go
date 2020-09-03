package data

import (
	"errors"
	"github.com/goki/ki/kit"
)

type DataLoopOrder int

const (
	SEQUENTIAL DataLoopOrder = iota
	PERMUTED
	RANDOM
	DataLoopOrderN
)

var KiT_DataLoopOrder = kit.Enums.AddEnum(DataLoopOrderN, kit.NotBitFlag, nil)

func IntSequence(begin, end, step int) (sequence []int) {
	if step == 0 {
		panic(errors.New("step must not be zero"))
	}
	count := 0
	if (end > begin && step > 0) || (end < begin && step < 0) {
		count = (end-step-begin)/step + 1
	}

	sequence = make([]int, count)
	for i := 0; i < count; i, begin = i+1, begin+step {
		sequence[i] = begin
	}
	return
}
