package pvlv

import "github.com/emer/etable/etensor"

type Inputs interface {
	Empty() bool
	FromString(s string) Inputs
	OneHot() int
	Tensor() etensor.Tensor
	TensorScaled(scale float32) etensor.Tensor
}
