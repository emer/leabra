// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"github.com/emer/etable/v2/etensor"
)

// var StimRe, _ = regexp.Compile("([ABCDEFUVWXYZ])([ABCDEFUVWXYZ]?)_(Rf|NR)")
var CtxRe, _ = regexp.Compile("([ABCDEFUVWXYZ])([ABCDEFUVWXYZ]?)_?([ABCDEFUVWXYZ]?)")

// Stim : conditioned stimuli
var StimInShape = []int{12, 1}

type Stim int //enums:enum

const (
	StmA    Stim = iota // A
	StmB                // B
	StmC                // C
	StmD                // D
	StmE                // E
	StmF                // F
	StmU                // U
	StmV                // V
	StmW                // W
	StmX                // X
	StmY                // Y
	StmZ                // Z
	StmNone             // NoStim
)

var StimMap = map[string]Stim{
	StmA.String():    StmA,
	StmB.String():    StmB,
	StmC.String():    StmC,
	StmD.String():    StmD,
	StmE.String():    StmE,
	StmF.String():    StmF,
	StmU.String():    StmU,
	StmV.String():    StmV,
	StmW.String():    StmW,
	StmX.String():    StmX,
	StmY.String():    StmY,
	StmZ.String():    StmZ,
	StmNone.String(): StmNone,
	"":               StmNone,
}

func (stm Stim) Empty() bool {
	return stm == StmNone
}
func (stm Stim) FromString(s string) Inputs {
	return StimMap[s]
}
func (stm Stim) OneHot() int {
	return 1 << int(stm)
}
func (stm Stim) Tensor() etensor.Tensor {
	tsr := etensor.NewInt([]int{12}, nil, []string{"Stim"})
	tsr.SetZeros()
	tsr.Set1D(int(stm), 1)
	return tsr
}
func (stm Stim) TensorScaled(scale float32) etensor.Tensor {
	tsr := etensor.NewFloat32([]int{12}, nil, []string{"Stim"})
	tsr.SetZeros()
	tsr.Set1D(int(stm), 1.0/scale)
	return tsr
}

// end Stim

// Context
var ContextInShape = []int{20, 3}

type Context int //enums:enum

const (
	CtxA      Context = iota // A
	CtxA_B                   // A_B
	CtxA_C                   // A_C
	CtxB                     // B
	CtxB_B                   // B_B
	CtxB_C                   // B_C
	CtxC                     // C
	CtxC_B                   // C_B
	CtxC_C                   // C_C
	CtxD                     // D
	CtxD_B                   // D_B
	CtxD_C                   // D_C
	CtxE                     // E
	CtxE_B                   // E_B
	CtxE_C                   // E_C
	CtxF                     // F
	CtxF_B                   // F_B
	CtxF_C                   // F_C
	CtxU                     // U
	CtxU_B                   // U_B
	CtxU_C                   // U_C
	CtxV                     // V
	CtxV_B                   // V_B
	CtxV_C                   // V_C
	CtxW                     // W
	CtxW_B                   // W_B
	CtxW_C                   // W_C
	CtxX                     // X
	CtxX_B                   // X_B
	CtxX_C                   // X_C
	CtxY                     // Y
	CtxY_B                   // Y_B
	CtxY_C                   // Y_C
	CtxZ                     // Z
	CtxZ_B                   // Z_B
	CtxZ_C                   // Z_C
	CtxAX                    // AX
	CtxAX_B                  // AX_B
	CtxAX_C                  // AX_C
	CtxAB                    // AB
	CtxAB_B                  // AB_B
	CtxAB_C                  // AB_C
	CtxBY                    // BY
	CtxBY_B                  // BY_B
	CtxBY_C                  // BY_C
	CtxCD                    // CD
	CtxCD_B                  // CD_B
	CtxCD_C                  // CD_C
	CtxCX                    // CX
	CtxCX_B                  // CX_B
	CtxCX_C                  // CX_C
	CtxCY                    // CY
	CtxCY_B                  // CY_B
	CtxCY_C                  // CY_C
	CtxCZ                    // CZ
	CtxCZ_B                  // CZ_B
	CtxCZ_C                  // CZ_C
	CtxDU                    // DU
	CtxNone                  // NoContext
	NContexts = CtxNone
)

var CtxMap = map[string]Context{
	CtxA.String():    CtxA,
	CtxA_B.String():  CtxA_B,
	CtxA_C.String():  CtxA_C,
	CtxB.String():    CtxB,
	CtxB_B.String():  CtxB_B,
	CtxB_C.String():  CtxB_C,
	CtxC.String():    CtxC,
	CtxC_B.String():  CtxC_B,
	CtxC_C.String():  CtxC_C,
	CtxD.String():    CtxD,
	CtxD_B.String():  CtxD_B,
	CtxD_C.String():  CtxD_C,
	CtxE.String():    CtxE,
	CtxE_B.String():  CtxE_B,
	CtxE_C.String():  CtxE_C,
	CtxF.String():    CtxF,
	CtxF_B.String():  CtxF_B,
	CtxF_C.String():  CtxF_C,
	CtxU.String():    CtxU,
	CtxU_B.String():  CtxU_B,
	CtxU_C.String():  CtxU_C,
	CtxV.String():    CtxV,
	CtxV_B.String():  CtxV_B,
	CtxV_C.String():  CtxV_C,
	CtxW.String():    CtxW,
	CtxW_B.String():  CtxW_B,
	CtxW_C.String():  CtxW_C,
	CtxX.String():    CtxX,
	CtxX_B.String():  CtxX_B,
	CtxX_C.String():  CtxX_C,
	CtxY.String():    CtxY,
	CtxY_B.String():  CtxY_B,
	CtxY_C.String():  CtxY_C,
	CtxZ.String():    CtxZ,
	CtxZ_B.String():  CtxZ_B,
	CtxZ_C.String():  CtxZ_C,
	CtxAX.String():   CtxAX,
	CtxAX_B.String(): CtxAX_B,
	CtxAX_C.String(): CtxAX_C,
	CtxAB.String():   CtxAB,
	CtxAB_B.String(): CtxAB_B,
	CtxAB_C.String(): CtxAB_C,
	CtxBY.String():   CtxBY,
	CtxBY_B.String(): CtxBY_B,
	CtxBY_C.String(): CtxBY_C,
	CtxCD.String():   CtxCD,
	CtxCD_B.String(): CtxCD_B,
	CtxCD_C.String(): CtxCD_C,
	CtxCX.String():   CtxCX,
	CtxCX_B.String(): CtxCX_B,
	CtxCX_C.String(): CtxCX_C,
	CtxCY.String():   CtxCY,
	CtxCY_B.String(): CtxCY_B,
	CtxCY_C.String(): CtxCY_C,
	CtxCZ.String():   CtxCZ,
	CtxCZ_B.String(): CtxCZ_B,
	CtxCZ_C.String(): CtxCZ_C,
	CtxDU.String():   CtxDU,
}

func (ctx Context) Empty() bool {
	return ctx == CtxNone
}
func (ctx Context) FromString(s string) Inputs {
	return CtxMap[s]
}
func (ctx Context) Int() int {
	return int(ctx)
}
func (ctx Context) OneHot() int {
	iCtx := ctx.Int()
	reshuffled := iCtx/3 + iCtx%3
	return 1 << reshuffled
}
func (ctx Context) Parts() []int {
	iCtx := ctx.Int()
	return []int{iCtx / 3, iCtx % 3}
}
func (ctx Context) Tensor() etensor.Tensor {
	tsr := etensor.NewInt([]int{20, 3}, nil, []string{"Ctx", "CS"})
	tsr.SetZeros()
	tsr.Set(ctx.Parts(), 1)
	return tsr
}
func (ctx Context) TensorScaled(scale float32) etensor.Tensor {
	tsr := etensor.NewFloat32([]int{20, 3}, nil, []string{"Ctx", "CS"})
	tsr.SetZeros()
	tsr.SetFloat(ctx.Parts(), 1.0/float64(scale))
	return tsr
}

// end Context

// Valence
type Valence int //enums:enum

const (
	ValNone Valence = iota // NoValence
	POS
	NEG
)

var ValMap = map[string]Valence{
	POS.String(): POS,
	NEG.String(): NEG,
}

func (val Valence) Empty() bool {
	return val == ValNone
}
func (val Valence) FromString(s string) Inputs {
	return ValMap[s]
}
func (val Valence) OneHot() int {
	return 1 << int(val)
}
func (val Valence) Tensor() etensor.Tensor {
	tsr := etensor.NewInt([]int{2}, nil, []string{"Valence"})
	tsr.Set1D(int(val), 1)
	return tsr
}
func (val Valence) TensorScaled(scale float32) etensor.Tensor {
	tsr := etensor.NewFloat32([]int{2}, nil, []string{"Valence"})
	tsr.Set1D(int(val), 1.0/scale)
	return tsr
}
func (val Valence) Negate() Valence {
	return Valence(1 - (int(val)))
}

var Fooey Inputs = POS // for testing
//end Valence

// US, either positive or negative Valence
type IUS interface {
	Val() Valence
	String() string
	Int() int
}

type US int

func (us US) Int() int {
	return int(us)
}

func (us US) Val() Valence {
	var ius interface{} = us
	switch ius.(IUS).(type) {
	case PosUS:
		return POS
	default:
		return NEG
	}
}

var USInShape = []int{4}

func OneHotUS(us US) int {
	return 1 << us.Int()
}

func Tensor(us US) etensor.Tensor {
	tsr := etensor.NewInt([]int{4}, nil, []string{"US"})
	if !us.Empty() {
		tsr.Set1D(us.Int(), 1)
	}
	return tsr
}

func TensorScaled(us US, scale float32) etensor.Tensor {
	tsr := etensor.NewFloat32([]int{4}, nil, []string{"US"})
	if !us.Empty() {
		tsr.Set1D(us.Int(), 1.0/scale)
	}
	return tsr
}

// end US

// positive and negative subtypes of US
type PosUS US //enums:enum -no-extend
type NegUS US //enums:enum -no-extend

const (
	Water PosUS = iota
	Food
	Mate
	OtherPos
	PosUSNone // NoPosUS
	NPosUS    = PosUSNone
)
const (
	Shock NegUS = iota
	Nausea
	Sharp
	OtherNeg
	NegUSNone // NoNegUS
	NNegUS    = NegUSNone
)

var USNone = US(PosUSNone)
var _ IUS = Water // check for interface implementation
var _ IUS = Shock

var PosSMap = map[string]PosUS{
	Water.String():    Water,
	Food.String():     Food,
	Mate.String():     Mate,
	OtherPos.String(): OtherPos,
}
var NegSMap = map[string]NegUS{
	Shock.String():    Shock,
	Nausea.String():   Nausea,
	Sharp.String():    Sharp,
	OtherNeg.String(): OtherNeg,
}

func (pos PosUS) PosUSEmpty() bool {
	return pos == PosUSNone
}
func (neg NegUS) NegUSEmpty() bool {
	return neg == NegUSNone
}

func (us US) Empty() bool {
	if us.Val() == POS {
		return NegUS(us).NegUSEmpty()
	} else {
		return PosUS(us).PosUSEmpty()
	}
}

func (us US) FromString(s string) Inputs {
	if us.Val() == POS {
		return US(PosSMap[s])
	} else {
		return US(NegSMap[s])
	}
}

func (pos PosUS) FromString(s string) PosUS {
	return PosSMap[s]
}
func (neg NegUS) FromString(s string) NegUS {
	return NegSMap[s]
}

func (pos PosUS) OneHot() int {
	return US(pos).OneHot()
}
func (neg NegUS) OneHot() int {
	return US(neg).OneHot()
}
func (us US) OneHot() int {
	return OneHotUS(us)
}

func (pos PosUS) Val() Valence {
	return POS
}
func (neg NegUS) Val() Valence {
	return NEG
}

func (pos PosUS) Int() int {
	return int(pos)
}
func (neg NegUS) Int() int {
	return int(neg)
}

func (pos PosUS) Tensor() etensor.Tensor {
	return US(pos).Tensor()
}
func (neg NegUS) Tensor() etensor.Tensor {
	return US(neg).Tensor()
}
func (us US) Tensor() etensor.Tensor {
	return Tensor(us)
}

//	func (pos PosUS) TensorScaled(scale float32) etensor.Tensor {
//		return TensorScaled(pos, 1.0 / scale)
//	}
//
//	func (neg NegUS) TensorScaled(scale float32) etensor.Tensor {
//		return TensorScaled(neg, 1.0 / scale)
//	}
func (us US) TensorScaled(scale float32) etensor.Tensor {
	return TensorScaled(us, 1.0/scale)
}

func (us US) String() string {
	if us.Val() == POS {
		return PosUS(us).String()
	} else {
		return NegUS(us).String()
	}
}

// end US subtypes

// Tick
type Tick int //enums:enum

const (
	T0 Tick = iota
	T1
	T2
	T3
	T4
	T5
	T6
	T7
	T8
	T9
	TckNone
)

var TickMap = map[string]Tick{
	T0.String():      T0,
	T1.String():      T1,
	T2.String():      T2,
	T3.String():      T3,
	T4.String():      T4,
	T5.String():      T5,
	T6.String():      T6,
	T7.String():      T7,
	T8.String():      T8,
	T9.String():      T9,
	TckNone.String(): TckNone,
}

func (t Tick) Empty() bool {
	return t == TckNone
}
func (t Tick) FromString(s string) Inputs {
	i, _ := strconv.Atoi(strings.TrimPrefix(strings.ToLower(s), "t"))
	return Tick(i)
}
func (t Tick) Int() int {
	return int(t)
}
func (t Tick) OneHot() int {
	return 1 << int(t)
}
func (t Tick) Tensor() etensor.Tensor {
	tsr := etensor.NewInt([]int{5}, nil, []string{"US"})
	if t.Empty() {
		tsr.Set1D(t.Int(), 1)
	}
	return tsr
}
func (t Tick) TensorScaled(scale float32) etensor.Tensor {
	tsr := etensor.NewFloat32([]int{5}, nil, []string{"US"})
	if t.Empty() {
		tsr.Set1D(t.Int(), 1.0/scale)
	}
	return tsr
}

// Tick

// USTimeIn
var USTimeInShape = []int{16, 2, 4, 5}

type USTimeState struct {

	// CS value
	Stm Stim

	// a US value or absent (USNone)
	US US

	// PV d, POS, NEG, or absent (ValNone)
	Val Valence

	// Within-trial timestep
	Tck Tick
}
type PackedUSTimeState int64

const USTimeNone PackedUSTimeState = 0

func (pus PackedUSTimeState) FromString(s string) PackedUSTimeState {
	return USTFromString(s).Pack()
}

func PUSTFromString(s string) PackedUSTimeState {
	return USTimeNone.FromString(s)
}

func (usts USTimeState) Pack() PackedUSTimeState {
	var usShift, valShift, tckShift int
	ret := USTimeNone
	if usts.Stm != StmNone {
		ret = 1 << usts.Stm
	}
	if !usts.US.Empty() {
		usShift = int(StimN) + usts.US.Int()
		ret += 1 << usShift
	}
	if usts.Val != ValNone {
		valShift = int(StimN) + int(NPosUS) + int(usts.Val)
		ret += 1 << valShift
	}
	if usts.Tck != TckNone {
		tckShift = int(StimN) + int(NPosUS) + int(ValenceN) + int(usts.Tck)
		ret += 1 << tckShift
	}
	return ret
}

func (ps PackedUSTimeState) Unpack() USTimeState {
	usts := USTimeState{Stm: StmNone, US: USNone, Val: ValNone, Tck: TckNone}

	var stimMask int = (1 << StimN) - 1
	stmTmp := int(ps) & stimMask
	if stmTmp != 0 {
		usts.Stm = Stim(math.Log2(float64(stmTmp)))
	}

	usShift := int(StimN)
	var usMask int = ((1 << NPosUS) - 1) << usShift
	usTmp := (int(ps) & usMask) >> usShift
	if usTmp != 0 {
		usts.US = US(int(math.Log2(float64(usTmp))))
	}

	valShift := usShift + int(NPosUS)
	var valMask int = ((1 << ValenceN) - 1) << valShift
	valTmp := (int(ps) & valMask) >> valShift
	if valTmp != 0 {
		usts.Val = Valence(math.Log2(float64(valTmp)))
	}

	tckShift := valShift + int(ValenceN)
	var tckMask int = ((1 << TickN) - 1) << tckShift
	tckTmp := (int(ps) & tckMask) >> tckShift
	if tckTmp != 0 {
		usts.Tck = Tick(math.Log2(float64(tckTmp)))
	}

	return usts
}

func (ps PackedUSTimeState) Empty() bool {
	return ps == USTimeNone
}

func (usts USTimeState) Empty() bool {
	return usts.Pack().Empty()
}

func (ps PackedUSTimeState) String() string {
	if ps.Empty() {
		return "USTimeNone"
	} else {
		return ps.Unpack().String()
	}
}

func (ps PackedUSTimeState) Tensor() etensor.Tensor {
	return ps.Unpack().Tensor()
}
func (ps PackedUSTimeState) TensorScaled(scale float32) etensor.Tensor {
	return ps.Unpack().TensorScaled(scale)
}

func (usts USTimeState) String() string {
	for _, part := range []Inputs{usts.Val, usts.US, usts.Tck} {
		if part.Empty() {
			return USTimeNone.String()
		}
	}
	ss := ""
	if usts.Empty() {
		return "USTimeNone"
	} else {
		if usts.Stm.Empty() {
			ss = ""
		} else {
			ss = usts.Stm.String() + "_"
		}
		vs := usts.Val.String()
		if !usts.Val.Empty() {
			vs = strings.Title(strings.ToLower(vs))
		}
		return ss + vs + "US" +
			strconv.Itoa(usts.US.Int()) + "_" + strings.ToLower(usts.Tck.String())
	}
}

var StmGrpMap = map[Stim]int{
	StmNone: 0,
	StmA:    1,
	StmB:    2,
	StmC:    3,
	StmD:    1,
	StmE:    2,
	StmF:    3,
	StmX:    4,
	StmU:    4,
	StmY:    5,
	StmV:    5,
	StmZ:    6,
	StmW:    7,
}

const NoUSTimeIn = 320

func (usts USTimeState) EnumVal() int {
	mults := []int{5, 2, 4, 8}
	stim, ok := StmGrpMap[usts.Stm]
	if !ok {
		fmt.Printf("Unrecognized Stim: %v\n", usts.Stm)
	}
	parts := []int{int(usts.Tck), int(usts.Val) - 1, usts.US.Int(), stim}
	if parts[1] < 0 {
		//fmt.Println("invalid Valence in USTS EnumVal")
		return NoUSTimeIn
	}
	ret := 0
	mult := 1
	for i := 0; i < len(parts); i++ {
		ret += parts[i] * mult
		mult *= mults[i]
	}
	return ret
}

func (usts USTimeState) Coords() []int {
	valCoord := map[Valence]int{
		POS:     0,
		NEG:     1,
		ValNone: 0,
	}
	return []int{StmGrpMap[usts.Stm], valCoord[usts.Val], usts.US.Int(), int(usts.Tck)}
}

func (usts USTimeState) CoordsString() string {
	str := "{"
	for i, n := range usts.Coords() {
		str = str + strconv.Itoa(n)
		if i < 3 {
			str += ","
		}
	}
	str += "}"
	return str
}

func (usts USTimeState) Tensor() etensor.Tensor {
	tsr := etensor.NewFloat32(USTimeInShape, nil, nil)
	tsr.SetFloat(usts.TsrOffset(), 1.0)
	return tsr
}
func (usts USTimeState) TensorScaled(scale float32) etensor.Tensor {
	tsr := etensor.NewFloat32(USTimeInShape, nil, nil)
	tsr.SetFloat(usts.TsrOffset(), 1.0/float64(scale))
	return tsr
}

func (usts USTimeState) TensorScaleAndAdd(scale float32, other USTimeState) etensor.Tensor {
	res := usts.TensorScaled(scale)
	res.SetFloat(other.TsrOffset(), 1.0/float64(scale))
	return res
}

func (usts USTimeState) TsrOffset() []int {
	return []int{int(usts.Stm), int(usts.Val), int(usts.Tck), usts.US.Int()}
}

func (ps PackedUSTimeState) Shape() []int {
	return ps.Unpack().TsrOffset()
}

var USTRe, _ = regexp.Compile("([ABCDEFUVWXYZ]?)_?(Pos|Neg)US([0123])_t([01234])")

func USTFromString(uss string) USTimeState {
	matches := USTRe.FindStringSubmatch(uss)
	if matches == nil || uss == "USTimeNone" {
		return USTimeNone.Unpack()
	}
	stim, ok := StimMap[matches[1]]
	if !ok {
		fmt.Printf("Unrecognized StimIn: %v\n", matches[1])
		stim = StmNone
	}
	val, ok := ValMap[strings.ToUpper(matches[2])]
	if !ok {
		fmt.Printf("Unrecognized Valence: %v\n", matches[2])
		val = ValNone
	}
	ius, _ := strconv.Atoi(matches[3])
	var us US
	us = US(ius)
	//if val == POS {
	//	us = US(PosUS(ius))
	//} else {
	//	us = US(NegUS(ius))
	//}
	itick, _ := strconv.Atoi(matches[4])
	tick := Tick(itick)
	ret := USTimeState{stim, us, val, tick}
	return ret
}

//type USTimeStateList []USTimeState
//func (ustsl USTimeStateList) testStimPackUnpack() {
//	for _, usts := range ustsl {
//		decoded := usts.Pack().Unpack()
//		if decoded == usts {
//			fmt.Println("PASSED for", usts)
//		} else {
//			fmt.Println("FAILED for", usts, "decoded is", decoded)
//		}
//	}
//}

func (ps PackedUSTimeState) Stim() Stim {
	return ps.Unpack().Stm
}

func (ps PackedUSTimeState) US() US {
	return ps.Unpack().US
}

func (ps PackedUSTimeState) Valence() Valence {
	return ps.Unpack().Val
}

func (ps PackedUSTimeState) USTimeIn() Tick {
	return ps.Unpack().Tck
}

func (usts USTimeState) OneHot(scale float32) etensor.Tensor {
	tsr := etensor.NewFloat32([]int{16, 2, 4, 5}, nil, []string{"CS", "Valence", "US", "Time"})
	tsr.Set([]int{int(usts.Stm), int(usts.Tck), int(usts.Val) - 1, usts.US.Int()}, 1.0/scale)
	return tsr
}
