// Code generated by "core generate"; DO NOT EDIT.

package pvlv

import (
	"cogentcore.org/core/enums"
)

var _AcqExtValues = []AcqExt{0, 1, 2}

// AcqExtN is the highest valid value for type AcqExt, plus one.
const AcqExtN AcqExt = 3

var _AcqExtValueMap = map[string]AcqExt{`Acq`: 0, `Ext`: 1, `NAcqExt`: 2}

var _AcqExtDescMap = map[AcqExt]string{0: ``, 1: ``, 2: ``}

var _AcqExtMap = map[AcqExt]string{0: `Acq`, 1: `Ext`, 2: `NAcqExt`}

// String returns the string representation of this AcqExt value.
func (i AcqExt) String() string { return enums.String(i, _AcqExtMap) }

// SetString sets the AcqExt value from its string representation,
// and returns an error if the string is invalid.
func (i *AcqExt) SetString(s string) error { return enums.SetString(i, s, _AcqExtValueMap, "AcqExt") }

// Int64 returns the AcqExt value as an int64.
func (i AcqExt) Int64() int64 { return int64(i) }

// SetInt64 sets the AcqExt value from an int64.
func (i *AcqExt) SetInt64(in int64) { *i = AcqExt(in) }

// Desc returns the description of the AcqExt value.
func (i AcqExt) Desc() string { return enums.Desc(i, _AcqExtDescMap) }

// AcqExtValues returns all possible values for the type AcqExt.
func AcqExtValues() []AcqExt { return _AcqExtValues }

// Values returns all possible values for the type AcqExt.
func (i AcqExt) Values() []enums.Enum { return enums.Values(_AcqExtValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i AcqExt) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *AcqExt) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "AcqExt") }

var _StimValues = []Stim{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

// StimN is the highest valid value for type Stim, plus one.
const StimN Stim = 13

var _StimValueMap = map[string]Stim{`StmA`: 0, `StmB`: 1, `StmC`: 2, `StmD`: 3, `StmE`: 4, `StmF`: 5, `StmU`: 6, `StmV`: 7, `StmW`: 8, `StmX`: 9, `StmY`: 10, `StmZ`: 11, `StmNone`: 12}

var _StimDescMap = map[Stim]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``, 5: ``, 6: ``, 7: ``, 8: ``, 9: ``, 10: ``, 11: ``, 12: ``}

var _StimMap = map[Stim]string{0: `StmA`, 1: `StmB`, 2: `StmC`, 3: `StmD`, 4: `StmE`, 5: `StmF`, 6: `StmU`, 7: `StmV`, 8: `StmW`, 9: `StmX`, 10: `StmY`, 11: `StmZ`, 12: `StmNone`}

// String returns the string representation of this Stim value.
func (i Stim) String() string { return enums.String(i, _StimMap) }

// SetString sets the Stim value from its string representation,
// and returns an error if the string is invalid.
func (i *Stim) SetString(s string) error { return enums.SetString(i, s, _StimValueMap, "Stim") }

// Int64 returns the Stim value as an int64.
func (i Stim) Int64() int64 { return int64(i) }

// SetInt64 sets the Stim value from an int64.
func (i *Stim) SetInt64(in int64) { *i = Stim(in) }

// Desc returns the description of the Stim value.
func (i Stim) Desc() string { return enums.Desc(i, _StimDescMap) }

// StimValues returns all possible values for the type Stim.
func StimValues() []Stim { return _StimValues }

// Values returns all possible values for the type Stim.
func (i Stim) Values() []enums.Enum { return enums.Values(_StimValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Stim) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Stim) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "Stim") }

var _ContextValues = []Context{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58}

// ContextN is the highest valid value for type Context, plus one.
const ContextN Context = 59

var _ContextValueMap = map[string]Context{`CtxA`: 0, `CtxA_B`: 1, `CtxA_C`: 2, `CtxB`: 3, `CtxB_B`: 4, `CtxB_C`: 5, `CtxC`: 6, `CtxC_B`: 7, `CtxC_C`: 8, `CtxD`: 9, `CtxD_B`: 10, `CtxD_C`: 11, `CtxE`: 12, `CtxE_B`: 13, `CtxE_C`: 14, `CtxF`: 15, `CtxF_B`: 16, `CtxF_C`: 17, `CtxU`: 18, `CtxU_B`: 19, `CtxU_C`: 20, `CtxV`: 21, `CtxV_B`: 22, `CtxV_C`: 23, `CtxW`: 24, `CtxW_B`: 25, `CtxW_C`: 26, `CtxX`: 27, `CtxX_B`: 28, `CtxX_C`: 29, `CtxY`: 30, `CtxY_B`: 31, `CtxY_C`: 32, `CtxZ`: 33, `CtxZ_B`: 34, `CtxZ_C`: 35, `CtxAX`: 36, `CtxAX_B`: 37, `CtxAX_C`: 38, `CtxAB`: 39, `CtxAB_B`: 40, `CtxAB_C`: 41, `CtxBY`: 42, `CtxBY_B`: 43, `CtxBY_C`: 44, `CtxCD`: 45, `CtxCD_B`: 46, `CtxCD_C`: 47, `CtxCX`: 48, `CtxCX_B`: 49, `CtxCX_C`: 50, `CtxCY`: 51, `CtxCY_B`: 52, `CtxCY_C`: 53, `CtxCZ`: 54, `CtxCZ_B`: 55, `CtxCZ_C`: 56, `CtxDU`: 57, `CtxNone`: 58}

var _ContextDescMap = map[Context]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``, 5: ``, 6: ``, 7: ``, 8: ``, 9: ``, 10: ``, 11: ``, 12: ``, 13: ``, 14: ``, 15: ``, 16: ``, 17: ``, 18: ``, 19: ``, 20: ``, 21: ``, 22: ``, 23: ``, 24: ``, 25: ``, 26: ``, 27: ``, 28: ``, 29: ``, 30: ``, 31: ``, 32: ``, 33: ``, 34: ``, 35: ``, 36: ``, 37: ``, 38: ``, 39: ``, 40: ``, 41: ``, 42: ``, 43: ``, 44: ``, 45: ``, 46: ``, 47: ``, 48: ``, 49: ``, 50: ``, 51: ``, 52: ``, 53: ``, 54: ``, 55: ``, 56: ``, 57: ``, 58: ``}

var _ContextMap = map[Context]string{0: `CtxA`, 1: `CtxA_B`, 2: `CtxA_C`, 3: `CtxB`, 4: `CtxB_B`, 5: `CtxB_C`, 6: `CtxC`, 7: `CtxC_B`, 8: `CtxC_C`, 9: `CtxD`, 10: `CtxD_B`, 11: `CtxD_C`, 12: `CtxE`, 13: `CtxE_B`, 14: `CtxE_C`, 15: `CtxF`, 16: `CtxF_B`, 17: `CtxF_C`, 18: `CtxU`, 19: `CtxU_B`, 20: `CtxU_C`, 21: `CtxV`, 22: `CtxV_B`, 23: `CtxV_C`, 24: `CtxW`, 25: `CtxW_B`, 26: `CtxW_C`, 27: `CtxX`, 28: `CtxX_B`, 29: `CtxX_C`, 30: `CtxY`, 31: `CtxY_B`, 32: `CtxY_C`, 33: `CtxZ`, 34: `CtxZ_B`, 35: `CtxZ_C`, 36: `CtxAX`, 37: `CtxAX_B`, 38: `CtxAX_C`, 39: `CtxAB`, 40: `CtxAB_B`, 41: `CtxAB_C`, 42: `CtxBY`, 43: `CtxBY_B`, 44: `CtxBY_C`, 45: `CtxCD`, 46: `CtxCD_B`, 47: `CtxCD_C`, 48: `CtxCX`, 49: `CtxCX_B`, 50: `CtxCX_C`, 51: `CtxCY`, 52: `CtxCY_B`, 53: `CtxCY_C`, 54: `CtxCZ`, 55: `CtxCZ_B`, 56: `CtxCZ_C`, 57: `CtxDU`, 58: `CtxNone`}

// String returns the string representation of this Context value.
func (i Context) String() string { return enums.String(i, _ContextMap) }

// SetString sets the Context value from its string representation,
// and returns an error if the string is invalid.
func (i *Context) SetString(s string) error {
	return enums.SetString(i, s, _ContextValueMap, "Context")
}

// Int64 returns the Context value as an int64.
func (i Context) Int64() int64 { return int64(i) }

// SetInt64 sets the Context value from an int64.
func (i *Context) SetInt64(in int64) { *i = Context(in) }

// Desc returns the description of the Context value.
func (i Context) Desc() string { return enums.Desc(i, _ContextDescMap) }

// ContextValues returns all possible values for the type Context.
func ContextValues() []Context { return _ContextValues }

// Values returns all possible values for the type Context.
func (i Context) Values() []enums.Enum { return enums.Values(_ContextValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Context) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Context) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "Context") }

var _ValenceValues = []Valence{0, 1, 2}

// ValenceN is the highest valid value for type Valence, plus one.
const ValenceN Valence = 3

var _ValenceValueMap = map[string]Valence{`ValNone`: 0, `POS`: 1, `NEG`: 2}

var _ValenceDescMap = map[Valence]string{0: ``, 1: ``, 2: ``}

var _ValenceMap = map[Valence]string{0: `ValNone`, 1: `POS`, 2: `NEG`}

// String returns the string representation of this Valence value.
func (i Valence) String() string { return enums.String(i, _ValenceMap) }

// SetString sets the Valence value from its string representation,
// and returns an error if the string is invalid.
func (i *Valence) SetString(s string) error {
	return enums.SetString(i, s, _ValenceValueMap, "Valence")
}

// Int64 returns the Valence value as an int64.
func (i Valence) Int64() int64 { return int64(i) }

// SetInt64 sets the Valence value from an int64.
func (i *Valence) SetInt64(in int64) { *i = Valence(in) }

// Desc returns the description of the Valence value.
func (i Valence) Desc() string { return enums.Desc(i, _ValenceDescMap) }

// ValenceValues returns all possible values for the type Valence.
func ValenceValues() []Valence { return _ValenceValues }

// Values returns all possible values for the type Valence.
func (i Valence) Values() []enums.Enum { return enums.Values(_ValenceValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Valence) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Valence) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "Valence") }

var _PosUSValues = []PosUS{0, 1, 2, 3, 4}

// PosUSN is the highest valid value for type PosUS, plus one.
const PosUSN PosUS = 5

var _PosUSValueMap = map[string]PosUS{`Water`: 0, `Food`: 1, `Mate`: 2, `OtherPos`: 3, `PosUSNone`: 4}

var _PosUSDescMap = map[PosUS]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``}

var _PosUSMap = map[PosUS]string{0: `Water`, 1: `Food`, 2: `Mate`, 3: `OtherPos`, 4: `PosUSNone`}

// String returns the string representation of this PosUS value.
func (i PosUS) String() string { return enums.String(i, _PosUSMap) }

// SetString sets the PosUS value from its string representation,
// and returns an error if the string is invalid.
func (i *PosUS) SetString(s string) error { return enums.SetString(i, s, _PosUSValueMap, "PosUS") }

// Int64 returns the PosUS value as an int64.
func (i PosUS) Int64() int64 { return int64(i) }

// SetInt64 sets the PosUS value from an int64.
func (i *PosUS) SetInt64(in int64) { *i = PosUS(in) }

// Desc returns the description of the PosUS value.
func (i PosUS) Desc() string { return enums.Desc(i, _PosUSDescMap) }

// PosUSValues returns all possible values for the type PosUS.
func PosUSValues() []PosUS { return _PosUSValues }

// Values returns all possible values for the type PosUS.
func (i PosUS) Values() []enums.Enum { return enums.Values(_PosUSValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i PosUS) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *PosUS) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "PosUS") }

var _NegUSValues = []NegUS{0, 1, 2, 3, 4}

// NegUSN is the highest valid value for type NegUS, plus one.
const NegUSN NegUS = 5

var _NegUSValueMap = map[string]NegUS{`Shock`: 0, `Nausea`: 1, `Sharp`: 2, `OtherNeg`: 3, `NegUSNone`: 4}

var _NegUSDescMap = map[NegUS]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``}

var _NegUSMap = map[NegUS]string{0: `Shock`, 1: `Nausea`, 2: `Sharp`, 3: `OtherNeg`, 4: `NegUSNone`}

// String returns the string representation of this NegUS value.
func (i NegUS) String() string { return enums.String(i, _NegUSMap) }

// SetString sets the NegUS value from its string representation,
// and returns an error if the string is invalid.
func (i *NegUS) SetString(s string) error { return enums.SetString(i, s, _NegUSValueMap, "NegUS") }

// Int64 returns the NegUS value as an int64.
func (i NegUS) Int64() int64 { return int64(i) }

// SetInt64 sets the NegUS value from an int64.
func (i *NegUS) SetInt64(in int64) { *i = NegUS(in) }

// Desc returns the description of the NegUS value.
func (i NegUS) Desc() string { return enums.Desc(i, _NegUSDescMap) }

// NegUSValues returns all possible values for the type NegUS.
func NegUSValues() []NegUS { return _NegUSValues }

// Values returns all possible values for the type NegUS.
func (i NegUS) Values() []enums.Enum { return enums.Values(_NegUSValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i NegUS) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *NegUS) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "NegUS") }

var _TickValues = []Tick{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

// TickN is the highest valid value for type Tick, plus one.
const TickN Tick = 11

var _TickValueMap = map[string]Tick{`T0`: 0, `T1`: 1, `T2`: 2, `T3`: 3, `T4`: 4, `T5`: 5, `T6`: 6, `T7`: 7, `T8`: 8, `T9`: 9, `TckNone`: 10}

var _TickDescMap = map[Tick]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``, 5: ``, 6: ``, 7: ``, 8: ``, 9: ``, 10: ``}

var _TickMap = map[Tick]string{0: `T0`, 1: `T1`, 2: `T2`, 3: `T3`, 4: `T4`, 5: `T5`, 6: `T6`, 7: `T7`, 8: `T8`, 9: `T9`, 10: `TckNone`}

// String returns the string representation of this Tick value.
func (i Tick) String() string { return enums.String(i, _TickMap) }

// SetString sets the Tick value from its string representation,
// and returns an error if the string is invalid.
func (i *Tick) SetString(s string) error { return enums.SetString(i, s, _TickValueMap, "Tick") }

// Int64 returns the Tick value as an int64.
func (i Tick) Int64() int64 { return int64(i) }

// SetInt64 sets the Tick value from an int64.
func (i *Tick) SetInt64(in int64) { *i = Tick(in) }

// Desc returns the description of the Tick value.
func (i Tick) Desc() string { return enums.Desc(i, _TickDescMap) }

// TickValues returns all possible values for the type Tick.
func TickValues() []Tick { return _TickValues }

// Values returns all possible values for the type Tick.
func (i Tick) Values() []enums.Enum { return enums.Values(_TickValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i Tick) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *Tick) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "Tick") }

var _DaRTypeValues = []DaRType{0, 1}

// DaRTypeN is the highest valid value for type DaRType, plus one.
const DaRTypeN DaRType = 2

var _DaRTypeValueMap = map[string]DaRType{`D1R`: 0, `D2R`: 1}

var _DaRTypeDescMap = map[DaRType]string{0: `D1R: primarily expresses Dopamine D1 Receptors -- dopamine is excitatory and bursts of dopamine lead to increases in synaptic weight, while dips lead to decreases -- direct pathway in dorsal striatum`, 1: `D2R: primarily expresses Dopamine D2 Receptors -- dopamine is inhibitory and bursts of dopamine lead to decreases in synaptic weight, while dips lead to increases -- indirect pathway in dorsal striatum`}

var _DaRTypeMap = map[DaRType]string{0: `D1R`, 1: `D2R`}

// String returns the string representation of this DaRType value.
func (i DaRType) String() string { return enums.String(i, _DaRTypeMap) }

// SetString sets the DaRType value from its string representation,
// and returns an error if the string is invalid.
func (i *DaRType) SetString(s string) error {
	return enums.SetString(i, s, _DaRTypeValueMap, "DaRType")
}

// Int64 returns the DaRType value as an int64.
func (i DaRType) Int64() int64 { return int64(i) }

// SetInt64 sets the DaRType value from an int64.
func (i *DaRType) SetInt64(in int64) { *i = DaRType(in) }

// Desc returns the description of the DaRType value.
func (i DaRType) Desc() string { return enums.Desc(i, _DaRTypeDescMap) }

// DaRTypeValues returns all possible values for the type DaRType.
func DaRTypeValues() []DaRType { return _DaRTypeValues }

// Values returns all possible values for the type DaRType.
func (i DaRType) Values() []enums.Enum { return enums.Values(_DaRTypeValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i DaRType) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *DaRType) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "DaRType") }

var _StriatalCompartmentValues = []StriatalCompartment{0, 1, 2}

// StriatalCompartmentN is the highest valid value for type StriatalCompartment, plus one.
const StriatalCompartmentN StriatalCompartment = 3

var _StriatalCompartmentValueMap = map[string]StriatalCompartment{`PATCH`: 0, `MATRIX`: 1, `NSComp`: 2}

var _StriatalCompartmentDescMap = map[StriatalCompartment]string{0: ``, 1: ``, 2: ``}

var _StriatalCompartmentMap = map[StriatalCompartment]string{0: `PATCH`, 1: `MATRIX`, 2: `NSComp`}

// String returns the string representation of this StriatalCompartment value.
func (i StriatalCompartment) String() string { return enums.String(i, _StriatalCompartmentMap) }

// SetString sets the StriatalCompartment value from its string representation,
// and returns an error if the string is invalid.
func (i *StriatalCompartment) SetString(s string) error {
	return enums.SetString(i, s, _StriatalCompartmentValueMap, "StriatalCompartment")
}

// Int64 returns the StriatalCompartment value as an int64.
func (i StriatalCompartment) Int64() int64 { return int64(i) }

// SetInt64 sets the StriatalCompartment value from an int64.
func (i *StriatalCompartment) SetInt64(in int64) { *i = StriatalCompartment(in) }

// Desc returns the description of the StriatalCompartment value.
func (i StriatalCompartment) Desc() string { return enums.Desc(i, _StriatalCompartmentDescMap) }

// StriatalCompartmentValues returns all possible values for the type StriatalCompartment.
func StriatalCompartmentValues() []StriatalCompartment { return _StriatalCompartmentValues }

// Values returns all possible values for the type StriatalCompartment.
func (i StriatalCompartment) Values() []enums.Enum { return enums.Values(_StriatalCompartmentValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i StriatalCompartment) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *StriatalCompartment) UnmarshalText(text []byte) error {
	return enums.UnmarshalText(i, text, "StriatalCompartment")
}

var _DALrnRuleValues = []DALrnRule{0, 1}

// DALrnRuleN is the highest valid value for type DALrnRule, plus one.
const DALrnRuleN DALrnRule = 2

var _DALrnRuleValueMap = map[string]DALrnRule{`DAHebbVS`: 0, `TraceNoThalVS`: 1}

var _DALrnRuleDescMap = map[DALrnRule]string{0: ``, 1: ``}

var _DALrnRuleMap = map[DALrnRule]string{0: `DAHebbVS`, 1: `TraceNoThalVS`}

// String returns the string representation of this DALrnRule value.
func (i DALrnRule) String() string { return enums.String(i, _DALrnRuleMap) }

// SetString sets the DALrnRule value from its string representation,
// and returns an error if the string is invalid.
func (i *DALrnRule) SetString(s string) error {
	return enums.SetString(i, s, _DALrnRuleValueMap, "DALrnRule")
}

// Int64 returns the DALrnRule value as an int64.
func (i DALrnRule) Int64() int64 { return int64(i) }

// SetInt64 sets the DALrnRule value from an int64.
func (i *DALrnRule) SetInt64(in int64) { *i = DALrnRule(in) }

// Desc returns the description of the DALrnRule value.
func (i DALrnRule) Desc() string { return enums.Desc(i, _DALrnRuleDescMap) }

// DALrnRuleValues returns all possible values for the type DALrnRule.
func DALrnRuleValues() []DALrnRule { return _DALrnRuleValues }

// Values returns all possible values for the type DALrnRule.
func (i DALrnRule) Values() []enums.Enum { return enums.Values(_DALrnRuleValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i DALrnRule) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *DALrnRule) UnmarshalText(text []byte) error {
	return enums.UnmarshalText(i, text, "DALrnRule")
}

var _ModNeuronVarValues = []ModNeuronVar{0, 1, 2, 3, 4, 5, 6, 7, 8}

// ModNeuronVarN is the highest valid value for type ModNeuronVar, plus one.
const ModNeuronVarN ModNeuronVar = 9

var _ModNeuronVarValueMap = map[string]ModNeuronVar{`DA`: 0, `ACh`: 1, `SE`: 2, `ModAct`: 3, `ModLevel`: 4, `ModNet`: 5, `ModLrn`: 6, `PVAct`: 7, `Cust1`: 8}

var _ModNeuronVarDescMap = map[ModNeuronVar]string{0: ``, 1: ``, 2: ``, 3: ``, 4: ``, 5: ``, 6: ``, 7: ``, 8: ``}

var _ModNeuronVarMap = map[ModNeuronVar]string{0: `DA`, 1: `ACh`, 2: `SE`, 3: `ModAct`, 4: `ModLevel`, 5: `ModNet`, 6: `ModLrn`, 7: `PVAct`, 8: `Cust1`}

// String returns the string representation of this ModNeuronVar value.
func (i ModNeuronVar) String() string { return enums.String(i, _ModNeuronVarMap) }

// SetString sets the ModNeuronVar value from its string representation,
// and returns an error if the string is invalid.
func (i *ModNeuronVar) SetString(s string) error {
	return enums.SetString(i, s, _ModNeuronVarValueMap, "ModNeuronVar")
}

// Int64 returns the ModNeuronVar value as an int64.
func (i ModNeuronVar) Int64() int64 { return int64(i) }

// SetInt64 sets the ModNeuronVar value from an int64.
func (i *ModNeuronVar) SetInt64(in int64) { *i = ModNeuronVar(in) }

// Desc returns the description of the ModNeuronVar value.
func (i ModNeuronVar) Desc() string { return enums.Desc(i, _ModNeuronVarDescMap) }

// ModNeuronVarValues returns all possible values for the type ModNeuronVar.
func ModNeuronVarValues() []ModNeuronVar { return _ModNeuronVarValues }

// Values returns all possible values for the type ModNeuronVar.
func (i ModNeuronVar) Values() []enums.Enum { return enums.Values(_ModNeuronVarValues) }

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i ModNeuronVar) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *ModNeuronVar) UnmarshalText(text []byte) error {
	return enums.UnmarshalText(i, text, "ModNeuronVar")
}
