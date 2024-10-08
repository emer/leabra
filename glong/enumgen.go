// Code generated by "core generate"; DO NOT EDIT.

package glong

import (
	"cogentcore.org/core/enums"
)

var _PathTypeValues = []PathType{4}

// PathTypeN is the highest valid value for type PathType, plus one.
const PathTypeN PathType = 5

var _PathTypeValueMap = map[string]PathType{`NMDA_`: 4}

var _PathTypeDescMap = map[PathType]string{4: ``}

var _PathTypeMap = map[PathType]string{4: `NMDA_`}

// String returns the string representation of this PathType value.
func (i PathType) String() string {
	return enums.StringExtended[PathType, PathTypes](i, _PathTypeMap)
}

// SetString sets the PathType value from its string representation,
// and returns an error if the string is invalid.
func (i *PathType) SetString(s string) error {
	return enums.SetStringExtended(i, (*PathTypes)(i), s, _PathTypeValueMap)
}

// Int64 returns the PathType value as an int64.
func (i PathType) Int64() int64 { return int64(i) }

// SetInt64 sets the PathType value from an int64.
func (i *PathType) SetInt64(in int64) { *i = PathType(in) }

// Desc returns the description of the PathType value.
func (i PathType) Desc() string {
	return enums.DescExtended[PathType, PathTypes](i, _PathTypeDescMap)
}

// PathTypeValues returns all possible values for the type PathType.
func PathTypeValues() []PathType {
	return enums.ValuesGlobalExtended(_PathTypeValues, PathTypesValues())
}

// Values returns all possible values for the type PathType.
func (i PathType) Values() []enums.Enum {
	return enums.ValuesExtended(_PathTypeValues, PathTypesValues())
}

// MarshalText implements the [encoding.TextMarshaler] interface.
func (i PathType) MarshalText() ([]byte, error) { return []byte(i.String()), nil }

// UnmarshalText implements the [encoding.TextUnmarshaler] interface.
func (i *PathType) UnmarshalText(text []byte) error { return enums.UnmarshalText(i, text, "PathType") }
