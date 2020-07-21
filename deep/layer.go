// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

//////////////////////////////////////////////////////////////////////////////////////
//  LayerType

// note: need to define a new type for these extensions for the GUI interface,
// but need to use the *old type* in the code, so we have this unfortunate
// redundancy here.

// LayerType has the DeepLeabra extensions to the emer.LayerType types, for gui
type LayerType emer.LayerType

//go:generate stringer -type=LayerType

var KiT_LayerType = kit.Enums.AddEnumExt(emer.KiT_LayerType, LayerTypeN, kit.NotBitFlag, nil)

const (
	// CT are layer 6 corticothalamic projecting neurons, which drive predictions
	// in TRC (Pulvinar) via standard projections.
	CT emer.LayerType = emer.LayerTypeN + iota

	// TRC are thalamic relay cell neurons in the Pulvinar / MD thalamus,
	// which alternately reflect predictions driven by Deep layer projections,
	// and actual outcomes driven by Burst activity from corresponding
	// Super layer neurons that provide strong driving inputs to TRC neurons.
	TRC
)

// gui versions
const (
	CT_ LayerType = LayerType(emer.LayerTypeN) + iota
	TRC_
	LayerTypeN
)

// LayerProps are required to get the extended EnumType
var LayerProps = ki.Props{
	"EnumType:Typ": KiT_LayerType,
	"ToolBar": ki.PropSlice{
		{"Defaults", ki.Props{
			"icon": "reset",
			"desc": "return all parameters to their intial default values",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's weight values according to prjn parameters, for all *sending* projections out of this layer",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"LesionNeurons", ki.Props{
			"icon": "close",
			"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
			"Args": ki.PropSlice{
				{"Proportion", ki.Props{
					"desc": "proportion (0 -- 1) of neurons to lesion",
				}},
			},
		}},
		{"UnLesionNeurons", ki.Props{
			"icon": "reset",
			"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
		}},
	},
}
