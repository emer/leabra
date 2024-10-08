// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

//go:generate core generate

//////////////////////////////////////////////////////////////////////////////////////
//  LayerType

// note: need to define a new type for these extensions for the GUI interface,
// but need to use the *old type* in the code, so we have this unfortunate
// redundancy here.

// LayerType has the DeepLeabra extensions to the LayerTypes types, for gui
type LayerType LayerTypes //enums:enum

const (
	// CT are layer 6 corticothalamic projecting neurons, which drive predictions
	// in TRC (Pulvinar) via standard pathways.
	CT LayerTypes = LayerTypesN + iota

	// TRC are thalamic relay cell neurons in the Pulvinar / MD thalamus,
	// which alternately reflect predictions driven by Deep layer pathways,
	// and actual outcomes driven by Burst activity from corresponding
	// Super layer neurons that provide strong driving inputs to TRC neurons.
	TRC
)

// gui versions
const (
	CT_ LayerType = LayerType(LayerTypesN) + iota
	TRC_
)

// LayerProps are required to get the extended EnumType
var LayerProps = tree.Props{
	// "EnumType:Typ": KiT_LayerType,
	// "ToolBar": tree.PropSlice{
	// 	{"Defaults", tree.Props{
	// 		"icon": "reset",
	// 		"desc": "return all parameters to their intial default values",
	// 	}},
	// 	{"InitWeights", tree.Props{
	// 		"icon": "update",
	// 		"desc": "initialize the layer's weight values according to path parameters, for all *sending* pathways out of this layer",
	// 	}},
	// 	{"InitActs", tree.Props{
	// 		"icon": "update",
	// 		"desc": "initialize the layer's activation values",
	// 	}},
	// 	{"sep-act", tree.BlankProp{}},
	// 	{"LesionNeurons", tree.Props{
	// 		"icon": "close",
	// 		"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
	// 		"Args": tree.PropSlice{
	// 			{"Proportion", tree.Props{
	// 				"desc": "proportion (0 -- 1) of neurons to lesion",
	// 			}},
	// 		},
	// 	}},
	// 	{"UnLesionNeurons", tree.Props{
	// 		"icon": "reset",
	// 		"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
	// 	}},
	// },
}
