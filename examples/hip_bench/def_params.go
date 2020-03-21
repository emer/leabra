// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "keeping default params for generic prjns",
				Params: params.Params{
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.WtBal.On":    "false",
				}},
			{Sel: ".EcCa1Prjn", Desc: "encoder projections -- no norm, moment",
				Params: params.Params{
					"Prjn.Learn.Lrate":        "0.04",
					"Prjn.Learn.Momentum.On":  "false",
					"Prjn.Learn.Norm.On":      "false",
					"Prjn.Learn.WtBal.On":     "true",  // better
					"Prjn.Learn.XCal.SetLLrn": "false", // bcm = better!  now avail
				}},
			{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
				Params: params.Params{
					"Prjn.CHL.Hebb":          "0.05", // .01 > .05? > .1?
					"Prjn.Learn.Lrate":       "0.2",  // .2 probably better? .4 was prev default
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Prjn prjns",
				Params: params.Params{
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
					"Prjn.Learn.Lrate":       "0.15", // err driven: .15 > .2 > .25 > .1
					// moss=4, delta=4, lr=0.2, test = 3 are best
				}},
			{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
				Params: params.Params{
					"Prjn.WtScale.Abs": "2.0", // 4 > 6 > 2 (fails)
					"Prjn.WtScale.Rel": "2.0", // Rel=4, Abs=1 fails entirely.  could try diff combos
				}},
			{Sel: "#InputToECin", Desc: "one-to-one input to EC",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.8",
					"Prjn.WtInit.Var":  "0.0",
				}},
			{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
					"Prjn.WtInit.Mean": "0.9",
					"Prjn.WtInit.Var":  "0.01",
					"Prjn.WtScale.Rel": "0.5", // .5 = .3? > .8 (fails)
				}},
			{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
				Params: params.Params{
					"Prjn.Learn.Learn": "false", // learning here definitely does NOT work!
					"Prjn.WtInit.Mean": "0.9",
					"Prjn.WtInit.Var":  "0.01",
					"Prjn.WtScale.Rel": "4", // err del 4: 4 > 6 > 8
				}},
			{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons: rel=1 slightly better than 2",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.1", // .1 = .2 > .5 > 1 !
					"Prjn.Learn.Lrate": "0.1", // .1 > .08 (close) > .15 > .2 > .04
				}},
			{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn":       "true", // absolutely essential to have on!
					"Prjn.CHL.Hebb":          ".5",   // .5 > 1 overall
					"Prjn.CHL.SAvgCor":       "0.1",  // .1 > .2 > .3 > .4 ?
					"Prjn.CHL.MinusQ1":       "true", // dg self err?
					"Prjn.Learn.Lrate":       "0.4",  // .4 > .3 > .2
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
				Params: params.Params{
					"Prjn.CHL.Hebb":          "0.01", // .01 > .005 > .02 > .002 > .001 > .05 (crazy)
					"Prjn.CHL.SAvgCor":       "0.4",
					"Prjn.Learn.Lrate":       "0.1", // CHL: .1 =~ .08 > .15 > .2, .05 (sig worse)
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.WtBal.On":    "true",
				}},
			{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level",
				Params: params.Params{
					"Layer.Act.Gbar.L":        ".1",
					"Layer.Inhib.ActAvg.Init": "0.2",
					"Layer.Inhib.Layer.On":    "false",
					"Layer.Inhib.Pool.Gi":     "2.0",
					"Layer.Inhib.Pool.On":     "true",
				}},
			{Sel: "#DG", Desc: "very sparse = high inibhition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.01",
					"Layer.Inhib.Layer.Gi":    "3.8", // 3.8 > 3.6 > 4.0 (too far -- tanks)
				}},
			{Sel: "#CA3", Desc: "sparse = high inibhition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Layer.Gi":    "2.8", // 2.8 = 3.0 really -- some better, some worse
					"Layer.Learn.AvgL.Gain":   "2.5", // stick with 2.5
				}},
			{Sel: "#CA1", Desc: "CA1 only Pools",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.1",
					"Layer.Inhib.Layer.On":    "false",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "2.4", // 2.4 > 2.2 > 2.6 > 2.8 -- 2.4 better *for small net* but not for larger!
					"Layer.Learn.AvgL.Gain":   "2.5", // 2.5 > 2 > 3
				}},
		},
		// NOTE: it is essential not to put Pat / Hip params here, as we have to use Base
		// to initialize the network every time, even if it is a different size..
	}},
	{Name: "List010", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "10",
				}},
		},
	}},
	{Name: "List020", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "20",
				}},
		},
	}},
	{Name: "List030", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "30",
				}},
		},
	}},
	{Name: "List040", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "40",
				}},
		},
	}},
	{Name: "List050", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "50",
				}},
		},
	}},
	{Name: "List060", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "60",
				}},
		},
	}},
	{Name: "List070", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "70",
				}},
		},
	}},
	{Name: "List080", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "80",
				}},
		},
	}},
	{Name: "List090", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "90",
				}},
		},
	}},
	{Name: "List100", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "100",
				}},
		},
	}},
	{Name: "List120", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "120",
				}},
		},
	}},
	{Name: "List160", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "160",
				}},
		},
	}},
	{Name: "List200", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "200",
				}},
		},
	}},
	{Name: "SmallHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "10",
					"HipParams.CA1Pool.X": "10",
					"HipParams.CA3Size.Y": "20",
					"HipParams.CA3Size.X": "20",
					"HipParams.DGRatio":   "1.5",
				}},
		},
	}},
	{Name: "MedHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "15",
					"HipParams.CA1Pool.X": "15",
					"HipParams.CA3Size.Y": "30",
					"HipParams.CA3Size.X": "30",
					"HipParams.DGRatio":   "1.5",
				}},
		},
	}},
	{Name: "BigHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "20",
					"HipParams.CA1Pool.X": "20",
					"HipParams.CA3Size.Y": "40",
					"HipParams.CA3Size.X": "40",
					"HipParams.DGRatio":   "1.5",
				}},
		},
	}},
}
