// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build not

package main

import "github.com/emer/emergent/v2/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Path", Desc: "keeping default params for generic paths",
				Params: params.Params{
					"Path.Learn.Momentum.On": "true",
					"Path.Learn.Norm.On":     "true",
					"Path.Learn.WtBal.On":    "false",
				}},
			{Sel: ".EcCa1Path", Desc: "encoder pathways -- no norm, moment",
				Params: params.Params{
					"Path.Learn.Lrate":       "0.04",
					"Path.Learn.Momentum.On": "false",
					"Path.Learn.Norm.On":     "false",
					"Path.Learn.WtBal.On":    "true", // counteracting hogging
					//"Path.Learn.XCal.SetLLrn": "true", // bcm now avail, comment out = default LLrn
					//"Path.Learn.XCal.LLrn":    "0",    // 0 = turn off BCM, must with SetLLrn = true
				}},
			{Sel: ".HippoCHL", Desc: "hippo CHL pathways -- no norm, moment, but YES wtbal = sig better",
				Params: params.Params{
					"Path.CHL.Hebb":          "0.01", // .01 > .05? > .1?
					"Path.Learn.Lrate":       "0.2",  // .2 probably better? .4 was prev default
					"Path.Learn.Momentum.On": "false",
					"Path.Learn.Norm.On":     "false",
					"Path.Learn.WtBal.On":    "true",
				}},
			{Sel: ".PPath", Desc: "performant path, new Dg error-driven EcCa1Path paths",
				Params: params.Params{
					"Path.Learn.Lrate":       "0.15", // err driven: .15 > .2 > .25 > .1
					"Path.Learn.Momentum.On": "false",
					"Path.Learn.Norm.On":     "false",
					"Path.Learn.WtBal.On":    "true",
					//"Path.Learn.XCal.SetLLrn": "true", // bcm now avail, comment out = default LLrn
					//"Path.Learn.XCal.LLrn":    "0",    // 0 = turn off BCM, must with SetLLrn = true
				}},
			{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
				Params: params.Params{
					"Path.WtScale.Abs": "4.0", // 4 > 6 > 2 (fails)
				}},
			{Sel: "#InputToECin", Desc: "one-to-one input to EC",
				Params: params.Params{
					"Path.Learn.Learn": "false",
					"Path.WtInit.Mean": "0.8",
					"Path.WtInit.Var":  "0.0",
				}},
			{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
				Params: params.Params{
					"Path.Learn.Learn": "false",
					"Path.WtInit.Mean": "0.9",
					"Path.WtInit.Var":  "0.01",
					"Path.WtScale.Rel": "0.5", // .5 = .3? > .8 (fails); zycyc test this
				}},
			{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
				Params: params.Params{
					"Path.Learn.Learn": "false", // learning here definitely does NOT work!
					"Path.WtInit.Mean": "0.9",
					"Path.WtInit.Var":  "0.01",
					"Path.WtScale.Rel": "4", // err del 4: 4 > 6 > 8
					//"Path.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
				}},
			//{Sel: "#ECinToCA3", Desc: "ECin Perforant Path",
			//	Params: params.Params{
			//		"Path.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
			//	}},
			{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons: rel=2 still the best",
				Params: params.Params{
					"Path.WtScale.Rel": "2",   // 2 > 1 > .5 = .1
					"Path.Learn.Lrate": "0.1", // .1 > .08 (close) > .15 > .2 > .04;
					//"Path.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
				}},
			{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Path.Learn.Learn":       "true", // absolutely essential to have on! learning slow if off.
					"Path.CHL.Hebb":          "0.2",  // .2 seems good
					"Path.CHL.SAvgCor":       "0.1",  // 0.01 = 0.05 = .1 > .2 > .3 > .4 (listlize 20-100)
					"Path.CHL.MinusQ1":       "true", // dg self err slightly better
					"Path.Learn.Lrate":       "0.05", // .05 > .1 > .2 > .4; .01 less interference more learning time - key tradeoff param, .05 best for list20-100
					"Path.Learn.Momentum.On": "false",
					"Path.Learn.Norm.On":     "false",
					"Path.Learn.WtBal.On":    "true",
				}},
			{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
				Params: params.Params{
					"Path.CHL.Hebb":          "0.01", // .01 > .005 > .02 > .002 > .001 > .05 (crazy)
					"Path.CHL.SAvgCor":       "0.4",
					"Path.Learn.Lrate":       "0.1", // CHL: .1 =~ .08 > .15 > .2, .05 (sig worse)
					"Path.Learn.Momentum.On": "false",
					"Path.Learn.Norm.On":     "false",
					"Path.Learn.WtBal.On":    "true",
					//"Path.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
				}},
			//{Sel: "#ECinToCA1", Desc: "ECin Perforant Path",
			//	Params: params.Params{
			//		"Path.WtScale.Abs": "1.5", // zycyc, test if abs activation was not enough
			//	}},
			{Sel: "#ECoutToCA1", Desc: "ECout Perforant Path",
				Params: params.Params{
					"Path.WtScale.Rel": "0.3", // Back proj should generally be very weak but we're specifically setting this here bc others are set already
				}},
			{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level -- now for EC3 and EC5",
				Params: params.Params{
					"Layer.Act.Gbar.L":        "0.1",
					"Layer.Inhib.ActAvg.Init": "0.2",
					"Layer.Inhib.Layer.On":    "false",
					"Layer.Inhib.Pool.Gi":     "2.0",
					"Layer.Inhib.Pool.On":     "true",
				}},
			{Sel: "#DG", Desc: "very sparse = high inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.01",
					"Layer.Inhib.Layer.Gi":    "3.8", // 3.8 > 3.6 > 4.0 (too far -- tanks)
				}},
			{Sel: "#CA3", Desc: "sparse = high inhibition",
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
					//"Layer.Inhib.ActAvg.UseFirst": "false", // first activity is too low, throws off scaling, from Randy, zycyc: do we need this?
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
	{Name: "List125", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "125",
				}},
		},
	}},
	{Name: "List150", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "150",
				}},
		},
	}},
	{Name: "List175", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "175",
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
					"HipParams.DGRatio":   "2.236", // 1.5 before, sqrt(5) aligns with Ketz et al. 2013
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
					"HipParams.DGRatio":   "2.236", // 1.5 before
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
					"HipParams.DGRatio":   "2.236", // 1.5 before
				}},
		},
	}},
}
