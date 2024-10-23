// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// bench runs a benchmark model with 5 layers (3 hidden, Input, Output) all of the same
// size, for benchmarking different size networks.  These are not particularly realistic
// models for actual applications (e.g., large models tend to have much more topographic
// patterns of connectivity and larger layers with fewer connections), but they are
// easy to run..
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/leabra/v2/leabra"
)

var Net *leabra.Network
var Pats *table.Table
var EpcLog *table.Table
var Silent = false // non-verbose mode -- just reports result

var ParamSets = params.Sets{
	"Base": {
		{Sel: "Path", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
			Params: params.Params{
				"Path.Learn.Norm.On":     "true",
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.WtBal.On":    "false",
			}},
		{Sel: "Layer", Desc: "using default 1.8 inhib for all of network -- can explore",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.8",
				"Layer.Act.Gbar.L":     "0.2", // original value -- makes HUGE diff on perf!
			}},
		{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi": "1.4",
			}},
		{Sel: ".Back", Desc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Path.WtScale.Rel": "0.2",
			}},
	},
}

func ConfigNet(net *leabra.Network, units int) {
	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}

	inLay := net.AddLayer("Input", shp, leabra.InputLayer)
	hid1Lay := net.AddLayer("Hidden1", shp, leabra.SuperLayer)
	hid2Lay := net.AddLayer("Hidden2", shp, leabra.SuperLayer)
	hid3Lay := net.AddLayer("Hidden3", shp, leabra.SuperLayer)
	outLay := net.AddLayer("Output", shp, leabra.TargetLayer)

	net.ConnectLayers(inLay, hid1Lay, paths.NewFull(), leabra.ForwardPath)
	net.ConnectLayers(hid1Lay, hid2Lay, paths.NewFull(), leabra.ForwardPath)
	net.ConnectLayers(hid2Lay, hid3Lay, paths.NewFull(), leabra.ForwardPath)
	net.ConnectLayers(hid3Lay, outLay, paths.NewFull(), leabra.ForwardPath)

	net.ConnectLayers(outLay, hid3Lay, paths.NewFull(), leabra.BackPath)
	net.ConnectLayers(hid3Lay, hid2Lay, paths.NewFull(), leabra.BackPath)
	net.ConnectLayers(hid2Lay, hid1Lay, paths.NewFull(), leabra.BackPath)

	net.Defaults()
	net.ApplyParams(ParamSets["Base"], false) // no msg
	net.Build()
	net.InitWeights()
}

func ConfigPats(dt *table.Table, pats, units int) {
	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}
	// fmt.Printf("shape: %v\n", shp)

	dt.AddStringColumn("Name")
	dt.AddFloat32TensorColumn("Input", shp)
	dt.AddFloat32TensorColumn("Output", shp)
	dt.SetNumRows(pats)

	// note: actually can learn if activity is .15 instead of .25
	// but C++ benchmark is for .25..
	nOn := units / 6

	patgen.PermutedBinaryRows(dt.Columns[1], nOn, 1, 0)
	patgen.PermutedBinaryRows(dt.Columns[2], nOn, 1, 0)
}

func ConfigEpcLog(dt *table.Table) {
	dt.AddIntColumn("Epoch")
	dt.AddFloat32Column("CosDiff")
	dt.AddFloat32Column("AvgCosDiff")
	dt.AddFloat32Column("SSE")
	dt.AddFloat32Column("Avg SSE")
	dt.AddFloat32Column("Count Err")
	dt.AddFloat32Column("Pct Err")
	dt.AddFloat32Column("Pct Cor")
	dt.AddFloat32Column("Hid1 ActAvg")
	dt.AddFloat32Column("Hid2 ActAvg")
	dt.AddFloat32Column("Out ActAvg")
}

func TrainNet(net *leabra.Network, pats, epcLog *table.Table, epcs int) {
	ctx := leabra.NewContext()
	net.InitWeights()
	np := pats.NumRows()
	porder := rand.Perm(np) // randomly permuted order of ints

	epcLog.SetNumRows(epcs)

	inLay := net.LayerByName("Input")
	hid1Lay := net.LayerByName("Hidden1")
	hid2Lay := net.LayerByName("Hidden2")
	outLay := net.LayerByName("Output")

	_ = hid1Lay
	_ = hid2Lay

	inPats, _ := pats.ColumnByName("Input")
	outPats, _ := pats.ColumnByName("Output")

	tmr := timer.Time{}
	tmr.Start()
	for epc := 0; epc < epcs; epc++ {
		randx.PermuteInts(porder)
		outCosDiff := float32(0)
		cntErr := 0
		sse := 0.0
		avgSSE := 0.0
		for pi := 0; pi < np; pi++ {
			ppi := porder[pi]
			inp := inPats.SubSpace([]int{ppi})
			outp := outPats.SubSpace([]int{ppi})

			inLay.ApplyExt(inp)
			outLay.ApplyExt(outp)

			net.AlphaCycInit(true)
			ctx.AlphaCycStart()
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < ctx.CycPerQtr; cyc++ {
					net.Cycle(ctx)
					ctx.CycleInc()
				}
				net.QuarterFinal(ctx)
				ctx.QuarterInc()
			}
			net.DWt()
			net.WtFromDWt()
			outCosDiff += outLay.CosDiff.Cos
			pSSE, pAvgSSE := outLay.MSE(0.5)
			sse += pSSE
			avgSSE += pAvgSSE
			if pSSE != 0 {
				cntErr++
			}
		}
		outCosDiff /= float32(np)
		sse /= float64(np)
		avgSSE /= float64(np)
		pctErr := float64(cntErr) / float64(np)
		pctCor := 1 - pctErr
		// fmt.Printf("epc: %v  \tCosDiff: %v \tAvgCosDif: %v\n", epc, outCosDiff, outLay.CosDiff.Avg)
		epcLog.SetFloat("Epoch", epc, float64(epc))
		epcLog.SetFloat("CosDiff", epc, float64(outCosDiff))
		epcLog.SetFloat("AvgCosDiff", epc, float64(outLay.CosDiff.Avg))
		epcLog.SetFloat("SSE", epc, sse)
		epcLog.SetFloat("Avg SSE", epc, avgSSE)
		epcLog.SetFloat("Count Err", epc, float64(cntErr))
		epcLog.SetFloat("Pct Err", epc, pctErr)
		epcLog.SetFloat("Pct Cor", epc, pctCor)
		epcLog.SetFloat("Hid1 ActAvg", epc, float64(hid1Lay.Pools[0].ActAvg.ActPAvgEff))
		epcLog.SetFloat("Hid2 ActAvg", epc, float64(hid2Lay.Pools[0].ActAvg.ActPAvgEff))
		epcLog.SetFloat("Out ActAvg", epc, float64(outLay.Pools[0].ActAvg.ActPAvgEff))
	}
	tmr.Stop()
	if Silent {
		fmt.Printf("%v\n", tmr.Total)
	} else {
		fmt.Printf("Took %v for %v epochs, avg per epc: m%6.4g\n", tmr.Total, epcs, float64(tmr.Total)/float64(int(time.Second)*epcs))
	}
}

func main() {
	var epochs int
	var pats int
	var units int

	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
	}

	// process command args
	flag.IntVar(&epochs, "epochs", 2, "number of epochs to run")
	flag.IntVar(&pats, "pats", 10, "number of patterns per epoch")
	flag.IntVar(&units, "units", 100, "number of units per layer -- uses NxN where N = sqrt(units)")
	flag.BoolVar(&Silent, "silent", false, "only report the final time")
	flag.Parse()

	if !Silent {
		fmt.Printf("Running bench with: %v epochs, %v pats, %v units\n", epochs, pats, units)
	}

	Net = leabra.NewNetwork("Bench")
	ConfigNet(Net, units)

	Pats = &table.Table{}
	ConfigPats(Pats, pats, units)

	EpcLog = &table.Table{}
	ConfigEpcLog(EpcLog)

	TrainNet(Net, Pats, EpcLog, epochs)

	EpcLog.SaveCSV("bench_epc.dat", ',', table.Headers)
}
