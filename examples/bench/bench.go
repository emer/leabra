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

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/timer"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/leabra/leabra"
)

var Net *leabra.Network
var Pats *etable.Table
var EpcLog *etable.Table
var Thread = false // much slower for small net

var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "true",
					"Prjn.Learn.Momentum.On": "true",
					"Prjn.Learn.WtBal.On":    "false",
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.4",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "0.2",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "50",
				}},
		},
	}},
}

func ConfigNet(net *leabra.Network, threads, units int) {
	net.InitName(net, "BenchNet")

	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}

	inLay := net.AddLayer("Input", shp, emer.Input)
	hid1Lay := net.AddLayer("Hidden1", shp, emer.Hidden)
	hid2Lay := net.AddLayer("Hidden2", shp, emer.Hidden)
	hid3Lay := net.AddLayer("Hidden3", shp, emer.Hidden)
	outLay := net.AddLayer("Output", shp, emer.Target)

	net.ConnectLayers(inLay, hid1Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid1Lay, hid2Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid2Lay, hid3Lay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(hid3Lay, outLay, prjn.NewFull(), emer.Forward)

	net.ConnectLayers(outLay, hid3Lay, prjn.NewFull(), emer.Back)
	net.ConnectLayers(hid3Lay, hid2Lay, prjn.NewFull(), emer.Back)
	net.ConnectLayers(hid2Lay, hid1Lay, prjn.NewFull(), emer.Back)

	switch threads {
	case 2:
		hid3Lay.SetThread(1)
		outLay.SetThread(1)
	case 4:
		hid2Lay.SetThread(1)
		hid3Lay.SetThread(2)
		outLay.SetThread(3)
	}

	net.Defaults()
	net.ApplyParams(ParamSets[0].Sheets["Network"], false) // no msg
	net.Build()
	net.InitWts()
}

func ConfigPats(dt *etable.Table, pats, units int) {
	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}
	fmt.Printf("shape: %v\n", shp)

	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, shp, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, shp, []string{"Y", "X"}},
	}, pats)

	// note: actually can learn if activity is .15 instead of .25
	// but C++ benchmark is for .25..
	nOn := units / 6

	patgen.PermutedBinaryRows(dt.Cols[1], nOn, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], nOn, 1, 0)
}

func ConfigEpcLog(dt *etable.Table) {
	dt.SetFromSchema(etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"CosDiff", etensor.FLOAT32, nil, nil},
		{"AvgCosDiff", etensor.FLOAT32, nil, nil},
		{"SSE", etensor.FLOAT32, nil, nil},
		{"Avg SSE", etensor.FLOAT32, nil, nil},
		{"Count Err", etensor.FLOAT32, nil, nil},
		{"Pct Err", etensor.FLOAT32, nil, nil},
		{"Pct Cor", etensor.FLOAT32, nil, nil},
		{"Hid1 ActAvg", etensor.FLOAT32, nil, nil},
		{"Hid2 ActAvg", etensor.FLOAT32, nil, nil},
		{"Out ActAvg", etensor.FLOAT32, nil, nil},
	}, 0)
}

func TrainNet(net *leabra.Network, pats, epcLog *etable.Table, epcs int) {
	ltime := leabra.NewTime()
	net.InitWts()
	np := pats.NumRows()
	porder := rand.Perm(np) // randomly permuted order of ints

	epcLog.SetNumRows(epcs)

	inLay := net.LayerByName("Input").(*leabra.Layer)
	hid1Lay := net.LayerByName("Hidden1").(*leabra.Layer)
	hid2Lay := net.LayerByName("Hidden2").(*leabra.Layer)
	outLay := net.LayerByName("Output").(*leabra.Layer)

	_ = hid1Lay
	_ = hid2Lay

	inPats := pats.ColByName("Input").(*etensor.Float32)
	outPats := pats.ColByName("Output").(*etensor.Float32)

	tmr := timer.Time{}
	tmr.Start()
	for epc := 0; epc < epcs; epc++ {
		erand.PermuteInts(porder)
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

			net.AlphaCycInit()
			ltime.AlphaCycStart()
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < ltime.CycPerQtr; cyc++ {
					net.Cycle(ltime)
					ltime.CycleInc()
				}
				net.QuarterFinal(ltime)
				ltime.QuarterInc()
			}
			net.DWt()
			net.WtFmDWt()
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
		epcLog.SetCellFloat("Epoch", epc, float64(epc))
		epcLog.SetCellFloat("CosDiff", epc, float64(outCosDiff))
		epcLog.SetCellFloat("AvgCosDiff", epc, float64(outLay.CosDiff.Avg))
		epcLog.SetCellFloat("SSE", epc, sse)
		epcLog.SetCellFloat("Avg SSE", epc, avgSSE)
		epcLog.SetCellFloat("Count Err", epc, float64(cntErr))
		epcLog.SetCellFloat("Pct Err", epc, pctErr)
		epcLog.SetCellFloat("Pct Cor", epc, pctCor)
		epcLog.SetCellFloat("Hid1 ActAvg", epc, float64(hid1Lay.Pools[0].ActAvg.ActPAvgEff))
		epcLog.SetCellFloat("Hid2 ActAvg", epc, float64(hid2Lay.Pools[0].ActAvg.ActPAvgEff))
		epcLog.SetCellFloat("Out ActAvg", epc, float64(outLay.Pools[0].ActAvg.ActPAvgEff))
	}
	tmr.Stop()
	fmt.Printf("Took %6.4g secs for %v epochs, avg per epc: %6.4g\n", tmr.TotalSecs(), epcs, tmr.TotalSecs()/float64(epcs))
	net.TimerReport()
}

func main() {
	var threads int
	var epochs int
	var pats int
	var units int

	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
	}

	// process command args
	flag.IntVar(&threads, "threads", 1, "number of threads (goroutines) to use")
	flag.IntVar(&epochs, "epochs", 2, "number of epochs to run")
	flag.IntVar(&pats, "pats", 10, "number of patterns per epoch")
	flag.IntVar(&units, "units", 100, "number of units per layer -- uses NxN where N = sqrt(units)")
	flag.Parse()

	fmt.Printf("Running bench with: %v threads, %v epochs, %v pats, %v units\n", threads, epochs, pats, units)

	Net = &leabra.Network{}
	ConfigNet(Net, threads, units)

	Pats = &etable.Table{}
	ConfigPats(Pats, pats, units)

	EpcLog = &etable.Table{}
	ConfigEpcLog(EpcLog)

	TrainNet(Net, Pats, EpcLog, epochs)

	EpcLog.SaveCSV("bench_epc.dat", ',', etable.Headers)
}
