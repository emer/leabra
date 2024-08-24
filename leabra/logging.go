// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"reflect"
	"strconv"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/plot/plotcore"
	"cogentcore.org/core/tensor/stats/split"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
)

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func LogTestErrors(lg *elog.Logs) {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := lg.TableDetailsScope(sk)
	ix, _ := lt.NamedIndexView("TestErrors")
	ix.Filter(func(et *table.Table, row int) bool {
		return et.Float("Err", row) > 0 // include error trials
	})
	lg.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.AggColumn(allsp, "UnitErr", stats.Sum)
	// note: can add other stats to compute
	lg.MiscTables["TestErrorStats"] = allsp.AggsToTable(table.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func PCAStats(net *Network, lg *elog.Logs, stats *estats.Stats) {
	stats.PCAStats(lg.IndexView(etime.Analyze, etime.Trial), "ActM", net.LayersByType(SuperLayer, TargetLayer, CTLayer, PTPredLayer))
}

//////////////////////////////////////////////////////////////////////////////
//  Log items

// LogAddDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across the given time levels, in higher to lower order, e.g., Epoch, Trial
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddDiagnosticItems(lg *elog.Logs, layerNames []string, mode etime.Modes, times ...etime.Times) {
	ntimes := len(times)
	for _, lnm := range layerNames {
		clnm := lnm
		itm := lg.AddItem(&elog.Item{
			Name:   clnm + "_ActMAvg",
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pools[0].ActAvg.ActMAvg)
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_ActMMax",
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pools[0].ActM.Max)
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:  clnm + "_CosDiff",
			Type:  reflect.Float64,
			Range: minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.CosDiff.Cos)
				}}})
		lg.AddStdAggs(itm, mode, times...)
	}
}

func LogInputLayer(lg *elog.Logs, net *Network, mode etime.Modes) {
	// input layer average activity -- important for tuning
	layerNames := net.LayersByType(InputLayer)
	for _, lnm := range layerNames {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   reflect.Float64,
			FixMax: true,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pools[0].ActM.Max)
				}}})
	}
}

// LogAddPCAItems adds PCA statistics to log for Hidden and Target layers
// across the given time levels, in higher to lower order, e.g., Run, Epoch, Trial
// These are useful for diagnosing the behavior of the network.
func LogAddPCAItems(lg *elog.Logs, net *Network, mode etime.Modes, times ...etime.Times) {
	ntimes := len(times)
	layers := net.LayersByType(SuperLayer, TargetLayer, CTLayer, PTPredLayer)
	for _, lnm := range layers {
		clnm := lnm
		cly := net.LayerByName(clnm)
		lg.AddItem(&elog.Item{
			Name:      clnm + "_ActM",
			Type:      reflect.Float64,
			CellShape: cly.GetSampleShape().Sizes,
			FixMax:    true,
			Range:     minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Analyze, times[ntimes-1]): func(ctx *elog.Context) {
					ctx.SetLayerSampleTensor(clnm, "ActM")
				}, etime.Scope(etime.Test, times[ntimes-1]): func(ctx *elog.Context) {
					ctx.SetLayerSampleTensor(clnm, "ActM")
				}}})
		itm := lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_NStrong",
			Type: reflect.Float64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1]...)

		itm = lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Top5",
			Type: reflect.Float64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1]...)

		itm = lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Next5",
			Type: reflect.Float64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1]...)

		itm = lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Rest",
			Type: reflect.Float64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1]...)
	}
}

// LayerActsLogConfigMetaData configures meta data for LayerActs table
func LayerActsLogConfigMetaData(dt *table.Table) {
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(elog.LogPrec))
	dt.SetMetaData("Type", "Bar")
	dt.SetMetaData("XAxis", "Layer")
	dt.SetMetaData("XAxisRot", "45")
	dt.SetMetaData("Nominal:On", "+")
	dt.SetMetaData("Nominal:FixMin", "+")
	dt.SetMetaData("ActM:On", "+")
	dt.SetMetaData("ActM:FixMin", "+")
	dt.SetMetaData("ActM:Max", "1")
	dt.SetMetaData("ActP:FixMin", "+")
	dt.SetMetaData("ActP:Max", "1")
	dt.SetMetaData("MaxGeM:FixMin", "+")
	dt.SetMetaData("MaxGeM:FixMax", "+")
	dt.SetMetaData("MaxGeM:Max", "3")
	dt.SetMetaData("MaxGeP:FixMin", "+")
	dt.SetMetaData("MaxGeP:FixMax", "+")
	dt.SetMetaData("MaxGeP:Max", "3")
}

// LayerActsLogConfig configures Tables to record
// layer activity for tuning the network inhibition, nominal activity,
// relative scaling, etc. in elog.MiscTables:
// LayerActs is current, LayerActsRec is record over trials,
// LayerActsAvg is average of recorded trials.
func LayerActsLogConfig(net *Network, lg *elog.Logs) {
	dt := lg.MiscTable("LayerActs")
	dt.SetMetaData("name", "LayerActs")
	dt.SetMetaData("desc", "Layer Activations")
	LayerActsLogConfigMetaData(dt)
	dtRec := lg.MiscTable("LayerActsRec")
	dtRec.SetMetaData("name", "LayerActsRec")
	dtRec.SetMetaData("desc", "Layer Activations Recorded")
	LayerActsLogConfigMetaData(dtRec)
	dtAvg := lg.MiscTable("LayerActsAvg")
	dtAvg.SetMetaData("name", "LayerActsAvg")
	dtAvg.SetMetaData("desc", "Layer Activations Averaged")
	LayerActsLogConfigMetaData(dtAvg)
	dts := []*table.Table{dt, dtRec, dtAvg}
	for _, t := range dts {
		t.AddStringColumn("Layer")
		t.AddFloat64Column("Nominal")
		t.AddFloat64Column("ActM")
		t.AddFloat64Column("ActP")
	}
	nlay := len(net.Layers)
	dt.SetNumRows(nlay)
	dtRec.SetNumRows(0)
	dtAvg.SetNumRows(nlay)
	for li, ly := range net.Layers {
		dt.SetString("Layer", li, ly.Name)
		dt.SetFloat("Nominal", li, float64(ly.Inhib.ActAvg.Init))
		dtAvg.SetString("Layer", li, ly.Name)
	}
}

// LayerActsLog records layer activity for tuning the network
// inhibition, nominal activity, relative scaling, etc.
// if gui is non-nil, plot is updated.
func LayerActsLog(net *Network, lg *elog.Logs, di int, gui *egui.GUI) {
	dt := lg.MiscTable("LayerActs")
	dtRec := lg.MiscTable("LayerActsRec")
	for li, ly := range net.Layers {
		lpl := &ly.Pools[0]
		dt.SetFloat("Nominal", li, float64(ly.Inhib.ActAvg.Init))
		dt.SetFloat("ActM", li, float64(lpl.ActAvg.ActMAvg))
		dt.SetFloat("ActP", li, float64(lpl.ActAvg.ActPAvg))
		dtRec.SetNumRows(dtRec.Rows + 1)
		dtRec.SetString("Layer", li, ly.Name)
		dtRec.SetFloat("Nominal", li, float64(ly.Inhib.ActAvg.Init))
		dtRec.SetFloat("ActM", li, float64(lpl.ActAvg.ActMAvg))
		dtRec.SetFloat("ActP", li, float64(lpl.ActAvg.ActPAvg))
	}
	if gui != nil {
		gui.UpdatePlotScope(etime.ScopeKey("LayerActs"))
	}
}

// LayerActsLogAvg computes average of LayerActsRec record
// of layer activity for tuning the network
// inhibition, nominal activity, relative scaling, etc.
// if gui is non-nil, plot is updated.
// if recReset is true, reset the recorded data after computing average.
func LayerActsLogAvg(net *Network, lg *elog.Logs, gui *egui.GUI, recReset bool) {
	dtRec := lg.MiscTable("LayerActsRec")
	dtAvg := lg.MiscTable("LayerActsAvg")
	if dtRec.Rows == 0 {
		return
	}
	ix := table.NewIndexView(dtRec)
	spl := split.GroupBy(ix, "Layer")
	split.AggAllNumericCols(spl, stats.Mean)
	ags := spl.AggsToTable(table.ColumnNameOnly)
	cols := []string{"Nominal", "ActM", "ActP", "MaxGeM", "MaxGeP"}
	for li, ly := range net.Layers {
		rw := errors.Log1(ags.RowsByString("Layer", ly.Name, table.Equals, table.UseCase))[0]
		for _, cn := range cols {
			dtAvg.SetFloat(cn, li, ags.Float(cn, rw))
		}
	}
	if recReset {
		dtRec.SetNumRows(0)
	}
	if gui != nil {
		gui.UpdatePlotScope(etime.ScopeKey("LayerActsAvg"))
	}
}

// LayerActsLogRecReset resets the recorded LayerActsRec data
// used for computing averages
func LayerActsLogRecReset(lg *elog.Logs) {
	dtRec := lg.MiscTable("LayerActsRec")
	dtRec.SetNumRows(0)
}

// LayerActsLogConfigGUI configures GUI for LayerActsLog Plot and LayerActs Avg Plot
func LayerActsLogConfigGUI(lg *elog.Logs, gui *egui.GUI) {
	pt := gui.Tabs.NewTab("LayerActs Plot")
	plt := plotcore.NewPlotEditor(pt)
	gui.Plots["LayerActs"] = plt
	plt.SetTable(lg.MiscTables["LayerActs"])

	pt = gui.Tabs.NewTab("LayerActs Avg Plot")
	plt = plotcore.NewPlotEditor(pt)
	gui.Plots["LayerActsAvg"] = plt
	plt.SetTable(lg.MiscTables["LayerActsAvg"])
}
