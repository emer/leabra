// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"reflect"
	"strings"
	"time"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/enums"
	"cogentcore.org/lab/matrix"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/emergent/v2/looper"
)

// StatsNode returns tensorfs Dir Node for given mode, level.
func StatsNode(statsDir *tensorfs.Node, mode, level enums.Enum) *tensorfs.Node {
	modeDir := statsDir.Dir(mode.String())
	return modeDir.Dir(level.String())
}

func StatsLayerValues(net *Network, curDir *tensorfs.Node, mode enums.Enum, di int, layName, varName string) *tensor.Float64 {
	curModeDir := curDir.Dir(mode.String())
	ly := net.LayerByName(layName)
	tsr := curModeDir.Float64(layName+"_"+varName, ly.Shape.Sizes...)
	ly.UnitValuesTensor(tsr, varName, di)
	return tsr
}

// LogFilename returns a standard log file name as netName_runName_logName.tsv
func LogFilename(netName, runName, logName string) string {
	return netName + "_" + runName + "_" + logName + ".tsv"
}

// OpenLogFile, if on == true, sets the log file for given table using given
// netName, runName, and logName in order.
func OpenLogFile(on bool, dt *table.Table, netName, runName, logName string) {
	if !on {
		return
	}
	fnm := LogFilename(netName, runName, logName)
	tensor.SetPrecision(dt, 4)
	dt.OpenLog(fnm, tensor.Tab)
}

// OpenLogFiles opens the log files for modes and levels of the looper,
// based on the lists of level names, ordered by modes in numerical order.
// The netName and runName are used for naming the file, along with
// the mode_level in lower case.
func OpenLogFiles(ls *looper.Stacks, statsDir *tensorfs.Node, netName, runName string, modeLevels [][]string) {
	modes := ls.Modes()
	for i, mode := range modes {
		if i >= len(modeLevels) {
			return
		}
		levels := modeLevels[i]
		st := ls.Stacks[mode]
		for _, level := range st.Order {
			on := false
			for _, lev := range levels {
				if lev == level.String() {
					on = true
					break
				}
			}
			if !on {
				continue
			}
			logName := strings.ToLower(mode.String() + "_" + level.String())
			dt := tensorfs.DirTable(StatsNode(statsDir, mode, level), nil)
			fnm := LogFilename(netName, runName, logName)
			tensor.SetPrecision(dt, 4)
			dt.OpenLog(fnm, tensor.Tab)
		}
	}
}

// CloseLogFiles closes all the log files for each mode and level of the looper,
// Excluding given level(s).
func CloseLogFiles(ls *looper.Stacks, statsDir *tensorfs.Node, exclude ...enums.Enum) {
	modes := ls.Modes() // mode enum order
	for _, mode := range modes {
		st := ls.Stacks[mode]
		for _, level := range st.Order {
			if StatExcludeLevel(level, exclude...) {
				continue
			}
			dt := tensorfs.DirTable(StatsNode(statsDir, mode, level), nil)
			dt.CloseLog()
		}
	}
}

// StatExcludeLevel returns true if given level is among the list of levels to exclude.
func StatExcludeLevel(level enums.Enum, exclude ...enums.Enum) bool {
	bail := false
	for _, ex := range exclude {
		if level == ex {
			bail = true
			break
		}
	}
	return bail
}

// StatLoopCounters adds the counters from each stack, loop level for given
// looper Stacks to the given tensorfs stats. This is typically the first
// Stat to add, so these counters will be used for X axis values.
// The stat is run with start = true before returning, so that the stats
// are already initialized first before anything else.
// The first mode's counters (typically Train) are automatically added to all
// subsequent modes so they automatically track training levels.
//   - currentDir is a tensorfs directory to store the current values of each counter.
//   - trialLevel is the Trial level enum, which automatically handles the
//     iteration over ndata parallel trials.
//   - exclude is a list of loop levels to exclude (e.g., Cycle).
func StatLoopCounters(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum, exclude ...enums.Enum) func(mode, level enums.Enum, start bool) {
	modes := ls.Modes() // mode enum order
	fun := func(mode, level enums.Enum, start bool) {
		for mi := range 2 {
			st := ls.Stacks[mode]
			prefix := ""
			if mi == 0 {
				if modes[mi].Int64() == mode.Int64() { // skip train in train..
					continue
				}
				ctrMode := modes[mi]
				st = ls.Stacks[ctrMode]
				prefix = ctrMode.String()
			}
			for _, lev := range st.Order {
				// don't record counter for levels above it
				if level.Int64() > lev.Int64() {
					continue
				}
				if StatExcludeLevel(lev, exclude...) {
					continue
				}
				name := prefix + lev.String() // name of stat = level
				ndata := 1
				modeDir := statsDir.Dir(mode.String())
				curModeDir := currentDir.Dir(mode.String())
				levelDir := modeDir.Dir(level.String())
				tsr := levelDir.Int(name)
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0)
					})
					if level.Int64() == trialLevel.Int64() {
						for di := range ndata {
							curModeDir.Int(name, ndata).SetInt1D(0, di)
						}
					}
					continue
				}
				ctr := st.Loops[lev].Counter.Cur
				if level.Int64() == trialLevel.Int64() {
					for di := range ndata {
						curModeDir.Int(name, ndata).SetInt1D(ctr, di)
						tsr.AppendRowInt(ctr)
						if lev.Int64() == trialLevel.Int64() {
							ctr++
						}
					}
				} else {
					curModeDir.Int(name, 1).SetInt1D(ctr, 0)
					tsr.AppendRowInt(ctr)
				}
			}
		}
	}
	for _, md := range modes {
		st := ls.Stacks[md]
		for _, lev := range st.Order {
			if StatExcludeLevel(lev, exclude...) {
				continue
			}
			fun(md, lev, true)
		}
	}
	return fun
}

// StatRunName adds a "RunName" stat to every mode and level of looper,
// subject to exclusion list, which records the current value of the
// "RunName" string in ss.Current, which identifies the parameters and tag
// for this run.
func StatRunName(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum, exclude ...enums.Enum) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		name := "RunName"
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		tsr := levelDir.StringValue(name)
		ndata := 1
		runNm := currentDir.StringValue(name, 1).String1D(0)

		if start {
			tsr.SetNumRows(0)
			return
		}
		if level.Int64() == trialLevel.Int64() {
			for range ndata {
				tsr.AppendRowString(runNm)
			}
		} else {
			tsr.AppendRowString(runNm)
		}
	}
}

// StatTrialName adds a "TrialName" stat to the given Trial level in every mode of looper,
// which records the current value of the "TrialName" string in ss.Current, which
// contains a string description of the current trial.
func StatTrialName(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		if level.Int64() != trialLevel.Int64() {
			return
		}
		name := "TrialName"
		modeDir := statsDir.Dir(mode.String())
		curModeDir := currentDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		tsr := levelDir.StringValue(name)
		ndata := 1
		if start {
			tsr.SetNumRows(0)
			return
		}
		for di := range ndata {
			trlNm := curModeDir.StringValue(name, ndata).String1D(di)
			tsr.AppendRowString(trlNm)
		}
	}
}

// StatPerTrialMSec returns a Stats function that reports the number of milliseconds
// per trial, for the given levels and training mode enum values.
// Stats will be recorded a levels above the given trial level.
func StatPerTrialMSec(statsDir *tensorfs.Node, trainMode enums.Enum, trialLevel enums.Enum) func(mode, level enums.Enum, start bool) {
	var epcTimer timer.Time
	levels := make([]enums.Enum, 10) // should be enough
	levels[0] = trialLevel
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi <= 0 {
			return
		}
		levels[levi] = level
		name := "PerTrialMSec"
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		tsr := levelDir.Float64(name)
		if start {
			tsr.SetNumRows(0)
			plot.SetFirstStyler(tsr, func(s *plot.Style) {
				s.Range.SetMin(0)
			})
			return
		}
		switch levi {
		case 1:
			epcTimer.Stop()
			subDir := modeDir.Dir(levels[0].String())
			trls := errors.Ignore1(subDir.Values())[0] // must be a stat
			epcTimer.N = trls.Len()
			pertrl := float64(epcTimer.Avg()) / float64(time.Millisecond)
			tsr.AppendRowFloat(pertrl)
			epcTimer.ResetStart()
		default:
			subDir := modeDir.Dir(levels[levi-1].String())
			tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
		}
	}
}

// StatLayerActGe returns a Stats function that computes layer activity
// and Ge (excitatory conductdance; net input) stats, which are important targets
// of parameter tuning to ensure everything is in an appropriate dynamic range.
// It only runs for given trainMode at given trialLevel and above,
// with higher levels computing the Mean of lower levels.
func StatLayerActGe(statsDir *tensorfs.Node, net *Network, trainMode, trialLevel, runLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool) {
	statNames := []string{"ActMAvg", "ActMMax"} //, "MaxGeM"}
	levels := make([]enums.Enum, 10)            // should be enough
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		// ndata := 1
		for _, lnm := range layerNames {
			for si, statName := range statNames {
				ly := net.LayerByName(lnm)
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0)
					})
					continue
				}
				switch levi {
				case 0:
					var stat float32
					switch si {
					case 0:
						stat = ly.Pools[0].ActAvg.ActMAvg
					case 1:
						stat = ly.Pools[0].ActM.Max
						// case 2:
						// 	stat = PoolAvgMax(AMGeInt, AMMinus, Max, lpi, di)
					}
					tsr.AppendRowFloat(float64(stat))
				case int(runLevel.Int64() - trialLevel.Int64()):
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
				default:
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
				}
			}
		}
	}
}

// StatLayerState returns a Stats function that records layer state
// It runs for given mode and level, recording given variable
// for given layer names. if isTrialLevel is true, the level is a
// trial level that needs iterating over NData.
func StatLayerState(statsDir *tensorfs.Node, net *Network, smode, slevel enums.Enum, isTrialLevel bool, variable string, layerNames ...string) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		if mode.Int64() != smode.Int64() || level.Int64() != slevel.Int64() {
			return
		}
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		ndata := 1
		// if !isTrialLevel {
		// 	ndata = 1
		// }
		for _, lnm := range layerNames {
			ly := net.LayerByName(lnm)
			name := lnm + "_" + variable
			sizes := []int{ndata}
			sizes = append(sizes, ly.GetSampleShape().Sizes...)
			tsr := levelDir.Float64(name, sizes...)
			if start {
				tsr.SetNumRows(0)
				continue
			}
			for di := range ndata {
				row := tsr.DimSize(0)
				tsr.SetNumRows(row + 1)
				rtsr := tsr.RowTensor(row)
				ly.UnitValuesSampleTensor(rtsr, variable, di)
			}
		}
	}
}

// PCAStrongThr is the threshold for counting PCA eigenvalues as "strong".
var PCAStrongThr = 0.01

// StatPCA returns a Stats function that computes PCA NStrong, Top5, Next5, and Rest
// stats, which are important for tracking hogging dynamics where the representational
// space is not efficiently distributed. Uses Sample units for layers, and SVD computation
// is reasonably efficient.
// It only runs for given trainMode, from given Trial level upward,
// with higher levels computing the Mean of lower levels.
// Trial level just records ActM values for layers in a separate PCA subdir,
// which are input to next level computation where PCA is computed.
func StatPCA(statsDir, currentDir *tensorfs.Node, net *Network, interval int, trainMode, trialLevel, runLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool, epc int) {
	statNames := []string{"PCA_NStrong", "PCA_Top5", "PCA_Next", "PCA_Rest"}
	levels := make([]enums.Enum, 10) // should be enough
	return func(mode, level enums.Enum, start bool, epc int) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.Dir(mode.String())
		curModeDir := currentDir.Dir(mode.String())
		curPCADir := curModeDir.Dir("PCA")
		pcaDir := statsDir.Dir("PCA")
		levelDir := modeDir.Dir(level.String())
		ndata := 1
		for _, lnm := range layerNames {
			ly := net.LayerByName(lnm)
			sizes := []int{ndata}
			sizes = append(sizes, ly.GetSampleShape().Sizes...)
			vtsr := pcaDir.Float64(lnm, sizes...)
			if levi == 0 {
				ltsr := curPCADir.Float64(lnm+"_ActM", ly.GetSampleShape().Sizes...)
				if start {
					vtsr.SetNumRows(0)
				} else {
					for di := range ndata {
						ly.UnitValuesSampleTensor(ltsr, "ActM", di)
						vtsr.AppendRow(ltsr)
					}
				}
				continue
			}
			var svals [4]float64 // in statNames order
			hasNew := false
			if !start && levi == 1 {
				if interval > 0 && epc%interval == 0 {
					hasNew = true
					vals := curPCADir.Float64("Vals_" + lnm)
					covar := curPCADir.Float64("Covar_" + lnm)
					metric.CovarianceMatrixOut(metric.Covariance, vtsr, covar)
					matrix.SVDValuesOut(covar, vals)
					ln := vals.Len()
					for i := range ln {
						v := vals.Float1D(i)
						if v < PCAStrongThr {
							svals[0] = float64(i)
							break
						}
					}
					for i := range 5 {
						if ln >= 5 {
							svals[1] += vals.Float1D(i)
						}
						if ln >= 10 {
							svals[2] += vals.Float1D(i + 5)
						}
					}
					svals[1] /= 5
					svals[2] /= 5
					if ln > 10 {
						sum := stats.Sum(vals).Float1D(0)
						svals[3] = (sum - (svals[1] + svals[2])) / float64(ln-10)
					}
				}
			}
			for si, statName := range statNames {
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0)
					})
					continue
				}
				switch levi {
				case 1:
					var stat float64
					nr := tsr.DimSize(0)
					if nr > 0 {
						stat = tsr.FloatRow(nr-1, 0)
					}
					if hasNew {
						stat = svals[si]
					}
					tsr.AppendRowFloat(float64(stat))
				case int(runLevel.Int64() - trialLevel.Int64()):
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
				default:
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
				}
			}
		}
	}
}

// StatCorSim returns a Stats function that records 1 - [LayerPhaseDiff] stats,
// i.e., Correlation-based similarity, for given layer names.
func StatCorSim(statsDir, currentDir *tensorfs.Node, net *Network, trialLevel, runLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool) {
	levels := make([]enums.Enum, 10) // should be enough
	levels[0] = trialLevel
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.Dir(mode.String())
		curModeDir := currentDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		ndata := 1
		for _, lnm := range layerNames {
			ly := net.LayerByName(lnm)
			name := lnm + "_CorSim"
			tsr := levelDir.Float64(name)
			if start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					s.On = true
				})
				continue
			}
			switch levi {
			case 0: // trial
				for di := range ndata {
					stat := 1.0 - float64(ly.CosDiff.Cos)
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					tsr.AppendRowFloat(float64(stat))
				}
			case int(runLevel.Int64() - trialLevel.Int64()):
				subDir := modeDir.Dir(levels[levi-1].String())
				tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
			default:
				subDir := modeDir.Dir(levels[levi-1].String())
				tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
			}
		}
	}
}

// StatPrevCorSim returns a Stats function that compute correlations
// between previous trial activity state and current minus phase and
// plus phase state. This is important for predictive learning.
func StatPrevCorSim(statsDir, currentDir *tensorfs.Node, net *Network, trialLevel, runLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool) {
	statNames := []string{"PrevToM", "PrevToP"}
	levels := make([]enums.Enum, 10) // should be enough
	levels[0] = trialLevel
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.Dir(mode.String())
		curModeDir := currentDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		ndata := 1
		for _, lnm := range layerNames {
			for si, statName := range statNames {
				ly := net.LayerByName(lnm)
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					plot.SetFirstStyler(tsr, func(s *plot.Style) {
						s.Range.SetMin(0).SetMax(1)
					})
					continue
				}
				switch levi {
				case 0:
					// note: current lnm + _var is standard reusable unit vals buffer
					actM := curModeDir.Float64(lnm+"_ActM", ly.GetSampleShape().Sizes...)
					actP := curModeDir.Float64(lnm+"_ActP", ly.GetSampleShape().Sizes...)
					// note: CaD is sufficiently stable that it is fine to compare with ActM and ActP
					prev := curModeDir.Float64(lnm+"_CaDPrev", ly.GetSampleShape().Sizes...)
					for di := range ndata {
						ly.UnitValuesSampleTensor(prev, "CaDPrev", di)
						prev.SetShapeSizes(prev.Len()) // set to 1D -- inexpensive and faster for computation
						var stat float64
						switch si {
						case 0:
							ly.UnitValuesSampleTensor(actM, "ActM", di)
							actM.SetShapeSizes(actM.Len())
							cov := metric.Correlation(actM, prev)
							stat = cov.Float1D(0)
						case 1:
							ly.UnitValuesSampleTensor(actP, "ActP", di)
							actP.SetShapeSizes(actP.Len())
							cov := metric.Correlation(actP, prev)
							stat = cov.Float1D(0)
						}
						curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
						tsr.AppendRowFloat(stat)
					}
				case int(runLevel.Int64() - trialLevel.Int64()):
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatFinal.Call(subDir.Value(name)))
				default:
					subDir := modeDir.Dir(levels[levi-1].String())
					tsr.AppendRow(stats.StatMean.Call(subDir.Value(name)))
				}
			}
		}
	}
}

// StatLevelAll returns a Stats function that copies stats from given mode
// and level, without resetting at the start, to accumulate all rows
// over time until reset manually. The styleFunc, if non-nil, does plot styling
// based on the current column.
func StatLevelAll(statsDir *tensorfs.Node, srcMode, srcLevel enums.Enum, styleFunc func(s *plot.Style, col tensor.Values)) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		if srcMode.Int64() != mode.Int64() || srcLevel.Int64() != level.Int64() {
			return
		}
		modeDir := statsDir.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		allDir := modeDir.Dir(level.String() + "All")
		cols := levelDir.NodesFunc(nil) // all nodes
		for _, cl := range cols {
			clv := cl.Tensor.(tensor.Values)
			if clv.NumDims() == 0 || clv.DimSize(0) == 0 {
				continue
			}
			if start {
				trg := tensorfs.ValueType(allDir, cl.Name(), clv.DataType(), clv.ShapeSizes()...)
				if trg.Len() == 0 {
					if styleFunc != nil {
						plot.SetFirstStyler(trg, func(s *plot.Style) {
							styleFunc(s, clv)
						})
					}
					trg.SetNumRows(0)
				}
			} else {
				trg := tensorfs.ValueType(allDir, cl.Name(), clv.DataType())
				trg.AppendRow(clv.RowTensor(clv.DimSize(0) - 1))
			}
		}
	}
}

// FieldValue holds the value of a field in a struct.
type FieldValue struct {
	Path          string
	Field         reflect.StructField
	Value, Parent reflect.Value
}

// StructValues returns a list of [FieldValue]s for fields of given struct,
// including any sub-fields, subject to filtering from the given should function
// which returns true for anything to include and false to exclude.
// You must pass a pointer to the object, so that the values are addressable.
func StructValues(obj any, should func(parent reflect.Value, field reflect.StructField, value reflect.Value) bool) []*FieldValue {
	var vals []*FieldValue
	val := reflect.ValueOf(obj).Elem()
	parName := ""
	WalkFields(val, should,
		func(parent reflect.Value, field reflect.StructField, value reflect.Value) {
			fkind := field.Type.Kind()
			fname := field.Name
			if val.Addr().Interface() == parent.Addr().Interface() { // top-level
				if fkind == reflect.Struct {
					parName = fname
					return
				}
			} else {
				fname = parName + "." + fname
			}
			sv := &FieldValue{Path: fname, Field: field, Value: value, Parent: parent}
			vals = append(vals, sv)
		})
	return vals
}

func WalkFields(parent reflect.Value, should func(parent reflect.Value, field reflect.StructField, value reflect.Value) bool, walk func(parent reflect.Value, field reflect.StructField, value reflect.Value)) {
	typ := parent.Type()
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if !field.IsExported() {
			continue
		}
		value := parent.Field(i)
		if !should(parent, field, value) {
			continue
		}
		if field.Type.Kind() == reflect.Struct {
			walk(parent, field, value)
			WalkFields(value, should, walk)
		} else {
			walk(parent, field, value)
		}
	}
}
