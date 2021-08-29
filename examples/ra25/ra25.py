#!/usr/local/bin/pyleabra

# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# pyleabra -i ra25.py 
# to run in gui interactive mode from the command line (or pyleabra, import ra25)
# see main function at the end for startup args

# to run this python version of the demo:
# * install gopy, currently in fork at https://github.com/goki/gopy
#   e.g., 'go get github.com/goki/gopy -u ./...' and then cd to that package
#   and do 'go install'
# * go to the python directory in this emergent repository, read README.md there, and 
#   type 'make' -- if that works, then type make install (may need sudo)
# * cd back here, and run 'pyemergent' which was installed into /usr/local/bin
# * then type 'import ra25' and this should run
# * you'll need various standard packages such as pandas, numpy, matplotlib, etc

# labra25ra runs a simple random-associator 5x5 = 25 four-layer leabra network

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, pygiv, pyparams, mat32

import importlib as il  #il.reload(ra25) -- doesn't seem to work for reasons unknown
import io, sys, getopt
from datetime import datetime, timezone

# import numpy as np
# import matplotlib
# matplotlib.use('SVG')
# import matplotlib.pyplot as plt
# plt.rcParams['svg.fonttype'] = 'none'  # essential for not rendering fonts as paths

# note: pandas, xarray or pytorch TensorDataSet can be used for input / output
# patterns and recording of "log" data for plotting.  However, the etable.Table
# has better GUI and API support, and handles tensor columns directly unlike
# pandas.  Support for easy migration between these is forthcoming.
# import pandas as pd

# this will become Sim later.. 
TheSim = 1

# LogPrec is precision for saving float values in logs
LogPrec = 4

# note: we cannot use methods for callbacks from Go -- must be separate functions
# so below are all the callbacks from the GUI toolbar actions

def InitCB(recv, send, sig, data):
    TheSim.Init()
    TheSim.UpdateClassView()
    TheSim.vp.SetNeedsFullRender()

def TrainCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.Train()

def StopCB(recv, send, sig, data):
    TheSim.Stop()

def StepTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TrainTrial()
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def StepEpochCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.TrainEpoch()

def StepRunCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.TrainRun()

def TestTrialCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.TestTrial(False)
        TheSim.IsRunning = False
        TheSim.UpdateClassView()
        TheSim.vp.SetNeedsFullRender()

def TestItemCB2(recv, send, sig, data):
    win = gi.Window(handle=recv)
    vp = win.WinViewport2D()
    dlg = gi.Dialog(handle=send)
    if sig != gi.DialogAccepted:
        return
    val = gi.StringPromptDialogValue(dlg)
    idxs = TheSim.TestEnv.Table.RowsByString("Name", val, True, True) # contains, ignoreCase
    if len(idxs) == 0:
        gi.PromptDialog(vp, gi.DlgOpts(Title="Name Not Found", Prompt="No patterns found containing: " + val), True, False, go.nil, go.nil)
    else:
        if not TheSim.IsRunning:
            TheSim.IsRunning = True
            print("testing index: %s" % idxs[0])
            TheSim.TestItem(idxs[0])
            TheSim.IsRunning = False
            vp.SetNeedsFullRender()

def TestItemCB(recv, send, sig, data):
    win = gi.Window(handle=recv)
    gi.StringPromptDialog(win.WinViewport2D(), "", "Test Item",
        gi.DlgOpts(Title="Test Item", Prompt="Enter the Name of a given input pattern to test (case insensitive, contains given string."), win, TestItemCB2)

def TestAllCB(recv, send, sig, data):
    if not TheSim.IsRunning:
        TheSim.IsRunning = True
        TheSim.ToolBar.UpdateActions()
        TheSim.RunTestAll()

def ResetRunLogCB(recv, send, sig, data):
    TheSim.RunLog.SetNumRows(0)
    TheSim.RunPlot.Update()

def NewRndSeedCB(recv, send, sig, data):
    TheSim.NewRndSeed()

def ReadmeCB(recv, send, sig, data):
    gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")

def FilterSSE(et, row):
    return etable.Table(handle=et).CellFloat("SSE", row) > 0 # include error trials    

def UpdtFuncNotRunning(act):
    act.SetActiveStateUpdt(not TheSim.IsRunning)
    
def UpdtFuncRunning(act):
    act.SetActiveStateUpdt(TheSim.IsRunning)

    
#####################################################    
#     Sim

class Sim(pygiv.ClassViewObj):
    """
    Sim encapsulates the entire simulation model, and we define all the
    functionality as methods on this struct.  This structure keeps all relevant
    state information organized and available without having to pass everything around
    as arguments to methods, and provides the core GUI interface (note the view tags
    for the fields which provide hints to how things should be displayed).
    """

    def __init__(self):
        super(Sim, self).__init__()
        self.Net = leabra.Network()
        self.SetTags("Net", 'view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"')
        self.Pats = etable.Table()
        self.SetTags("Pats", 'view:"no-inline" desc:"the training patterns to use"')
        self.TrnEpcLog = etable.Table()
        self.SetTags("TrnEpcLog", 'view:"no-inline" desc:"training epoch-level log data"')
        self.TstEpcLog = etable.Table()
        self.SetTags("TstEpcLog", 'view:"no-inline" desc:"testing epoch-level log data"')
        self.TstTrlLog = etable.Table()
        self.SetTags("TstTrlLog", 'view:"no-inline" desc:"testing trial-level log data"')
        self.TstErrLog = etable.Table()
        self.SetTags("TstErrLog", 'view:"no-inline" desc:"log of all test trials where errors were made"')
        self.TstErrStats = etable.Table()
        self.SetTags("TstErrStats", 'view:"no-inline" desc:"stats on test trials where errors were made"')
        self.TstCycLog = etable.Table()
        self.SetTags("TstCycLog", 'view:"no-inline" desc:"testing cycle-level log data"')
        self.RunLog = etable.Table()
        self.SetTags("RunLog", 'view:"no-inline" desc:"summary log of each run"')
        self.RunStats = etable.Table()
        self.SetTags("RunStats", 'view:"no-inline" desc:"aggregate stats on all runs"')
        self.Params = params.Sets()
        self.SetTags("Params", 'view:"no-inline" desc:"full collection of param sets"')
        self.ParamSet = str()
        self.SetTags("ParamSet", 'desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don\'t put spaces in ParamSet names!)"')
        self.Tag = str()
        self.SetTags("Tag", 'desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"')
        self.MaxRuns = int(10)
        self.SetTags("MaxRuns", 'desc:"maximum number of model runs to perform"')
        self.MaxEpcs = int(50)
        self.SetTags("MaxEpcs", 'desc:"maximum number of epochs to run per model run"')
        self.NZeroStop = int(5)
        self.SetTags("NZeroStop", 'desc:"if a positive number, training will stop after this many epochs with zero SSE"')
        self.TrainEnv = env.FixedTable()
        self.SetTags("TrainEnv", 'desc:"Training environment -- contains everything about iterating over input / output patterns over training"')
        self.TestEnv = env.FixedTable()
        self.SetTags("TestEnv", 'desc:"Testing environment -- manages iterating over testing"')
        self.Time = leabra.Time()
        self.SetTags("Time", 'desc:"leabra timing parameters and state"')
        self.ViewOn = True
        self.SetTags("ViewOn", 'desc:"whether to update the network view while running"')
        self.TrainUpdt = leabra.TimeScales.AlphaCycle
        self.SetTags("TrainUpdt", 'desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestUpdt = leabra.TimeScales.Cycle
        self.SetTags("TestUpdt", 'desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"')
        self.TestInterval = int(5)
        self.SetTags("TestInterval", 'desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"')
        self.LayStatNms = go.Slice_string(["Hidden1", "Hidden2", "Output"])
        self.SetTags("LayStatNms", 'desc:"names of layers to collect more detailed stats on (avg act, etc)"')

        # statistics: note use float64 as that is best for etable.Table
        self.TrlErr = float()
        self.SetTags("TrlErr", 'inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"')
        self.TrlSSE = float()
        self.SetTags("TrlSSE", 'inactive:"+" desc:"current trial\'s sum squared error"')
        self.TrlAvgSSE = float()
        self.SetTags("TrlAvgSSE", 'inactive:"+" desc:"current trial\'s average sum squared error"')
        self.TrlCosDiff = float()
        self.SetTags("TrlCosDiff", 'inactive:"+" desc:"current trial\'s cosine difference"')
        self.EpcSSE = float()
        self.SetTags("EpcSSE", 'inactive:"+" desc:"last epoch\'s total sum squared error"')
        self.EpcAvgSSE = float()
        self.SetTags("EpcAvgSSE", 'inactive:"+" desc:"last epoch\'s average sum squared error (average over trials, and over units within layer)"')
        self.EpcPctErr = float()
        self.SetTags("EpcPctErr", 'inactive:"+" desc:"last epoch\'s average TrlErr"')
        self.EpcPctCor = float()
        self.SetTags("EpcPctCor", 'inactive:"+" desc:"1 - last epoch\'s average TrlErr"')
        self.EpcCosDiff = float()
        self.SetTags("EpcCosDiff", 'inactive:"+" desc:"last epoch\'s average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"')
        self.EpcPerTrlMSec = float()
        self.SetTags("EpcPerTrlMSec", 'inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"')
        self.FirstZero = int()
        self.SetTags("FirstZero", 'inactive:"+" desc:"epoch at when SSE first went to zero"')
        self.NZero = int()
        self.SetTags("NZero", 'inactive:"+" desc:"number of epochs in a row with zero SSE"')

        # internal state - view:"-"
        self.SumErr = float()
        self.SetTags("SumErr", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumSSE = float()
        self.SetTags("SumSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumAvgSSE = float()
        self.SetTags("SumAvgSSE", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')
        self.SumCosDiff = float()
        self.SetTags("SumCosDiff", 'view:"-" inactive:"+" desc:"sum to increment as we go through epoch"')

        self.Win = 0
        self.SetTags("Win", 'view:"-" desc:"main GUI window"')
        self.NetView = 0
        self.SetTags("NetView", 'view:"-" desc:"the network viewer"')
        self.ToolBar = 0
        self.SetTags("ToolBar", 'view:"-" desc:"the master toolbar"')
        self.TrnEpcPlot = 0
        self.SetTags("TrnEpcPlot", 'view:"-" desc:"the training epoch plot"')
        self.TstEpcPlot = 0
        self.SetTags("TstEpcPlot", 'view:"-" desc:"the testing epoch plot"')
        self.TstTrlPlot = 0
        self.SetTags("TstTrlPlot", 'view:"-" desc:"the test-trial plot"')
        self.TstCycPlot = 0
        self.SetTags("TstCycPlot", 'view:"-" desc:"the test-cycle plot"')
        self.RunPlot = 0
        self.SetTags("RunPlot", 'view:"-" desc:"the run plot"')
        self.TrnEpcFile = 0
        self.SetTags("TrnEpcFile", 'view:"-" desc:"log file"')
        self.RunFile = 0
        self.SetTags("RunFile", 'view:"-" desc:"log file"')
        self.ValsTsrs = {}
        self.SetTags("ValsTsrs", 'view:"-" desc:"for holding layer values"')
        self.SaveWts = False
        self.SetTags("SaveWts", 'view:"-" desc:"for command-line run only, auto-save final weights after each run"')
        self.NoGui = False
        self.SetTags("NoGui", 'view:"-" desc:"if true, runing in no GUI mode"')
        self.LogSetParams = False
        self.SetTags("LogSetParams", 'view:"-" desc:"if true, print message for all params that are set"')
        self.IsRunning = False
        self.SetTags("IsRunning", 'view:"-" desc:"true if sim is running"')
        self.StopNow = False
        self.SetTags("StopNow", 'view:"-" desc:"flag to stop running"')
        self.NeedsNewRun = False
        self.SetTags("NeedsNewRun", 'view:"-" desc:"flag to initialize NewRun if last one finished"')
        self.RndSeed = int(1)
        self.SetTags("RndSeed", 'view:"-" desc:"the current random seed"')
        self.LastEpcTime = int()
        self.SetTags("LastEpcTime", 'view:"-" desc:"timer for last epoch"')
        self.vp  = 0 
        self.SetTags("vp", 'view:"-" desc:"viewport"')

    def InitParams(ss):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        ss.Params.OpenJSON("ra25_std.params")

    def Config(ss):
        """
        Config configures all the elements using the standard functions
        """
        ss.InitParams()
        ss.OpenPats()
        ss.ConfigEnv()
        ss.ConfigNet(ss.Net)
        ss.ConfigTrnEpcLog(ss.TrnEpcLog)
        ss.ConfigTstEpcLog(ss.TstEpcLog)
        ss.ConfigTstTrlLog(ss.TstTrlLog)
        ss.ConfigTstCycLog(ss.TstCycLog)
        ss.ConfigRunLog(ss.RunLog)

    def ConfigEnv(ss):
        if ss.MaxRuns == 0: # allow user override
            ss.MaxRuns = 10
        if ss.MaxEpcs == 0: # allow user override
            ss.MaxEpcs = 50
            ss.NZeroStop = 5

        ss.TrainEnv.Nm = "TrainEnv"
        ss.TrainEnv.Dsc = "training params and state"
        ss.TrainEnv.Table = etable.NewIdxView(ss.Pats)
        ss.TrainEnv.Validate()
        ss.TrainEnv.Run.Max = ss.MaxRuns # note: we are not setting epoch max -- do that manually

        ss.TestEnv.Nm = "TestEnv"
        ss.TestEnv.Dsc = "testing params and state"
        ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
        ss.TestEnv.Sequential = True
        ss.TestEnv.Validate()

        # note: to create a train / test split of pats, do this:
        # all := etable.NewIdxView(ss.Pats)
        # splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
        # ss.TrainEnv.Table = splits.Splits[0]
        # ss.TestEnv.Table = splits.Splits[1]

        ss.TrainEnv.Init(0)
        ss.TestEnv.Init(0)

    def ConfigNet(ss, net):
        net.InitName(net, "RA25")
        inp = net.AddLayer2D("Input", 5, 5, emer.Input)
        hid1 = net.AddLayer2D("Hidden1", 7, 7, emer.Hidden)
        hid2 = net.AddLayer4D("Hidden2", 2, 4, 3, 2, emer.Hidden)
        out = net.AddLayer2D("Output", 5, 5, emer.Target)

        # use this to position layers relative to each other
        # default is Above, YAlign = Front, XAlign = Center
        hid2.SetRelPos(relpos.Rel(Rel= relpos.RightOf, Other= "Hidden1", YAlign= relpos.Front, Space= 2))

        # note: see emergent/prjn module for all the options on how to connect
        # NewFull returns a new prjn.Full connectivity pattern
        full = prjn.NewFull()

        net.ConnectLayers(inp, hid1, full, emer.Forward)
        net.BidirConnectLayersPy(hid1, hid2, full)
        net.BidirConnectLayersPy(hid2, out, full)

        # note: can set these to do parallel threaded computation across multiple cpus
        # not worth it for this small of a model, but definitely helps for larger ones
        # if Thread {
        #   hid2.SetThread(1)
        #   out.SetThread(1)
        # }

        # note: if you wanted to change a layer type from e.g., Target to Compare, do this:
        # out.SetType(emer.Compare)
        # that would mean that the output layer doesn't reflect target values in plus phase
        # and thus removes error-driven learning -- but stats are still computed.

        net.Defaults()
        ss.SetParams("Network", ss.LogSetParams) # only set Network params
        net.Build()
        net.InitWts()

    def Init(ss):
        """
        Init restarts the run, and initializes everything, including network weights
        and resets the epoch log table
        """
        rand.Seed(ss.RndSeed)
        ss.ConfigEnv()

        ss.StopNow = False
        ss.SetParams("", ss.LogSetParams) # all sheets
        ss.NewRun()
        ss.UpdateView(True)

    def NewRndSeed(ss):
        """
        NewRndSeed gets a new random seed based on current time -- otherwise uses
        the same random seed for every run
        """
        ss.RndSeed = int(datetime.now(timezone.utc).timestamp())

    def Counters(ss, train):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        if train:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
        else:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)

    def UpdateView(ss, train):
        if ss.NetView != 0 and ss.NetView.IsVisible():
            ss.NetView.Record(ss.Counters(train))

            ss.NetView.GoUpdate()

    def AlphaCyc(ss, train):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters) of processing.
        External inputs must have already been applied prior to calling,
        using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
        If train is true, then learning DWt or WtFmDWt calls are made.
        Handles netview updating within scope of AlphaCycle
        """

        if ss.Win != 0:
            ss.Win.PollEvents() # this is essential for GUI responsiveness while running
        viewUpdt = ss.TrainUpdt.value
        if not train:
            viewUpdt = ss.TestUpdt.value

        # update prior weight changes at start, so any DWt values remain visible at end
        # you might want to do this less frequently to achieve a mini-batch update
        # in which case, move it out to the TrainTrial method where the relevant
        # counters are being dealt with.
        if train:
            ss.Net.WtFmDWt()

        ss.Net.AlphaCycInit(train)
        ss.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(ss.Time.CycPerQtr):
                ss.Net.Cycle(ss.Time)
                if not train:
                    ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
                ss.Time.CycleInc()
                if ss.ViewOn:
                    if viewUpdt == leabra.Cycle:
                        if cyc != ss.Time.CycPerQtr-1: # will be updated by quarter
                            ss.UpdateView(train)
                    if viewUpdt == leabra.FastSpike:
                        if (cyc+1)%10 == 0:
                            ss.UpdateView(train)
            ss.Net.QuarterFinal(ss.Time)
            ss.Time.QuarterInc()
            if ss.ViewOn:
                if viewUpdt <= leabra.Quarter:
                    ss.UpdateView(train)
                if viewUpdt == leabra.Phase:
                    if qtr >= 2:
                        ss.UpdateView(train)

        if train:
            ss.Net.DWt()
        if ss.ViewOn and viewUpdt == leabra.AlphaCycle:
            ss.UpdateView(train)
        if ss.TstCycPlot != 0 and not train:
            ss.TstCycPlot.GoUpdate() # make sure up-to-date at end


    def ApplyInputs(ss, en):
        """
        ApplyInputs applies input patterns from given envirbonment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """

        lays = ["Input", "Output"]
        for lnm in lays :
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            pats = en.State(ly.Nm)
            if pats != 0:
                ly.ApplyExt(pats)

    def TrainTrial(ss):
        """
        TrainTrial runs one trial of training using TrainEnv
        """
        if ss.NeedsNewRun:
            ss.NewRun()

        ss.TrainEnv.Step()

        # Key to query counters FIRST because current state is in NEXT epoch
        # if epoch counter has changed
        epc = env.CounterCur(ss.TrainEnv, env.Epoch)
        chg = env.CounterChg(ss.TrainEnv, env.Epoch)

        if chg:
            ss.LogTrnEpc(ss.TrnEpcLog)
            if ss.ViewOn and ss.TrainUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(True)
            if ss.TestInterval > 0 and epc%ss.TestInterval == 0: # note: epc is *next* so won't trigger first time
                ss.TestAll()
            if epc >= ss.MaxEpcs or (ss.NZeroStop > 0 and ss.NZero >= ss.NZeroStop):
                # done with training..
                ss.RunEnd()
                if ss.TrainEnv.Run.Incr(): # we are done!
                    ss.StopNow = True
                    return
                else:
                    ss.NeedsNewRun = True
                    return

        ss.ApplyInputs(ss.TrainEnv)
        ss.AlphaCyc(True)   # train
        ss.TrialStats(True) # accumulate

    def RunEnd(ss):
        """
        RunEnd is called at the end of a run -- save weights, record final log, etc here
        """
        ss.LogRun(ss.RunLog)
        if ss.SaveWts:
            fnm = ss.WeightsFileName()
            print("Saving Weights to: %s\n" % fnm)
            ss.Net.SaveWtsJSON(gi.FileName(fnm))

    def NewRun(ss):
        """
        NewRun intializes a new run of the model, using the TrainEnv.Run counter
        for the new run value
        """
        run = ss.TrainEnv.Run.Cur
        ss.TrainEnv.Init(run)
        ss.TestEnv.Init(run)
        ss.Time.Reset()
        ss.Net.InitWts()
        ss.InitStats()
        ss.TrnEpcLog.SetNumRows(0)
        ss.TstEpcLog.SetNumRows(0)
        ss.NeedsNewRun = False

    def InitStats(ss):
        """
        InitStats initializes all the statistics, especially important for the
        cumulative epoch stats -- called at start of new run
        """

        ss.SumErr = 0
        ss.SumSSE = 0
        ss.SumAvgSSE = 0
        ss.SumCosDiff = 0
        ss.FirstZero = -1
        ss.NZero = 0

        ss.TrlErr = 0
        ss.TrlSSE = 0
        ss.TrlAvgSSE = 0
        ss.EpcSSE = 0
        ss.EpcAvgSSE = 0
        ss.EpcPctErr = 0
        ss.EpcCosDiff = 0

    def TrialStats(ss, accum):
        """
        TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
        accum is true.  Note that we're accumulating stats here on the Sim side so the
        core algorithm side remains as simple as possible, and doesn't need to worry about
        different time-scales over which stats could be accumulated etc.
        You can also aggregate directly from log data, as is done for testing stats
        """
        out = leabra.Layer(ss.Net.LayerByName("Output"))
        ss.TrlCosDiff = float(out.CosDiff.Cos)
        ss.TrlSSE = out.SSE(0.5) # 0.5 = per-unit tolerance -- right side of .5
        ss.TrlAvgSSE = ss.TrlSSE / len(out.Neurons)
        if ss.TrlSSE > 0:
            ss.TrlErr = 1
        else:
            ss.TrlErr = 0
        if accum:
            ss.SumErr += ss.TrlErr
            ss.SumSSE += ss.TrlSSE
            ss.SumAvgSSE += ss.TrlAvgSSE
            ss.SumCosDiff += ss.TrlCosDiff

    def TrainEpoch(ss):
        """
        TrainEpoch runs training trials for remainder of this epoch
        """
        ss.StopNow = False
        curEpc = ss.TrainEnv.Epoch.Cur
        while True:
            ss.TrainTrial()
            if ss.StopNow or ss.TrainEnv.Epoch.Cur != curEpc:
                break
        ss.Stopped()

    def TrainRun(ss):
        """
        TrainRun runs training trials for remainder of run
        """
        ss.StopNow = False
        curRun = ss.TrainEnv.Run.Cur
        while True:
            ss.TrainTrial()
            if ss.StopNow or ss.TrainEnv.Run.Cur != curRun:
                break
        ss.Stopped()

    def Train(ss):
        """
        Train runs the full training from this point onward
        """
        ss.StopNow = False
        while True:
            ss.TrainTrial()
            if ss.StopNow:
                break
        ss.Stopped()

    def Stop(ss):
        """
        Stop tells the sim to stop running
        """
        ss.StopNow = True

    def Stopped(ss):
        """
        Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
        """
        ss.IsRunning = False
        if ss.Win != 0:
            vp = ss.Win.WinViewport2D()
            if ss.ToolBar != 0:
                ss.ToolBar.UpdateActions()
            vp.SetNeedsFullRender()
            ss.UpdateClassView()

    def SaveWeights(ss, filename):
        """
        SaveWeights saves the network weights -- when called with giv.CallMethod
        it will auto-prompt for filename
        """
        ss.Net.SaveWtsJSON(filename)

    def TestTrial(ss, returnOnChg):
        """
        TestTrial runs one trial of testing -- always sequentially presented inputs
        """
        ss.TestEnv.Step()

        chg = env.CounterChg(ss.TestEnv, env.Epoch)
        if chg:
            if ss.ViewOn and ss.TestUpdt.value > leabra.AlphaCycle:
                ss.UpdateView(False)
            ss.LogTstEpc(ss.TstEpcLog)
            if returnOnChg:
                return

        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.LogTstTrl(ss.TstTrlLog)

    def TestItem(ss, idx):
        """
        TestItem tests given item which is at given index in test item list
        """
        cur = ss.TestEnv.Trial.Cur
        ss.TestEnv.Trial.Cur = idx
        ss.TestEnv.SetTrialName()
        ss.ApplyInputs(ss.TestEnv)
        ss.AlphaCyc(False)
        ss.TrialStats(False)
        ss.TestEnv.Trial.Cur = cur

    def TestAll(ss):
        """
        TestAll runs through the full set of testing items
        """
        ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
        while True:
            ss.TestTrial(True)
            chg = env.CounterChg(ss.TestEnv, env.Epoch)
            if chg or ss.StopNow:
                break

    def RunTestAll(ss):
        """
        RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
        """
        ss.StopNow = False
        ss.TestAll()
        ss.Stopped()

    def ParamsName(ss):
        """
        ParamsName returns name of current set of parameters
        """
        if ss.ParamSet == "":
            return "Base"
        return ss.ParamSet

    def SetParams(ss, sheet, setMsg):
        """
        SetParams sets the params for "Base" and then current ParamSet.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        if sheet == "":
            ss.Params.ValidateSheets(go.Slice_string(["Network", "Sim"]))
        ss.SetParamsSet("Base", sheet, setMsg)
        if ss.ParamSet != "" and ss.ParamSet != "Base":
            sps = ss.ParamSet.split()
            for ps in sps:
                ss.SetParamsSet(ps, sheet, setMsg)

    def SetParamsSet(ss, setNm, sheet, setMsg):
        """
        SetParamsSet sets the params for given params.Set name.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        pset = ss.Params.SetByNameTry(setNm)
        if sheet == "" or sheet == "Network":
            if "Network" in pset.Sheets:
                netp = pset.SheetByNameTry("Network")
                ss.Net.ApplyParams(netp, setMsg)

        if sheet == "" or sheet == "Sim":
            if "Sim" in pset.Sheets:
                simp= pset.SheetByNameTry("Sim")
                pyparams.ApplyParams(ss, simp, setMsg)

    def ConfigPats(ss):
        dt = ss.Pats
        dt.SetMetaData("name", "TrainPats")
        dt.SetMetaData("desc", "Training patterns")
        sch = etable.Schema(
            [etable.Column("Name", etensor.STRING, go.nil, go.nil),
            etable.Column("Input", etensor.FLOAT32, go.Slice_int([5, 5]), go.Slice_string(["Y", "X"])),
            etable.Column("Output", etensor.FLOAT32, go.Slice_int([5, 5]), go.Slice_string(["Y", "X"]))]
        )
        dt.SetFromSchema(sch, 25)

        patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
        patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
        dt.SaveCSV("random_5x5_25_gen.tsv", etable.Tab, etable.Headers)

    def OpenPats(ss):
        dt = ss.Pats
        dt.SetMetaData("name", "TrainPats")
        dt.SetMetaData("desc", "Training patterns")
        dt.OpenCSV("random_5x5_25.tsv", etable.Tab)

    def ValsTsr(ss, name):
        """
        ValsTsr gets value tensor of given name, creating if not yet made
        """
        if name in ss.ValsTsrs:
            return ss.ValsTsrs[name]
        tsr = etensor.Float32()
        ss.ValsTsrs[name] = tsr
        return tsr

    def RunName(ss):
        """
        RunName returns a name for this run that combines Tag and Params -- add this to
        any file names that are saved.
        """
        if ss.Tag != "":
            return ss.Tag + "_" + ss.ParamsName()
        else:
            return ss.ParamsName()

    def RunEpochName(ss, run, epc):
        """
        RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
        for using in weights file names.  Uses 3, 5 digits for each.
        """
        return "%03d_%05d" % (run, epc)

    def WeightsFileName(ss):
        """
        WeightsFileName returns default current weights file name
        """
        return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"

    def LogFileName(ss, lognm):
        """
        LogFileName returns default log file name
        """
        return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"

    def LogTrnEpc(ss, dt):
        """
        LogTrnEpc adds data from current epoch to the TrnEpcLog table.
        computes epoch averages prior to logging.
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        epc = ss.TrainEnv.Epoch.Prv
        nt = float(len(ss.TrainEnv.Order))

        ss.EpcSSE = ss.SumSSE / nt
        ss.SumSSE = 0
        ss.EpcAvgSSE = ss.SumAvgSSE / nt
        ss.SumAvgSSE = 0
        ss.EpcPctErr = float(ss.SumErr) / nt
        ss.SumErr = 0
        ss.EpcPctCor = 1 - ss.EpcPctErr
        ss.EpcCosDiff = ss.SumCosDiff / nt
        ss.SumCosDiff = 0
        if ss.FirstZero < 0 and ss.EpcPctErr == 0:
            ss.FirstZero = epc
        if ss.EpcPctErr == 0:
            ss.NZero += 1
        else:
            ss.NZero = 0

        # if ss.LastEpcTime.IsZero():
        #     ss.EpcPerTrlMSec = 0
        # else:
        #     iv = time.Now().Sub(ss.LastEpcTime)
        #     ss.EpcPerTrlMSec = float(iv) / (nt * float(time.Millisecond))
        # ss.LastEpcTime = time.Now()

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("SSE", row, ss.EpcSSE)
        dt.SetCellFloat("AvgSSE", row, ss.EpcAvgSSE)
        dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
        dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
        dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
        # dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

        for lnm in ss.LayStatNms:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+"_ActAvg", row, float(ly.Pool(0).ActAvg.ActPAvgEff))

        if ss.TrnEpcPlot != 0:
            ss.TrnEpcPlot.GoUpdate()
        if ss.TrnEpcFile != 0:
            if ss.TrainEnv.Run.Cur == 0 and epc == 0:
                dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
            dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)

    def ConfigTrnEpcLog(ss, dt):
        dt.SetMetaData("name", "TrnEpcLog")
        dt.SetMetaData("desc", "Record of performance over epochs of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PerTrlMSec", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.LayStatNms:
            sch.append(etable.Column(lnm + "_ActAvg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, 0)

    def ConfigTrnEpcPlot(ss, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)

        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

        for lnm in ss.LayStatNms:
            plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
        return plt

    def LogTstTrl(ss, dt):
        """
        LogTstTrl adds data from current trial to the TstTrlLog table.
        log always contains number of testing items
        """
        epc = ss.TrainEnv.Epoch.Prv
        inp = leabra.Layer(ss.Net.LayerByName("Input"))
        out = leabra.Layer(ss.Net.LayerByName("Output"))

        trl = ss.TestEnv.Trial.Cur
        row = trl

        if dt.Rows <= row:
            dt.SetNumRows(row + 1)

        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("Trial", row, float(trl))
        dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
        dt.SetCellFloat("Err", row, ss.TrlErr)
        dt.SetCellFloat("SSE", row, ss.TrlSSE)
        dt.SetCellFloat("AvgSSE", row, ss.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

        for lnm in ss.LayStatNms:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float(ly.Pool(0).ActM.Avg))
        ivt = ss.ValsTsr("Input")
        ovt = ss.ValsTsr("Output")
        inp.UnitValsTensor(ivt, "Act")
        dt.SetCellTensor("InAct", row, ivt)
        out.UnitValsTensor(ovt, "ActM")
        dt.SetCellTensor("OutActM", row, ovt)
        out.UnitValsTensor(ovt, "ActP")
        dt.SetCellTensor("OutActP", row, ovt)

        if ss.TstTrlPlot != 0:
            ss.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(ss, dt):
        inp = leabra.Layer(ss.Net.LayerByName("Input"))
        out = leabra.Layer(ss.Net.LayerByName("Output"))

        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        nt = ss.TestEnv.Table.Len() # number in view
        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("Trial", etensor.INT64, go.nil, go.nil),
            etable.Column("TrialName", etensor.STRING, go.nil, go.nil),
            etable.Column("Err", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        for lnm in ss.LayStatNms :
            sch.append(etable.Column(lnm + " ActM.Avg", etensor.FLOAT64, go.nil, go.nil))
            
        sch.append(etable.Column("InAct", etensor.FLOAT64, inp.Shp.Shp, go.nil))
        sch.append(etable.Column("OutActM", etensor.FLOAT64, out.Shp.Shp, go.nil))
        sch.append(etable.Column("OutActP", etensor.FLOAT64, out.Shp.Shp, go.nil))
        dt.SetFromSchema(sch, nt)

    def ConfigTstTrlPlot(ss, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

        for lnm in ss.LayStatNms:
            plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)

        plt.SetColParams("InAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("OutActM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("OutActP", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogTstEpc(ss, dt):
        row = dt.Rows
        dt.SetNumRows(row + 1)

        trl = ss.TstTrlLog
        tix = etable.NewIdxView(trl)
        epc = ss.TrainEnv.Epoch.Prv # ?

        # note: this shows how to use agg methods to compute summary data from another
        # data table, instead of incrementing on the Sim
        dt.SetCellFloat("Run", row, float(ss.TrainEnv.Run.Cur))
        dt.SetCellFloat("Epoch", row, float(epc))
        dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
        dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

        trlix = etable.NewIdxView(trl)
        trlix.Filter(FilterSSE) # requires separate function

        ss.TstErrLog = trlix.NewTable()

        allsp = split.All(trlix)
        split.Agg(allsp, "SSE", agg.AggSum)
        split.Agg(allsp, "AvgSSE", agg.AggMean)
        split.Agg(allsp, "InAct", agg.AggMean)
        split.Agg(allsp, "OutActM", agg.AggMean)
        split.Agg(allsp, "OutActP", agg.AggMean)

        ss.TstErrStats = allsp.AggsToTable(etable.AddAggName)

        # note: essential to use Go version of update when called from another goroutine
        if ss.TstEpcPlot != 0:
            ss.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(ss, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Epoch", etensor.INT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigTstEpcPlot(ss, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot
        plt.SetColParams("PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) # default plot
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def LogTstCyc(ss, dt, cyc):
        """
        LogTstCyc adds data from current trial to the TstCycLog table.
        log just has 100 cycles, is overwritten
        """
        if dt.Rows <= cyc:
            dt.SetNumRows(cyc + 1)

        dt.SetCellFloat("Cycle", cyc, float(cyc))
        for lnm in ss.LayStatNms:
            ly = leabra.Layer(ss.Net.LayerByName(lnm))
            dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc,  float(ly.Pool(0).Inhib.Ge.Avg))
            dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float(ly.Pool(0).Inhib.Act.Avg))

        if ss.TstCycPlot != 0 and cyc%10 == 0: # too slow to do every cyc
            # note: essential to use Go version of update when called from another goroutine
            ss.TstCycPlot.GoUpdate()

    def ConfigTstCycLog(ss, dt):
        dt.SetMetaData("name", "TstCycLog")
        dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        np = 100 # max cycles
        sch = etable.Schema(
            [etable.Column("Cycle", etensor.INT64, go.nil, go.nil)]
        )
        for lnm in ss.LayStatNms:
            sch.append(etable.Column(lnm + " Ge.Avg", etensor.FLOAT64, go.nil, go.nil))
            sch.append(etable.Column(lnm + " Act.Avg", etensor.FLOAT64, go.nil, go.nil))
        dt.SetFromSchema(sch, np)

    def ConfigTstCycPlot(ss, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Test Cycle Plot"
        plt.Params.XAxisCol = "Cycle"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        for lnm in ss.LayStatNms:
            plt.SetColParams(lnm+" Ge.Avg", True, True, 0, True, .5)
            plt.SetColParams(lnm+" Act.Avg", True, True, 0, True, .5)
        return plt

    def LogRun(ss, dt):
        """
        LogRun adds data from current run to the RunLog table.
        """
        epclog = ss.TrnEpcLog
        epcix = etable.NewIdxView(epclog)
        if epcix.Len() == 0:
            return
        
        run = ss.TrainEnv.Run.Cur # this is NOT triggered by increment yet -- use Cur
        row = dt.Rows
        dt.SetNumRows(row + 1)

        # compute mean over last N epochs for run level
        nlast = 5
        if nlast > epcix.Len()-1:
            nlast = epcix.Len() - 1
        epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

        params = ss.RunName() # includes tag

        dt.SetCellFloat("Run", row, float(run))
        dt.SetCellString("Params", row, params)
        dt.SetCellFloat("FirstZero", row, float(ss.FirstZero))
        dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
        dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

        runix = etable.NewIdxView(dt)
        spl = split.GroupBy(runix, go.Slice_string(["Params"]))
        split.Desc(spl, "FirstZero")
        split.Desc(spl, "PctCor")
        ss.RunStats = spl.AggsToTable(etable.AddAggName)

        # note: essential to use Go version of update when called from another goroutine
        if ss.RunPlot != 0:
            ss.RunPlot.GoUpdate()
        if ss.RunFile != 0:
            if row == 0:
                dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
            dt.WriteCSVRow(ss.RunFile, row, etable.Tab)

    def ConfigRunLog(ss, dt):
        dt.SetMetaData("name", "RunLog")
        dt.SetMetaData("desc", "Record of performance at end of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))

        sch = etable.Schema(
            [etable.Column("Run", etensor.INT64, go.nil, go.nil),
            etable.Column("Params", etensor.STRING, go.nil, go.nil),
            etable.Column("FirstZero", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("SSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("AvgSSE", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctErr", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("PctCor", etensor.FLOAT64, go.nil, go.nil),
            etable.Column("CosDiff", etensor.FLOAT64, go.nil, go.nil)]
        )
        dt.SetFromSchema(sch, 0)

    def ConfigRunPlot(ss, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) # default plot
        plt.SetColParams("SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
        plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
        return plt

    def ConfigGui(ss):
        """
        ConfigGui configures the GoGi gui interface for this simulation,
        """
        width = 1600
        height = 1200

        gi.SetAppName("ra25")
        gi.SetAppAbout('This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>')

        win = gi.NewMainWindow("ra25", "Leabra Random Associator", width, height)
        ss.Win = win

        vp = win.WinViewport2D()
        ss.vp = vp
        updt = vp.UpdateStart()

        mfr = win.SetMainFrame()

        tbar = gi.AddNewToolBar(mfr, "tbar")
        tbar.SetStretchMaxWidth()
        ss.ToolBar = tbar

        split = gi.AddNewSplitView(mfr, "split")
        split.Dim = mat32.X
        split.SetStretchMax()

        cv = ss.NewClassView("sv")
        cv.AddFrame(split)
        cv.Config()

        tv = gi.AddNewTabView(split, "tv")

        nv = netview.NetView()
        tv.AddTab(nv, "NetView")
        nv.Var = "Act"
        # nv.Params.ColorMap = "Jet" // default is ColdHot
        # which fares pretty well in terms of discussion here:
        # https://matplotlib.org/tutorials/colors/colormaps.html
        nv.SetNet(ss.Net)
        ss.NetView = nv

        nv.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) # more "head on" than default which is more "top down"
        nv.Scene().Camera.LookAt(mat32.Vec3(0, 0, 0), mat32.Vec3(0, 1, 0))

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TrnEpcPlot")
        ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstTrlPlot")
        ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstCycPlot")
        ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstEpcPlot")
        ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

        plt = eplot.Plot2D()
        tv.AddTab(plt, "RunPlot")
        ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

        split.SetSplitsList(go.Slice_float32([.2, .8]))

        recv = win.This()
        
        tbar.AddAction(gi.ActOpts(Label="Init", Icon="update", Tooltip="Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc=UpdtFuncNotRunning), recv, InitCB)

        tbar.AddAction(gi.ActOpts(Label="Train", Icon="run", Tooltip="Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.", UpdateFunc=UpdtFuncNotRunning), recv, TrainCB)
        
        tbar.AddAction(gi.ActOpts(Label="Stop", Icon="stop", Tooltip="Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc=UpdtFuncRunning), recv, StopCB)
        
        tbar.AddAction(gi.ActOpts(Label="Step Trial", Icon="step-fwd", Tooltip="Advances one training trial at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Step Epoch", Icon="fast-fwd", Tooltip="Advances one epoch (complete set of training patterns) at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepEpochCB)

        tbar.AddAction(gi.ActOpts(Label="Step Run", Icon="fast-fwd", Tooltip="Advances one full training Run at a time.", UpdateFunc=UpdtFuncNotRunning), recv, StepRunCB)
        
        tbar.AddSeparator("test")
        
        tbar.AddAction(gi.ActOpts(Label="Test Trial", Icon="step-fwd", Tooltip="Runs the next testing trial.", UpdateFunc=UpdtFuncNotRunning), recv, TestTrialCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test Item", Icon="step-fwd", Tooltip="Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc=UpdtFuncNotRunning), recv, TestItemCB)
        
        tbar.AddAction(gi.ActOpts(Label="Test All", Icon="fast-fwd", Tooltip="Tests all of the testing trials.", UpdateFunc=UpdtFuncNotRunning), recv, TestAllCB)

        tbar.AddSeparator("log")
        
        tbar.AddAction(gi.ActOpts(Label="Reset RunLog", Icon="reset", Tooltip="Resets the accumulated log of all Runs, which are tagged with the ParamSet used"), recv, ResetRunLogCB)

        tbar.AddSeparator("misc")
        
        tbar.AddAction(gi.ActOpts(Label="New Seed", Icon="new", Tooltip="Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."), recv, NewRndSeedCB)

        tbar.AddAction(gi.ActOpts(Label="README", Icon="file-markdown", Tooltip="Opens your browser on the README file that contains instructions for how to run this model."), recv, ReadmeCB)

        # main menu
        appnm = gi.AppName()
        mmen = win.MainMenu
        mmen.ConfigMenus(go.Slice_string([appnm, "File", "Edit", "Window"]))

        amen = gi.Action(win.MainMenu.ChildByName(appnm, 0))
        amen.Menu.AddAppMenu(win)

        emen = gi.Action(win.MainMenu.ChildByName("Edit", 1))
        emen.Menu.AddCopyCutPaste(win)

        # note: Command in shortcuts is automatically translated into Control for
        # Linux, Windows or Meta for MacOS
        # fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
        # fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
        #   win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
        #       FileViewOpenSVG(vp)
        #   })
        # fmen.Menu.AddSeparator("csep")
        # fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
        #   win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
        #       win.Close()
        #   })

        win.MainMenuUpdated()
        vp.UpdateEndNoSig(updt)
        win.GoStartEventLoop()


# TheSim is the overall state for this simulation
TheSim = Sim()

def usage():
    print(sys.argv[0] + " --params=<param set> --tag=<extra tag> --setparams --wts --epclog=0 --runlog=0 --nogui")
    print("\t pyleabra -i %s to run in interactive, gui mode" % sys.argv[0])
    print("\t --params=<param set> additional params to apply on top of Base (name must be in loaded Params")
    print("\t --tag=<extra tag>    tag is appended to file names to uniquely identify this run") 
    print("\t --runs=<n>           number of runs to do")
    print("\t --setparams          show the parameter values that are set")
    print("\t --wts                save final trained weights after every run")
    print("\t --epclog=0/False     turn off save training epoch log data to file named by param set, tag")
    print("\t --runlog=0/False     turn off save run log data to file named by param set, tag")
    print("\t --nogui              if no other args needed, this prevents running under the gui")

def main(argv):
    TheSim.Config()

    # print("n args: %d" % len(argv))
    TheSim.NoGui = len(argv) > 1
    saveEpcLog = True
    saveRunLog = True
        
    try:
        opts, args = getopt.getopt(argv,"h:",["params=","tag=","runs=","setparams","wts","epclog=","runlog=","nogui"])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        # print("opt: %s  arg: %s" % (opt, arg))
        if opt == '-h':
            usage()
            sys.exit()
        elif opt == "--tag":
            TheSim.Tag = arg
        elif opt == "--runs":
            TheSim.MaxRuns = int(arg)
            print("Running %d runs" % TheSim.MaxRuns)
        elif opt == "--setparams":
            TheSim.LogSetParams = True
        elif opt == "--wts":
            TheSim.SaveWts = True
            print("Saving final weights per run")
        elif opt == "--epclog":
            if arg.lower() == "false" or arg == "0":
                saveEpcLog = False
        elif opt == "--runlog":
            if arg.lower() == "false" or arg == "0":
                saveRunLog = False
        elif opt == "--nogui":
            TheSim.NoGui = True

    TheSim.Init()
            
    if TheSim.NoGui:
        if saveEpcLog:
            fnm = TheSim.LogFileName("epc") 
            print("Saving epoch log to: %s" % fnm)
            TheSim.TrnEpcFile = efile.Create(fnm)
    
        if saveRunLog:
            fnm = TheSim.LogFileName("run") 
            print("Saving run log to: %s" % fnm)
            TheSim.RunFile = efile.Create(fnm)
            
        TheSim.Train()
        sys.exit(0)
    else:
        TheSim.ConfigGui()
        print("Note: run pyleabra -i ra25.py to run in interactive mode, or just pyleabra, then 'import ra25'")
        print("for non-gui background running, here are the args:")
        usage()
        import code
        code.interact(local=locals())

main(sys.argv[1:])

