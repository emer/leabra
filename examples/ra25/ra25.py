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

from leabra import go, leabra, emer, relpos, eplot, env, agg, patgen, prjn, etable, efile, split, etensor, params, netview, rand, erand, gi, giv, epygiv

import importlib as il  #il.reload(ra25) -- doesn't seem to work for reasons unknown
import io, sys, getopt
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

# use this for e.g., etable.Column construction args where nil would be passed
nilInts = go.Slice_int()

# use this for e.g., etable.Column construction args where nil would be passed
nilStrs = go.Slice_string()

# LogPrec is precision for saving float values in logs
LogPrec = 4

# note: we cannot use methods for callbacks from Go -- must be separate functions
# so below are all the callbacks from the GUI toolbar actions

def InitCB(recv, send, sig, data):
    TheSim.Init()
    TheSim.ClassView.Update()
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
        TheSim.ClassView.Update()
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
        TheSim.TestTrial()
        TheSim.IsRunning = False
        TheSim.ClassView.Update()
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

class Sim(object):
    """
    Sim encapsulates the entire simulation model, and we define all the
    functionality as methods on this struct.  This structure keeps all relevant
    state information organized and available without having to pass everything around
    as arguments to methods, and provides the core GUI interface (note the view tags
    for the fields which provide hints to how things should be displayed).
    """
    def __init__(self):
        self.Net = leabra.Network()
        self.Pats     = etable.Table()
        self.TrnEpcLog   = etable.Table()
        self.TstEpcLog   = etable.Table()
        self.TstTrlLog   = etable.Table()
        self.TstErrLog   = etable.Table()
        self.TstErrStats = etable.Table()
        self.TstCycLog   = etable.Table()
        self.RunLog      = etable.Table()
        self.RunStats    = etable.Table()
        self.Params     = params.Sets()
        self.ParamSet = ""
        self.Tag      = ""
        self.MaxRuns  = 10
        self.MaxEpcs  = 50
        self.TrainEnv = env.FixedTable()
        self.TestEnv  = env.FixedTable()
        self.Time     = leabra.Time()
        self.ViewOn   = True
        self.TrainUpdt = leabra.AlphaCycle
        self.TestUpdt = leabra.Cycle
        self.TestInterval = 5
        
        # statistics
        self.TrlSSE     = 0.0
        self.TrlAvgSSE  = 0.0
        self.TrlCosDiff = 0.0
        self.EpcSSE     = 0.0
        self.EpcAvgSSE  = 0.0
        self.EpcPctErr  = 0.0
        self.EpcPctCor  = 0.0
        self.EpcCosDiff = 0.0
        self.FirstZero  = -1
        
        # internal state - view:"-"
        self.SumSSE     = 0.0
        self.SumAvgSSE  = 0.0
        self.SumCosDiff = 0.0
        self.CntErr     = 0.0
        self.Win        = 0
        self.vp         = 0
        self.ToolBar    = 0
        self.NetView    = 0
        self.TrnEpcPlot = 0
        self.TstEpcPlot = 0
        self.TstTrlPlot = 0
        self.TstCycPlot = 0
        self.RunPlot    = 0
        self.TrnEpcFile = 0
        self.RunFile    = 0
        self.InputValsTsr = 0
        self.OutputValsTsr = 0
        self.SaveWts    = False
        self.NoGui        = False
        self.LogSetParams = False # True
        self.IsRunning    = False
        self.StopNow    = False
        self.RndSeed    = 0
        
        # ClassView tags for controlling display of fields
        self.Tags = {
            'TrlSSE': 'inactive:"+"',
            'TrlAvgSSE': 'inactive:"+"',
            'TrlCosDiff': 'inactive:"+"',
            'EpcSSE': 'inactive:"+"',
            'EpcAvgSSE': 'inactive:"+"',
            'EpcPctErr': 'inactive:"+"',
            'EpcPctCor': 'inactive:"+"',
            'EpcCosDiff': 'inactive:"+"',
            'FirstZero': 'inactive:"+"',
            'SumSSE': 'view:"-"',
            'SumAvgSSE': 'view:"-"',
            'SumCosDiff': 'view:"-"',
            'CntErr': 'view:"-"',
            'Win': 'view:"-"',
            'vp': 'view:"-"',
            'ToolBar': 'view:"-"',
            'NetView': 'view:"-"',
            'TrnEpcPlot': 'view:"-"',
            'TstEpcPlot': 'view:"-"',
            'TstTrlPlot': 'view:"-"',
            'TstCycPlot': 'view:"-"',
            'RunPlot': 'view:"-"',
            'TrnEpcFile': 'view:"-"',
            'RunFile': 'view:"-"',
            'InputValsTsr': 'view:"-"',
            'OutputValsTsr': 'view:"-"',
            'SaveWts': 'view:"-"',
            'NoGui': 'view:"-"',
            'LogSetParams': 'view:"-"',
            'IsRunning': 'view:"-"',
            'StopNow': 'view:"-"',
            'RndSeed': 'view:"-"',
            'ClassView': 'view:"-"',
            'Tags': 'view:"-"',
        }


    def InitParams(self):
        """
        Sets the default set of parameters -- Base is always applied, and others can be optionally
        selected to apply on top of that
        """
        self.Params.OpenJSON("ra25_std.params")

        # todo: the following expression SHOULD produce the same results but it ends up
        # adding the items in a random order relative to what is shown here -- each time
        # the order is different.  very strange
        # pars = params.Set(Name="Base", Desc="these are the best params", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="Prjn", Desc="norm and momentum on works better, but wt bal is not better for smaller nets",
        #                 Params=params.Params({
        #                     "Prjn.Learn.Norm.On":     "true",
        #                     "Prjn.Learn.Momentum.On": "true",
        #                     "Prjn.Learn.WtBal.On":    "false",
        #                 }).handle),
        #             params.Sel(Sel="Layer", Desc="using default 1.8 inhib for all of network -- can explore",
        #                 Params=params.Params({
        #                     "Layer.Inhib.Layer.Gi": "1.8",
        #                 }).handle),
        #             params.Sel(Sel="#Output", Desc="output definitely needs lower inhib -- true for smaller layers in general",
        #                 Params=params.Params({
        #                     "Layer.Inhib.Layer.Gi": "1.4",
        #                 }).handle),
        #             params.Sel(Sel=".Back", Desc="top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
        #                 Params=params.Params({
        #                     "Prjn.WtScale.Rel": "0.2",
        #                 }).handle),
        #             }).handle,
        #         "Sim": params.Sheet({
        #             params.Sel(Sel="Sim", Desc="best params always finish in this time",
        #                 Params=params.Params({
        #                     "Sim.MaxEpcs": "50",
        #                 }).handle),
        #             }).handle,
        #     }).handle),
        # params.Set(Name="DefaultInhib", Desc="output uses default inhib instead of lower", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="#Output", Desc="go back to default",
        #                 Params=params.Params({
        #                     "Layer.Inhib.Layer.Gi": "1.8",
        #                    }).handle),
        #                 }).handle,
        #         "Sim": params.Sheet({
        #             params.Sel(Sel="Sim", Desc="takes longer -- generally doesn't finish..",
        #                 Params=params.Params({
        #                     "Sim.MaxEpcs": "100",
        #                }).handle),
        #             }).handle,
        #      }).handle),
        # params.Set(Name="NoMomentum", Desc="no momentum or normalization", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="Prjn", Desc="no norm or momentum",
        #                 Params=params.Params({
        #                     "Prjn.Learn.Norm.On":     "false",
        #                     "Prjn.Learn.Momentum.On": "false",
        #                 }).handle),
        #             }).handle,
        #         }).handle),
        # params.Set(Name="WtBalOn", Desc="try with weight bal on", Sheets=params.Sheets({
        #         "Network": params.Sheet({
        #             params.Sel(Sel="Prjn", Desc="weight bal on",
        #                Params=params.Params({
        #                    "Prjn.Learn.WtBal.On": "true",
        #                }).handle),
        #            }).handle,
        #        }).handle),
        # })


    ######################################
    #   Configs

    def Config(self):
        """Config configures all the elements using the standard functions"""
        self.InitParams()
        # self.OpenPats()
        self.ConfigPats()
        self.ConfigEnv()
        self.ConfigNet(self.Net)
        self.ConfigTrnEpcLog(self.TrnEpcLog)
        self.ConfigTstEpcLog(self.TstEpcLog)
        self.ConfigTstTrlLog(self.TstTrlLog)
        self.ConfigTstCycLog(self.TstCycLog)
        self.ConfigRunLog(self.RunLog)

    def ConfigEnv(self): 
        if self.MaxRuns == 0: # allow user override
            self.MaxRuns = 10
        if self.MaxEpcs == 0: # allow user override
            self.MaxEpcs = 50
        
        self.TrainEnv.Nm = "TrainEnv"
        self.TrainEnv.Dsc = "training params and state"
        self.TrainEnv.Table = etable.NewIdxView(self.Pats)
        self.TrainEnv.Validate()
        self.TrainEnv.Run.Max = self.MaxRuns # note: we are not setting epoch max -- do that manually
        
        self.TestEnv.Nm = "TestEnv"
        self.TestEnv.Dsc = "testing params and state"
        self.TestEnv.Table = etable.NewIdxView(self.Pats)
        self.TestEnv.Sequential = True
        self.TestEnv.Validate()
        
        # note: to create a train / test split of pats, do this:
        # all = etable.NewIdxView(self.Pats)
        # splits = split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
        # self.TrainEnv.Table = splits.Splits[0]
        # self.TestEnv.Table = splits.Splits[1]
        
        self.TrainEnv.Init(0)
        self.TestEnv.Init(0)

    def ConfigNet(self, net):
        net.InitName(net, "RA25")
        inLay = net.AddLayer2D("Input", 5, 5, emer.Input)
        hid1Lay = net.AddLayer2D("Hidden1", 7, 7, emer.Hidden)
        hid2Lay = net.AddLayer2D("Hidden2", 7, 7, emer.Hidden)
        outLay = net.AddLayer2D("Output", 5, 5, emer.Target)
        
        # use this to position layers relative to each other
        # default is Above, YAlign = Front, XAlign = Center
        hid2Lay.SetRelPos(relpos.Rel(Rel=relpos.RightOf, Other="Hidden1", YAlign=relpos.Front, Space=2))

        # note: see emergent/prjn module for all the options on how to connect
        # NewFull returns a new prjn.Full connectivity pattern
        net.ConnectLayers(inLay, hid1Lay, prjn.NewFull(), emer.Forward)
        net.ConnectLayers(hid1Lay, hid2Lay, prjn.NewFull(), emer.Forward)
        net.ConnectLayers(hid2Lay, outLay, prjn.NewFull(), emer.Forward)
        
        net.ConnectLayers(outLay, hid2Lay, prjn.NewFull(), emer.Back)
        net.ConnectLayers(hid2Lay, hid1Lay, prjn.NewFull(), emer.Back)
        
        # note: can set these to do parallel threaded computation across multiple cpus
        # not worth it for this small of a model, but definitely helps for larger ones
        # if Thread {
        #     hid2Lay.SetThread(1)
        #     outLay.SetThread(1)
        # }
  
        # note: if you wanted to change a layer type from e.g., Target to Compare, do this:
        # outLay.SetType(emer.Compare)
        # that would mean that the output layer doesn't reflect target values in plus phase
        # and thus removes error-driven learning -- but stats are still computed.

        net.Defaults()
        self.SetParams("Network", self.LogSetParams) # only set Network params
        net.Build()
        net.InitWts()

    ######################################
    #   Init, utils
        
    def Init(self):
        """Init restarts the run, and initializes everything, including network weights and resets the epoch log table"""
        rand.Seed(self.RndSeed)
        self.ConfigEnv() # just in case another set of pats was selected..
        self.StopNow = False
        self.SetParams("", self.LogSetParams) # all sheets
        self.NewRun()
        self.UpdateView(True)

    def NewRndSeed(self):
        """NewRndSeed gets a new random seed based on current time -- otherwise uses the same random seed for every run"""
        # self.RndSeed = time.Now().UnixNano()

    def Counters(self, train):
        """
        Counters returns a string of the current counter state
        use tabs to achieve a reasonable formatting overall
        and add a few tabs at the end to allow for expansion..
        """
        if train:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t" % (self.TrainEnv.Run.Cur, self.TrainEnv.Epoch.Cur, self.TrainEnv.Trial.Cur, self.Time.Cycle, self.TrainEnv.TrialName.Cur)
        else:
            return "Run:\t%d\tEpoch:\t%d\tTrial:\t%d\t\tCycle:\t%dName:\t%s\t\t\t" % (self.TrainEnv.Run.Cur, self.TrainEnv.Epoch.Cur, self.TestEnv.Trial.Cur, self.Time.Cycle, self.TestEnv.TrialName.Cur)

    def UpdateView(self, train):
        if self.NetView != 0 and self.NetView.IsVisible():
            self.NetView.Record(self.Counters(train))
            # note: essential to use Go version of update when called from another goroutine
            self.NetView.GoUpdate() # note: using counters is significantly slower..

    ######################################
    #   Running the network
    
    def AlphaCyc(self, train):
        """
        AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)     of processing.
        External inputs must have already been applied prior to calling,
        using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
        If train is true, then learning DWt or WtFmDWt calls are made.
        Handles netview updating within scope of AlphaCycle
        """
        if self.Win != 0:
            self.Win.PollEvents() # this is essential for GUI responsiveness while running
        viewUpdt = self.TrainUpdt
        if not train:
            viewUpdt = self.TestUpdt
            
        # update prior weight changes at start, so any DWt values remain visible at end
        # you might want to do this less frequently to achieve a mini-batch update
        # in which case, move it out to the TrainTrial method where the relevant
        # counters are being dealt with.
        if train:
            self.Net.WtFmDWt()
            
        self.Net.AlphaCycInit()
        self.Time.AlphaCycStart()
        for qtr in range(4):
            for cyc in range(self.Time.CycPerQtr):
                self.Net.Cycle(self.Time)
                if not train:
                    self.LogTstCyc(self.TstCycLog, self.Time.Cycle)
                self.Time.CycleInc()
                if self.ViewOn:
                    if viewUpdt == leabra.Cycle:
                        self.UpdateView(train)
                    if viewUpdt == leabra.FastSpike:
                        if (cyc+1)%10 == 0:
                            self.UpdateView(train)
            self.Net.QuarterFinal(self.Time)
            self.Time.QuarterInc()
            if self.ViewOn:
                if viewUpdt == leabra.Quarter:
                    self.UpdateView(train)
                if viewUpdt == leabra.Phase:
                    if qtr >= 2:
                        self.UpdateView(train)
        if train:
            self.Net.DWt()
        if self.ViewOn and viewUpdt == leabra.AlphaCycle:
              self.UpdateView(train)
        if self.TstCycPlot != 0 and not train:
            self.TstCycPlot.GoUpdate()

    def ApplyInputs(self, en):
        """
        ApplyInputs applies input patterns from given environment.
        It is good practice to have this be a separate method with appropriate
        args so that it can be used for various different contexts
        (training, testing, etc).
        """
        self.Net.InitExt() # clear any existing inputs -- not strictly necessary if always
                           # going to the same layers, but good practice and cheap anyway
        inLay = leabra.Layer(self.Net.LayerByName("Input"))
        outLay = leabra.Layer(self.Net.LayerByName("Output"))

        inPats = en.State(inLay.Nm)
        if inPats != go.nil:
            inLay.ApplyExt(inPats)

        outPats = en.State(outLay.Nm)
        if inPats != go.nil:
            outLay.ApplyExt(outPats)
        
        # NOTE: this is how you can use a pandas.DataFrame() to apply inputs
        # we are using etable.Table instead because it provides a full GUI
        # for viewing your patterns, and has a more convenient API, that integrates
        # with the env environment interface.
        #
        # inLay = leabra.Layer(self.Net.LayerByName("Input"))
        # outLay = leabra.Layer(self.Net.LayerByName("Output"))
        # pidx = self.Trial
        # if not self.Sequential:
        #     pidx = self.Porder[self.Trial]
        # # note: these indexes must be updated based on columns in patterns..
        # inp = self.Pats.iloc[pidx,1:26].values
        # outp = self.Pats.iloc[pidx,26:26+25].values
        # self.ApplyExt(inLay, inp)
        # self.ApplyExt(outLay, outp)
        #
        # def ApplyExt(self, lay, nparray):
        # flt = np.ndarray.flatten(nparray, 'C')
        # slc = go.Slice_float32(flt)
        # lay.ApplyExt1D(slc)
    
    def TrainTrial(self):
        """ TrainTrial runs one trial of training using TrainEnv"""
        self.TrainEnv.Step() # the Env encapsulates and manages all counter state

        # Key to query counters FIRST because current state is in NEXT epoch
        # if epoch counter has changed
        epc = env.CounterCur(self.TrainEnv, env.Epoch)
        chg = env.CounterChg(self.TrainEnv, env.Epoch)
        if chg:
            self.LogTrnEpc(self.TrnEpcLog)
            if self.ViewOn and self.TrainUpdt > leabra.AlphaCycle:
                self.UpdateView(True)
            if epc % self.TestInterval == 0: # note: epc is *next* so won't trigger first time
                self.TestAll()
            if epc >= self.MaxEpcs: # done with training..
                self.RunEnd()
                if self.TrainEnv.Run.Incr(): # we are done!
                    self.StopNow = True
                    return
                else:
                    self.NewRun()
                    return

        self.ApplyInputs(self.TrainEnv)
        self.AlphaCyc(True)   # train
        self.TrialStats(True) # accumulate

    def RunEnd(self):
        """ RunEnd is called at the end of a run -- save weights, record final log, etc here """
        self.LogRun(self.RunLog)
        if self.SaveWts:
            fnm = self.WeightsFileName()
            fmt.Printf("Saving Weights to: %v", fnm)
            self.Net.SaveWtsJSON(gi.FileName(fnm))

    def NewRun(self):
        """ NewRun intializes a new run of the model, using the TrainEnv.Run counter for the new run value """
        run = self.TrainEnv.Run.Cur
        self.TrainEnv.Init(run)
        self.TestEnv.Init(run)
        self.Time.Reset()
        self.Net.InitWts()
        self.InitStats()
        self.TrnEpcLog.SetNumRows(0)
        self.TstEpcLog.SetNumRows(0)

    def InitStats(self):
        """ InitStats initializes all the statistics, especially important for the
            cumulative epoch stats -- called at start of new run """
        # accumulators
        self.SumSSE = 0.0
        self.SumAvgSSE = 0.0
        self.SumCosDiff = 0.0
        self.CntErr = 0.0
        self.FirstZero = -1
        # clear rest just to make Sim look initialized
        self.TrlSSE = 0.0
        self.TrlAvgSSE = 0.0
        self.EpcSSE = 0.0
        self.EpcAvgSSE = 0.0
        self.EpcPctErr = 0.0
        self.EpcCosDiff = 0.0
    
    def TrialStats(self, accum):
        """
        TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
        accum is true.  Note that we're accumulating stats here on the Sim side so the
        core algorithm side remains as simple as possible, and doesn't need to worry about
        different time-scales over which stats could be accumulated etc.
        You can also aggregate directly from log data, as is done for testing stats
        """
        outLay = leabra.Layer(self.Net.LayerByName("Output"))
        self.TrlCosDiff = outLay.CosDiff.Cos
        self.TrlSSE = outLay.SSE(0.5) # 0.5 = per-unit tolerance -- right side of .5
        self.TrlAvgSSE = self.TrlSSE / len(outLay.Neurons)
        if accum:
            self.SumSSE += self.TrlSSE
            self.SumAvgSSE += self.TrlAvgSSE
            self.SumCosDiff += self.TrlCosDiff
            if self.TrlSSE != 0:
                self.CntErr += 1.0

    def TrainEpoch(self):
        """ TrainEpoch runs training trials for remainder of this epoch """
        self.StopNow = False
        curEpc = self.TrainEnv.Epoch.Cur
        while True:
            self.TrainTrial()
            if self.StopNow or self.TrainEnv.Epoch.Cur != curEpc:
                break
        self.Stopped()

    def TrainRun(self):
        """ TrainRun runs training trials for remainder of run """
        self.StopNow = False
        curRun = self.TrainEnv.Run.Cur
        while True:
            self.TrainTrial()
            if self.StopNow or self.TrainEnv.Run.Cur != curRun:
                break
        self.Stopped()

    def Train(self):
        """ Train runs the full training from this point onward """
        self.StopNow = False
        while True:
            self.TrainTrial()
            if self.StopNow:
                break
        self.Stopped()

    def Stop(self):
        """ Stop tells the sim to stop running """
        self.StopNow = True

    def Stopped(self):
        """ Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar """
        self.IsRunning = False
        if self.Win != 0:
            self.vp.BlockUpdates()
            if self.ToolBar != go.nil:
                self.ToolBar.UpdateActions()
            self.vp.UnblockUpdates()
            self.ClassView.Update()
            self.vp.SetNeedsFullRender()


    ######################################
    #     Testing

    def TestTrial(self):
        """ TestTrial runs one trial of testing -- always sequentially presented inputs """
        self.TestEnv.Step()

        # Query counters FIRST
        chg = env.CounterChg(self.TestEnv, env.Epoch)
        if chg:
            if self.ViewOn and self.TestUpdt > leabra.AlphaCycle:
                self.UpdateView(False)
            self.LogTstEpc(self.TstEpcLog)
            return
            
        self.ApplyInputs(self.TestEnv)
        self.AlphaCyc(False)   # !train
        self.TrialStats(False) # !accumulate
        self.LogTstTrl(self.TstTrlLog)


    def TestItem(self, idx):
        """ TestItem tests given item which is at given index in test item list """
        cur = self.TestEnv.Trial.Cur
        self.TestEnv.Trial.Cur = idx
        self.TestEnv.SetTrialName()
        self.ApplyInputs(self.TestEnv)
        self.AlphaCyc(False)   # !train
        self.TrialStats(False) # !accumulate
        self.TestEnv.Trial.Cur = cur

    def TestAll(self):
        """ TestAll runs through the full set of testing items """
        self.TestEnv.Init(self.TrainEnv.Run.Cur)
        while True:
            self.TestTrial()
            chg = env.CounterChg(self.TestEnv, env.Epoch)
            if chg or self.StopNow:
                break

    def RunTestAll(self):
        """ RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui """
        self.StopNow = False
        self.TestAll()
        self.Stopped()

    ##########################################
    #   Params methods

    def ParamsName(self):
        """ ParamsName returns name of current set of parameters """
        if self.ParamSet == "":
            return "Base"
        return self.ParamSet

    def SetParams(self, sheet, setMsg):
        """
        SetParams sets the params for "Base" and then current ParamSet.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
    
        if sheet == "":
            # this is important for catching typos and ensuring that all sheets can be used
            self.Params.ValidateSheets(go.Slice_string(["Network", "Sim"]))
        self.SetParamsSet("Base", sheet, setMsg)
        if self.ParamSet != "" and self.ParamSet != "Base":
            self.SetParamsSet(self.ParamSet, sheet, setMsg)

    def SetParamsSet(self, setNm, sheet, setMsg):
        """
        SetParamsSet sets the params for given params.Set name.
        If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
        otherwise just the named sheet
        if setMsg = true then we output a message for each param that was set.
        """
        pset = self.Params.SetByNameTry(setNm)
        if pset == go.nil:
            return
        if sheet == "" or sheet == "Network":
            if "Network" in pset.Sheets:
                netp = pset.SheetByNameTry("Network")
                self.Net.ApplyParams(netp, setMsg)
        if sheet == "" or sheet == "Sim":
            if "Sim" in pset.Sheets:
                simp = pset.SheetByNameTry("Sim")
                epygiv.ApplyParams(self, simp, setMsg)
        # note: if you have more complex environments with parameters, definitely add
        # sheets for them, e.g., "TrainEnv", "TestEnv" etc

    def ConfigPats(self):
        # note: this is all go-based for using etable.Table instead of pandas
        dt = self.Pats
        sc = etable.Schema()
        sc.append(etable.Column("Name", etensor.STRING, nilInts, nilStrs))
        sc.append(etable.Column("Input", etensor.FLOAT32, go.Slice_int([5, 5]), go.Slice_string(["Y", "X"])))
        sc.append(etable.Column("Output", etensor.FLOAT32, go.Slice_int([5, 5]), go.Slice_string(["Y", "X"])))
        dt.SetFromSchema(sc, 25)
            
        patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
        patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
        dt.SaveCSV("random_5x5_25_gen.dat", etable.Tab, True)

    def OpenPats(self):
        dt = self.Pats
        self.Pats = dt
        dt.SetMetaData("name", "TrainPats")
        dt.SetMetaData("desc", "Training patterns")
        dt.OpenCSV("random_5x5_25.dat", etable.Tab)
        # Note: here's how to read into a pandas DataFrame
        # dt = pd.read_csv("random_5x5_25.dat", sep='\t')
        # dt = dt.drop(columns="_H:")
 
    ##########################################
    #   Logging

    def RunName(self):
        """
        RunName returns a name for this run that combines Tag and Params -- add this to
        any file names that are saved.
        """
        if self.Tag != "":
            return self.Tag + "_" + self.ParamsName()
        else:
            return self.ParamsName()

    def RunEpochName(self, run, epc):
        """
        RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
        for using in weights file names.  Uses 3, 5 digits for each.
        """
        return "%03d_%05d" % run, epc

    def WeightsFileName(self):
        """ WeightsFileName returns default current weights file name """
        return self.Net.Nm + "_" + self.RunName() + "_" + self.RunEpochName(self.TrainEnv.Run.Cur, self.TrainEnv.Epoch.Cur) + ".wts"

    def LogFileName(self, lognm):
        """ LogFileName returns default log file name """
        return self.Net.Nm + "_" + self.RunName() + "_" + lognm + ".csv"

    #############################
    #   TrnEpcLog
        
    def LogTrnEpc(self, dt):
        """
        LogTrnEpc adds data from current epoch to a TrnEpcLog table
        computes epoch averages prior to logging.
        """
        row = dt.Rows
        self.TrnEpcLog.SetNumRows(row + 1)
        
        hid1Lay = leabra.Layer(self.Net.LayerByName("Hidden1"))
        hid2Lay = leabra.Layer(self.Net.LayerByName("Hidden2"))
        outLay = leabra.Layer(self.Net.LayerByName("Output"))

        epc = self.TrainEnv.Epoch.Prv           # this is triggered by increment so use previous value
        nt = self.TrainEnv.Table.Len() # number of trials in view
        
        self.EpcSSE = self.SumSSE / nt
        self.SumSSE = 0.0
        self.EpcAvgSSE = self.SumAvgSSE / nt
        self.SumAvgSSE = 0.0
        self.EpcPctErr = self.CntErr / nt
        self.CntErr = 0.0
        self.EpcPctCor = 1.0 - self.EpcPctErr
        self.EpcCosDiff = self.SumCosDiff / nt
        self.SumCosDiff = 0.0
        if self.FirstZero < 0 and self.EpcPctErr == 0:
            self.FirstZero = epc

        dt.SetCellFloat("Run", row, self.TrainEnv.Run.Cur)
        dt.SetCellFloat("Epoch", row, epc)
        dt.SetCellFloat("SSE", row, self.EpcSSE)
        dt.SetCellFloat("AvgSSE", row, self.EpcAvgSSE)
        dt.SetCellFloat("PctErr", row, self.EpcPctErr)
        dt.SetCellFloat("PctCor", row, self.EpcPctCor)
        dt.SetCellFloat("CosDiff", row, self.EpcCosDiff)
        dt.SetCellFloat("Hid1 ActAvg", row, hid1Lay.Pool(0).ActAvg.ActPAvgEff)
        dt.SetCellFloat("Hid2 ActAvg", row, hid2Lay.Pool(0).ActAvg.ActPAvgEff)
        dt.SetCellFloat("Out ActAvg", row, outLay.Pool(0).ActAvg.ActPAvgEff)
        
        # note: essential to use Go version of update when called from another goroutine
        if self.TrnEpcPlot != 0:
            self.TrnEpcPlot.GoUpdate()
            
        if self.TrnEpcFile != 0:
            if self.TrainEnv.Run.Cur == 0 and epc == 0:
                dt.WriteCSVHeaders(self.TrnEpcFile, etable.Tab)
            dt.WriteCSVRow(self.TrnEpcFile, row, etable.Tab)

        # note: this is how you log to a pandas.DataFrame
        # nwdat = [epc, self.EpcSSE, self.EpcAvgSSE, self.EpcPctErr, self.EpcPctCor, self.EpcCosDiff, 0, 0, 0]
        # nrow = len(self.EpcLog.index)
        # self.EpcLog.loc[nrow] = nwdat # note: this is reportedly rather slow

    def ConfigTrnEpcLog(self, dt):
        dt.SetMetaData("name", "TrnEpcLog")
        dt.SetMetaData("desc", "Record of performance over epochs of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))
        
        sc = etable.Schema()
        sc.append(etable.Column("Run", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("Epoch", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("SSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("AvgSSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("PctErr", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("PctCor", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("CosDiff", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid1 ActAvg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid2 ActAvg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Out ActAvg", etensor.FLOAT64, nilInts, nilStrs))
        dt.SetFromSchema(sc, 0)
        
        # note: pandas.DataFrame version
        # self.EpcLog = pd.DataFrame(columns=["Epoch", "SSE", "Avg SSE", "Pct Err", "Pct Cor", "CosDiff", "Hid1 ActAvg", "Hid2 ActAvg", "Out ActAvg"])
        # self.PlotVals = ["SSE", "Pct Err"]
        # self.Plot = True

    def ConfigTrnEpcPlot(self, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", False, True, 0, False, 0)
        plt.SetColParams("Epoch", False, True, 0, False, 0)
        plt.SetColParams("SSE", False, True, 0, False, 0)
        plt.SetColParams("AvgSSE", False, True, 0, False, 0)
        plt.SetColParams("PctErr", True, True, 0, True, 1) # default plot
        plt.SetColParams("PctCor", True, True, 0, True, 1) # default plot
        plt.SetColParams("CosDiff", False, True, 0, True, 1)
        plt.SetColParams("Hid1 ActAvg", False, True, 0, True, .5)
        plt.SetColParams("Hid2 ActAvg", False, True, 0, True, .5)
        plt.SetColParams("Out ActAvg", False, True, 0, True, .5)
        return plt

    #############################
    #   TstTrlLog
        
    def LogTstTrl(self, dt):
        """
        LogTstTrl adds data from current epoch to the TstTrlLog table
        log always contains number of testing items
        """
        dt = self.TstTrlLog

        inLay = leabra.Layer(self.Net.LayerByName("Input"))
        hid1Lay = leabra.Layer(self.Net.LayerByName("Hidden1"))
        hid2Lay = leabra.Layer(self.Net.LayerByName("Hidden2"))
        outLay = leabra.Layer(self.Net.LayerByName("Output"))

        epc = self.TrainEnv.Epoch.Prv           # this is triggered by increment so use previous value
        trl = self.TestEnv.Trial.Cur  
        
        dt.SetCellFloat("Epoch", trl, epc)
        dt.SetCellFloat("Trial", trl, trl)
        dt.SetCellString("TrialName", trl, self.TestEnv.TrialName.Cur)
        dt.SetCellFloat("SSE", trl, self.TrlSSE)
        dt.SetCellFloat("AvgSSE", trl, self.TrlAvgSSE)
        dt.SetCellFloat("CosDiff", trl, self.TrlCosDiff)
        dt.SetCellFloat("Hid1 ActM.Avg", trl, hid1Lay.Pool(0).ActM.Avg)
        dt.SetCellFloat("Hid2 ActM.Avg", trl, hid2Lay.Pool(0).ActM.Avg)
        dt.SetCellFloat("Out ActM.Avg", trl, outLay.Pool(0).ActM.Avg)
        
        if self.InputValsTsr == 0: # re-use same tensors so not always reallocating mem
            self.InputValsTsr = etensor.Float32()
            self.OutputValsTsr = etensor.Float32()
        inLay.UnitValsTensor(self.InputValsTsr, "Act")
        dt.SetCellTensor("InAct", trl, self.InputValsTsr)
        outLay.UnitValsTensor(self.OutputValsTsr, "ActM")
        dt.SetCellTensor("OutActM", trl, self.OutputValsTsr)
        outLay.UnitValsTensor(self.OutputValsTsr, "ActP")
        dt.SetCellTensor("OutActP", trl, self.OutputValsTsr)
        
        # note: essential to use Go version of update when called from another goroutine
        if self.TstTrlPlot != 0:
            self.TstTrlPlot.GoUpdate()

    def ConfigTstTrlLog(self, dt):
        inLay = leabra.Layer(self.Net.LayerByName("Input"))
        outLay = leabra.Layer(self.Net.LayerByName("Output"))
        
        dt.SetMetaData("name", "TstTrlLog")
        dt.SetMetaData("desc", "Record of testing per input pattern")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))
        nt = self.TestEnv.Table.Len() # number in view
        
        sc = etable.Schema()
        sc.append(etable.Column("Run", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("Epoch", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("Trial", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("TrialName", etensor.STRING, nilInts, nilStrs))
        sc.append(etable.Column("SSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("AvgSSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("CosDiff", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid1 ActM.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid2 ActM.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Out ActM.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("InAct", etensor.FLOAT64, inLay.Shp.Shp, nilStrs))
        sc.append(etable.Column("OutActM", etensor.FLOAT64, outLay.Shp.Shp, nilStrs))
        sc.append(etable.Column("OutActP", etensor.FLOAT64, outLay.Shp.Shp, nilStrs))
        dt.SetFromSchema(sc, nt)
        

    def ConfigTstTrlPlot(self, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Test Trial Plot"
        plt.Params.XAxisCol = "Trial"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", False, True, 0, False, 0)
        plt.SetColParams("Epoch", False, True, 0, False, 0)
        plt.SetColParams("Trial", False, True, 0, False, 0)
        plt.SetColParams("TrialName", False, True, 0, False, 0)
        plt.SetColParams("SSE", False, True, 0, False, 0)
        plt.SetColParams("AvgSSE", False, True, 0, False, 0)
        plt.SetColParams("CosDiff", True, True, 0, True, 1)
        plt.SetColParams("Hid1 ActM.Avg", True, True, 0, True, .5)
        plt.SetColParams("Hid2 ActM.Avg", True, True, 0, True, .5)
        plt.SetColParams("Out ActM.Avg", True, True, 0, True, .5)

        plt.SetColParams("InAct", False, True, 0, True, 1)
        plt.SetColParams("OutActM", False, True, 0, True, 1)
        plt.SetColParams("OutActP", False, True, 0, True, 1)
        return plt
        
    #############################
    #   TstEpcLog
        
    def LogTstEpc(self, dt):
        """
        LogTstEpc adds data from current epoch to the TstEpcLog table
        log always contains number of testing items
        """
        row = dt.Rows
        dt.SetNumRows(row + 1)

        trl = self.TstTrlLog
        tix = etable.NewIdxView(trl)
        epc = self.TrainEnv.Epoch.Prv

        # note: this shows how to use agg methods to compute summary data from another
        # data table, instead of incrementing on the Sim
        dt.SetCellFloat("Run", row, self.TrainEnv.Run.Cur)
        dt.SetCellFloat("Epoch", row, epc)
        dt.SetCellFloat("SSE", row, agg.Sum(tix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(tix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.PropIf(tix, "SSE", lambda idx, val: val > 0)[0])
        dt.SetCellFloat("PctCor", row, agg.PropIf(tix, "SSE", lambda idx, val: val == 0)[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])
        
        trlix = etable.NewIdxView(trl)
        trlix.Filter(FilterSSE)
        
        self.TstErrLog = trlix.NewTable()
        
        allsp = split.All(trlix)
        split.Agg(allsp, "SSE", agg.AggSum)
        split.Agg(allsp, "AvgSSE", agg.AggMean)
        split.Agg(allsp, "InAct", agg.AggMean)
        split.Agg(allsp, "OutActM", agg.AggMean)
        split.Agg(allsp, "OutActP", agg.AggMean)
        
        self.TstErrStats = allsp.AggsToTable(False)
        
        # note: essential to use Go version of update when called from another goroutine
        if self.TstEpcPlot != 0:
            self.TstEpcPlot.GoUpdate()

    def ConfigTstEpcLog(self, dt):
        dt.SetMetaData("name", "TstEpcLog")
        dt.SetMetaData("desc", "Summary stats for testing trials")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))
        
        sc = etable.Schema()
        sc.append(etable.Column("Run", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("Epoch", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("SSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("AvgSSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("PctErr", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("PctCor", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("CosDiff", etensor.FLOAT64, nilInts, nilStrs))
        dt.SetFromSchema(sc, 0)
        

    def ConfigTstEpcPlot(self, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Testing Epoch Plot"
        plt.Params.XAxisCol = "Epoch"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", False, True, 0, False, 0)
        plt.SetColParams("Epoch", False, True, 0, False, 0)
        plt.SetColParams("SSE", False, True, 0, False, 0)
        plt.SetColParams("AvgSSE", False, True, 0, False, 0)
        plt.SetColParams("PctErr", True, True, 0, True, 1) # default plot
        plt.SetColParams("PctCor", True, True, 0, True, 1) # default plot
        plt.SetColParams("CosDiff", False, True, 0, True, 1)
        return plt
        
    #############################
    #   TstCycLog
        
    def LogTstCyc(self, dt, cyc):
        """
        LogTstCyc adds data from current trial to the TstCycLog table.
        log just has 100 cycles, is overwritten
        """
        if dt.Rows <= cyc:
            dt.SetNumRows(cyc + 1)
        
        hid1Lay = leabra.Layer(self.Net.LayerByName("Hidden1"))
        hid2Lay = leabra.Layer(self.Net.LayerByName("Hidden2"))
        outLay = leabra.Layer(self.Net.LayerByName("Output"))
        
        dt.SetCellFloat("Cycle", cyc, cyc)
        dt.SetCellFloat("Hid1 Ge.Avg", cyc, hid1Lay.Pool(0).Inhib.Ge.Avg)
        dt.SetCellFloat("Hid2 Ge.Avg", cyc, hid2Lay.Pool(0).Inhib.Ge.Avg)
        dt.SetCellFloat("Out Ge.Avg", cyc, outLay.Pool(0).Inhib.Ge.Avg)
        dt.SetCellFloat("Hid1 Act.Avg", cyc, hid1Lay.Pool(0).Inhib.Act.Avg)
        dt.SetCellFloat("Hid2 Act.Avg", cyc, hid2Lay.Pool(0).Inhib.Act.Avg)
        dt.SetCellFloat("Out Act.Avg", cyc, outLay.Pool(0).Inhib.Act.Avg)
        
        if self.TstCycPlot != 0 and cyc % 10 == 0: # too slow to do every cyc
            # note: essential to use Go version of update when called from another goroutine
            self.TstCycPlot.GoUpdate()

    def ConfigTstCycLog(self, dt):
        dt.SetMetaData("name", "TstCycLog")
        dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))
        np = 100 # max cycles
        
        sc = etable.Schema()
        sc.append(etable.Column("Cycle", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid1 Ge.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid2 Ge.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Out Ge.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid1 Act.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Hid2 Act.Avg", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("Out Act.Avg", etensor.FLOAT64, nilInts, nilStrs))
        dt.SetFromSchema(sc, np)
        
    def ConfigTstCycPlot(self, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Test Cycle Plot"
        plt.Params.XAxisCol = "Cycle"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Cycle", False, True, 0, False, 0)
        plt.SetColParams("Hid1 Ge.Avg", True, True, 0, True, .5)
        plt.SetColParams("Hid2 Ge.Avg", True, True, 0, True, .5)
        plt.SetColParams("Out Ge.Avg", True, True, 0, True, .5)
        plt.SetColParams("Hid1 Act.Avg", True, True, 0, True, .5)
        plt.SetColParams("Hid2 Act.Avg", True, True, 0, True, .5)
        plt.SetColParams("Out Act.Avg", True, True, 0, True, .5)
        return plt

    #############################
    #   RunLog
        
    def LogRun(self, dt):
        run = self.TrainEnv.Run.Cur # this is NOT triggered by increment yet -- use Cur
        row = dt.Rows
        self.RunLog.SetNumRows(row + 1)
        
        epclog = self.TrnEpcLog
        # compute mean over last N epochs for run level
        nlast = 10
        epcix = etable.NewIdxView(epclog)
        epcix.Idxs = go.Slice_int(epcix.Idxs[epcix.Len()-nlast-1:])
        # print(epcix.Idxs[epcix.Len()-nlast-1:])
        
        params = self.RunName() # includes tag
        
        dt.SetCellFloat("Run", row, run)
        dt.SetCellString("Params", row, params)
        dt.SetCellFloat("FirstZero", row, self.FirstZero)
        dt.SetCellFloat("SSE", row, agg.Mean(epcix, "SSE")[0])
        dt.SetCellFloat("AvgSSE", row, agg.Mean(epcix, "AvgSSE")[0])
        dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
        dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
        dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])
        
        runix = etable.NewIdxView(dt)
        spl = split.GroupBy(runix, go.Slice_string(["Params"]))
        split.Desc(spl, "FirstZero")
        split.Desc(spl, "PctCor")
        self.RunStats = spl.AggsToTable(False)
        
        # note: essential to use Go version of update when called from another goroutine
        if self.RunPlot != 0:
            self.RunPlot.GoUpdate()
            
        if self.RunFile != 0:
            if row == 0:
                dt.WriteCSVHeaders(self.RunFile, etable.Tab)
            dt.WriteCSVRow(self.RunFile, row, etable.Tab)
            
    def ConfigRunLog(self, dt):
        dt.SetMetaData("name", "RunLog")
        dt.SetMetaData("desc", "Record of performance at end of training")
        dt.SetMetaData("read-only", "true")
        dt.SetMetaData("precision", str(LogPrec))
        
        sc = etable.Schema()
        sc.append(etable.Column("Run", etensor.INT64, nilInts, nilStrs))
        sc.append(etable.Column("Params", etensor.STRING, nilInts, nilStrs))
        sc.append(etable.Column("FirstZero", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("SSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("AvgSSE", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("PctErr", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("PctCor", etensor.FLOAT64, nilInts, nilStrs))
        sc.append(etable.Column("CosDiff", etensor.FLOAT64, nilInts, nilStrs))
        dt.SetFromSchema(sc, 0)

    def ConfigRunPlot(self, plt, dt):
        plt.Params.Title = "Leabra Random Associator 25 Run Plot"
        plt.Params.XAxisCol = "Run"
        plt.SetTable(dt)
        # order of params: on, fixMin, min, fixMax, max
        plt.SetColParams("Run", False, True, 0, False, 0)
        plt.SetColParams("FirstZero", True, True, 0, False, 0) # default plot
        plt.SetColParams("SSE", False, True, 0, False, 0)
        plt.SetColParams("AvgSSE", False, True, 0, False, 0)
        plt.SetColParams("PctErr", False, True, 0, True, 1)
        plt.SetColParams("PctCor", False, True, 0, True, 1)
        plt.SetColParams("CosDiff", False, True, 0, True, 1)
        return plt

    ##############################
    #   ConfigGui

    def ConfigGui(self):
        """ConfigGui configures the GoGi gui interface for this simulation"""
        width = 1600
        height = 1200
        
        gi.SetAppName("ra25")
        gi.SetAppAbout('This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>')
        
        win = gi.NewMainWindow("ra25", "Leabra Random Associator", width, height)
        self.Win = win

        vp = win.WinViewport2D()
        self.vp = vp
        updt = vp.UpdateStart()
         
        mfr = win.SetMainFrame()
        
        tbar = gi.AddNewToolBar(mfr, "tbar")
        tbar.SetStretchMaxWidth()
        self.ToolBar = tbar
        
        split = gi.AddNewSplitView(mfr, "split")
        split.Dim = gi.X
        split.SetStretchMaxWidth()
        split.SetStretchMaxHeight()
         
        self.ClassView = epygiv.ClassView("ra25sv", self.Tags)
        self.ClassView.AddFrame(split)
        self.ClassView.SetClass(self)

        tv = gi.AddNewTabView(split, "tv")
        
        nv = netview.NetView()
        tv.AddTab(nv, "NetView")
        nv.Var = "Act"
        nv.SetNet(self.Net)
        self.NetView = nv
        
        plt = eplot.Plot2D()
        tv.AddTab(plt, "TrnEpcPlot")
        self.TrnEpcPlot = self.ConfigTrnEpcPlot(plt, self.TrnEpcLog)
        
        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstTrlPlot")
        self.TstTrlPlot = self.ConfigTstTrlPlot(plt, self.TstTrlLog)
        
        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstCycPlot")
        self.TstCycPlot = self.ConfigTstCycPlot(plt, self.TstCycLog)
        
        plt = eplot.Plot2D()
        tv.AddTab(plt, "TstEpcPlot")
        self.TstEpcPlot = self.ConfigTstEpcPlot(plt, self.TstEpcLog)
        
        plt = eplot.Plot2D()
        tv.AddTab(plt, "RunPlot")
        self.RunPlot = self.ConfigRunPlot(plt, self.RunLog)

        split.SetSplitsList(go.Slice_float32([.3, .7]))
        
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
        # fmen = win.MainMenu.ChildByName("File", 0).(*gi.Action)
        # fmen.Menu = make(gi.Menu, 0, 10)
        # fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
        #     recv, func(recv, send ki.Ki, sig int64, data interface{}) {
        #     FileViewOpenSVG(vp)
        #     })
        # fmen.Menu.AddSeparator("csep")
        # fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
        #     recv, func(recv, send ki.Ki, sig int64, data interface{}) {
        #     win.CloseReq()
        #     })
                
        #    win.SetCloseCleanFunc(func(w *gi.Window) {
        #         gi.Quit() # once main window is closed, quit
        #     })
        #         
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
    else:
        TheSim.ConfigGui()
        print("Note: run pyleabra -i ra25.py to run in interactive mode, or just pyleabra, then 'import ra25'")
        print("for non-gui background running, here are the args:")
        usage()
        
main(sys.argv[1:])

