#!/usr/local/bin/python3

# Copyright (c) 2020, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# use:
# leabra-to.py mysim.go
#
# Generates mysim.py conversion of mysim.go, attempting to 

import os, sys, subprocess

debug = False

# these are defined below
inserts = []
replaces = []
deletes = []

def read_as_string(fnm):
    # reads file as string
    if not os.path.isfile(fnm):
        return ""
    with open(fnm, "r") as f:
        val = f.read()
    return val

def write_string(fnm, stval):
    with open(fnm,"w") as f:
        f.write(stval)

def gotopy(fname):
    result = subprocess.run(["gotopy","-gogi", fname], capture_output=True)
    if len(result.stderr) > 0:
        print(str(result.stderr, "utf-8"))
    return str(result.stdout, "utf-8")
    
def repls(txt):
    txt = txt.replace("leabra.LeabraLayer(", "leabra.Layer(")
    txt = txt.replace(".AsLeabra()", "")
    txt = txt.replace("(`", "('")
    txt = txt.replace("`)", "')")
    txt = txt.replace(" = ss.TrainUpdt", " = ss.TrainUpdt.value")
    txt = txt.replace(" = ss.TestUpdt", " = ss.TestUpdt.value")
    txt = txt.replace(" ss.TrainUpdt >", " ss.TrainUpdt.value >")
    txt = txt.replace(" ss.TestUpdt >", " ss.TestUpdt.value >")
    return txt
    
def inserttxt(txt, ati, ins):
    if debug:
        print("\n##########\nins:")
        print(ins)
    for i, v in enumerate(ins):
        txt.insert(ati+i, v)

def repltxt(txt, ati, ftxt, itxt):
    if debug:
        print("\n##########\nrepl:")
        print(ftxt)
        print("with:")
        print(itxt)
    nf = len(ftxt)
    ni = len(itxt)
    for i, v in enumerate(itxt):
        if i < nf:
            txt[ati+i] = v
        else:
            txt.insert(ati+i, v)
    if nf > ni:
        del txt[ati+ni:ati+nf]

def diffs(txt):
    lns = txt.splitlines()
    nln = lns.copy()
    ni = 0
    insi = -1
    rpli = -1
    deli = -1
    didone = False
    for i, v in enumerate(lns):
        for j, ir in enumerate(inserts):
            if j <= insi:
                continue
            ftxt = ir[0]
            lnoff = ir[1]
            itxt = ir[2]
            if ftxt in v:
                inserttxt(nln, ni+lnoff, itxt)
                ni += len(itxt)
                insi = j
                break
        for j, rp in enumerate(replaces):
            if j <= rpli:
                continue
            ftxt = rp[0]
            itxt = rp[1]
            if ftxt[0] == v:
                repltxt(nln, ni, ftxt, itxt)
                ni += len(itxt) - len(ftxt)
                rpli = j
                break
        for j, ft in enumerate(deletes):
            if j <= deli:
                continue
            if ft[0] == v:
                if debug:
                    print("\n##########\ndel:")
                    print(ft)
                del nln[ni:ni+len(ft)]
                ni -= len(ft)
                deli = j
                break
        ni += 1
    return '\n'.join(nln)
    
def column(txt):
    lns = txt.splitlines()
    insc = False
    start = False
    for i, v in enumerate(lns):
        if " = etable.Schema(" in v:
            insc = True
            start = True
            continue
        if insc and "etensor." in v:
            op = v.find('("')
            if op < 0:
                insc = False
                continue
            if start:
                lns[i] = v[:op] + "[etable.Column" + v[op:]
                start = False
            else:
                lns[i] = v[:op] + "etable.Column" + v[op:]
        elif insc:
            lns[i-1] = lns[i-1][:-1] + "]"
            insc = False
            continue
    return '\n'.join(lns)

def main(argv):
    if len(argv) < 2 or argv[1] == "help":
        print("\n%s converts leabra .go sim file to Python .py file\n" % argv[0])
        print("usage: just takes the input filename")
        exit(0)

    fname = argv[1]
    outfn = os.path.splitext(fname)[0] + ".py"
    raw = gotopy(fname)
    txt = diffs(raw)
    txt = repls(txt)
    txt = column(txt)
    write_string(outfn, txt)

##############################################
### text edits

# the only constraint is that these must be *in sequential order* -- the index
# is incremented for every match, so it doesn't revisit any matches more than once

## tuple elements are: find, offset, text

inserts = [
("def New(ss):", -1, [
'''        self.vp  = 0''',
'''        self.SetTags("vp", 'view:"-" desc:"viewport"')''',
'',
'''    def InitParams(ss):''',
'''        """''',
'''        Sets the default set of parameters -- Base is always applied, and others can be optionally''',
'''        selected to apply on top of that''',
'''        """''',
'''        ss.Params.OpenJSON("my.params")''',
]),
("def Config(ss):", 4, [
'''        ss.InitParams()'''
]),
("viewUpdt = ss.TrainUpdt", 0, [
'''        if ss.Win != 0:''',
'''            ss.Win.PollEvents() # this is essential for GUI responsiveness while running''',
]),
("def Stopped(ss):", 10, [
'''            ss.UpdateClassView()'''
]),
("vp = win.WinViewport2D()", 1, [
'''        ss.vp = vp''',
]),
]

replaces = [
([
'''        err = net.Build()''',
'''        if err != 0:''',
'''            log.Println(err)''',
'''            return''',
],[
'''        net.Build()'''
]),
([
'''        ss.RndSeed = time.Now().UnixNano()''',
],[
'''        ss.RndSeed = int(datetime.now(timezone.utc).timestamp())''',
]),
([
'''                    switch viewUpdt:''',
'''                    if leabra.Cycle:''',
],[
'''                    if viewUpdt == leabra.Cycle:''',
]),
([
'''                    if leabra.FastSpike:'''
],[
'''                    if viewUpdt == leabra.FastSpike:'''
]),
([
'''            if ss.ViewOn:''',
'''                switch :'''
],[
'''            if ss.ViewOn:'''
]),
([
'''        epc, _, chg = ss.TrainEnv.Counter(env.Epoch)''',
],[
'''        epc = env.CounterCur(ss.TrainEnv, env.Epoch)''',
'''        chg = env.CounterChg(ss.TrainEnv, env.Epoch)''',
]),
([
'''        ss.TrlSSE, ss.TrlAvgSSE = out.MSE(0.5)''',
],[
'''        ss.TrlSSE = out.SSE(0.5) # 0.5 = per-unit tolerance -- right side of .5''',
'''        ss.TrlAvgSSE = ss.TrlSSE / len(out.Neurons)'''
]),
([
'''        _, _, chg = ss.TestEnv.Counter(env.Epoch)'''
],[
'''        chg = env.CounterChg(ss.TestEnv, env.Epoch)'''
]),
([
'''            _, _, chg = ss.TestEnv.Counter(env.Epoch)'''
],[
'''            chg = env.CounterChg(ss.TestEnv, env.Epoch)'''
]),
([
'''        err = ss.SetParamsSet("Base", sheet, setMsg)''',
'''        if ss.ParamSet != "" and ss.ParamSet != "Base":''',
'''            sps = ss.ParamSet.split()''',
'''            for ps in sps :''',
'''                err = ss.SetParamsSet(ps, sheet, setMsg)''',
'''        return err''',
],[
'''        ss.SetParamsSet("Base", sheet, setMsg)''',
'''        if ss.ParamSet != "" and ss.ParamSet != "Base":''',
'''            sps = ss.ParamSet.split()''',
'''            for ps in sps:''',
'''                ss.SetParamsSet(ps, sheet, setMsg)''',
]),
([
'''        pset, err = ss.Params.SetByNameTry(setNm)''',
'''        if err != 0:''',
'''            return err''',
'''        if sheet == "" or sheet == "Network":''',
'''            netp, ok = pset.Sheets["Network"]''',
'''            if ok:''',
'''                ss.Net.ApplyParams(netp, setMsg)''',
'',
'''        if sheet == "" or sheet == "Sim":''',
'''            simp, ok = pset.Sheets["Sim"]''',
'''            if ok:''',
'''                simp.Apply(ss, setMsg)''',
'',
'''        return err''',
],[
'''        pset = ss.Params.SetByNameTry(setNm)''',
'''        if sheet == "" or sheet == "Network":''',
'''            if "Network" in pset.Sheets:''',
'''                netp = pset.SheetByNameTry("Network")''',
'''                ss.Net.ApplyParams(netp, setMsg)''',
'''        if sheet == "" or sheet == "Sim":''',
'''            if "Sim" in pset.Sheets:''',
'''                simp= pset.SheetByNameTry("Sim")''',
'''                pyparams.ApplyParams(ss, simp, setMsg)''',
]),
([
'''        if ss.ValsTsrs == 0:''',
'''            ss.ValsTsrs = make({})''',
'''        tsr, ok = ss.ValsTsrs[name]''',
'''        if not ok:''',
'''            tsr = etensor.Float32()''',
'''            ss.ValsTsrs[name] = tsr''',
],[
'''        if name in ss.ValsTsrs:''',
'''            return ss.ValsTsrs[name]''',
'''        tsr = etensor.Float32()''',
'''        ss.ValsTsrs[name] = tsr''',
]),
([
'''        sv = giv.AddNewStructView(split, "sv")''',
'''        sv.SetStruct(ss)'''
],[
'''        cv = ss.NewClassView("sv")''',
'''        cv.AddFrame(split)''',
'''        cv.Config()'''
]),
([
'''        nv = *netview.NetView(tv.AddNewTab(netview.KiT_NetView, "NetView"))'''
],[
'''        nv = netview.NetView()''',
'''        tv.AddTab(nv, "NetView")'''
]),
([
'''        plt = *eplot.Plot2D(tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot"))'''
],[
'''        plt = eplot.Plot2D()''',
'''        tv.AddTab(plt, "TrnEpcPlot")'''
]),
([
'''        plt = *eplot.Plot2D(tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot"))'''
],[
'''        plt = eplot.Plot2D()''',
'''        tv.AddTab(plt, "TstTrlPlot")'''
]),
([
'''        plt = *eplot.Plot2D(tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot"))'''
],[
'''        plt = eplot.Plot2D()''',
'''        tv.AddTab(plt, "TstCycPlot")'''
]),
([
'''        plt = *eplot.Plot2D(tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot"))'''
],[
'''        plt = eplot.Plot2D()''',
'''        tv.AddTab(plt, "TstEpcPlot")'''
]),
([
'''        plt = *eplot.Plot2D(tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot"))'''
],[
'''        plt = eplot.Plot2D()''',
'''        tv.AddTab(plt, "RunPlot")'''
]),
([
'''        plt = *eplot.Plot2D(tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot"))'''
],[
'''        plt = eplot.Plot2D()''',
'''        tv.AddTab(plt, "TrnEpcPlot")'''
]),
([
'''        split.SetSplits(.2, .8)'''
],[
'''        split.SetSplitsList(go.Slice_float32([.2, .8]))''',
'''        recv = win.This()'''
]),
]

deletes = [
[
'''    # re-config env just in case a different set of patterns was''',
],[
'''    # selected or patterns have been modified etc''',
],[
'''    # ss.Win.PollEvents() // this can be used instead of running in a separate goroutine''',
],[
'''    # update prior weight changes at start, so any DWt values remain visible at end''',
'''    # you might want to do this less frequently to achieve a mini-batch update''',
'''    # in which case, move it out to the TrainTrial method where the relevant''',
'''    # counters are being dealt with.''',
]
]

main(sys.argv)
    
