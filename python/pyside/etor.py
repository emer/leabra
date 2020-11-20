# Copyright (c) 2020, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# etor is python-side library for eTorch, for saving and copying network
# state for visualization.

# note: from below here should be updated for standalone etorch vs. leabra

from leabra import go, etorch, gi, netview

import torch

class State(object):
    """
    State manages saving and copying of network state
    """
    def __init__(self, nn):
        self.nn = nn  # our torch.nn module
        self.record = True # set to False to turn off recording
        self.rec_wts = False # set to True to turn on recording of prjn-level weight state
        self.trace = False # print out dimensions of what is recorded -- useful for initial config
        self.layers = {} # dict of layer-level state vectors
        self.wtmap = {}  # dict of names for prjn weights
        self.net = 0 # network that we save to
    
    def init_net(self, net):
        """
        init initializes from etorch.Network, once it has been configured
        """
        self.net = net
        for li in net.Layers:
            ly = etorch.Layer(handle=li)
            self.layers[ly.Name()+".Act"] = torch.FloatTensor(ly.Shp.Len())
            self.layers[ly.Name()+".Net"] = torch.FloatTensor(ly.Shp.Len())
    
    def rec(self, x, var):
        """
        rec records current tensor state x to variable named var
        """
        if not self.record:
            return
        if self.trace:
            print(var, x.size())
        if var in self.layers:
            st = self.layers[var]
            st[:] = torch.flatten(x)[:]  # element-wise copy, re-using existing memory

    def update(self):
        """
        update copies saved state values into etorch.Network, for display
        """
        if not self.record:
            return
        sd = self.nn.state_dict()
        net = self.net

        def copy_lay(ly, var):
            tst = self.layers[ly.Name() + "." + var]
            nst = ly.States[var]
            nst.Values.copy(tst)
        
        for li in net.Layers:
            ly = etorch.Layer(handle=li)
            copy_lay(ly, "Act")
            copy_lay(ly, "Net")
            if self.rec_wts:
                for pi in ly.RcvPrjns:
                    pj = etorch.Prjn(handle=pi)
                    pnm = pj.Name()
                    if not pnm in self.wtmap:
                        continue
                    wnm = self.wtmap[pnm]
                    wts = sd[wnm + ".weight"]
                    pst = pj.States["Wt"]
                    pst.Values.copy(torch.flatten(wts))
                    bnm = wnm + ".bias"
                    if bnm in sd:
                        bst = sd[bnm]
                        lst = ly.States["Bias"]
                        lst.Values.copy(torch.flatten(bst))

    def report(self):
        """
        report prints out the state dimensions
        """
        for k, tt in self.layers:
            print("layer: ", k, "size: ", tt.size())
        
                        
class NetView(object):
    """
    NetView opens a separate window with the network view -- for standalone use.
    """
    def __init__(self, net):
        self.Net = net
        self.NetView = 0
        self.Win = 0
        self.vp = 0
        
    def open(ss):
        """
        open opens the window of this gui
        """
        width = 1600
        height = 1200

        win = gi.NewMainWindow("netview", "eTorch NetView", width, height)
        ss.Win = win

        vp = win.WinViewport2D()
        ss.vp = vp
        updt = vp.UpdateStart()

        mfr = win.SetMainFrame()

        nv = netview.NetView()
        mfr.AddChild(nv)
        nv.Var = "Act"
        nv.SetNet(ss.Net)
        ss.NetView = nv

        # main menu
        appnm = gi.AppName()
        mmen = win.MainMenu
        mmen.ConfigMenus(go.Slice_string([appnm, "File", "Edit", "Window"]))

        amen = gi.Action(win.MainMenu.ChildByName(appnm, 0))
        amen.Menu.AddAppMenu(win)

        emen = gi.Action(win.MainMenu.ChildByName("Edit", 1))
        emen.Menu.AddCopyCutPaste(win)
        win.MainMenuUpdated()
        vp.UpdateEndNoSig(updt)
        win.GoStartEventLoop()

    def update(ss):
        """
        call update to update display -- must call State.update() to get state first
        """
        ss.NetView.Record("") # note: can include any kind of textual state information here to display too
        ss.NetView.GoUpdate()
        
