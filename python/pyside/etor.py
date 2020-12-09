# Copyright (c) 2020, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# etor is python-side library for eTorch, for saving and copying network
# state for visualization.

# note: from below here should be updated for standalone etorch vs. leabra

from etorch import go, etorch, gi, netview

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
        self.wtmap = {}  # dict of names for prjn weights
        self.net = 0 # network that we save to
    
    def set_net(self, net):
        """
        set_net sets the etorch.Network to display to
        """
        self.net = net
    
    def rec(self, x, var):
        """
        rec records current tensor state x to variable named var
        """
        if not self.record:
            return
        if self.trace:
            print(var, x.size())

        sd = self.nn.state_dict()
        net = self.net
        
        nmv = var.split(".")
        vnm = nmv[-1]
        lnm = ".".join(nmv[:-1])
        ly = etorch.Layer(net.LayerByName(lnm))
        nst = ly.States[vnm]
        nst.Values.copy(torch.flatten(x))
        
        if not self.rec_wts:
            return
            
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
        call update to update display
        """
        ss.NetView.Record("") # note: can include any kind of textual state information here to display too
        ss.NetView.GoUpdate()
        
