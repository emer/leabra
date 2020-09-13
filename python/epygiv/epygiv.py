# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, gi, giv, etable, etview, params, simat, eplot, etensor
from enum import Enum

import pandas as pd

# ApplyParams applies params.Sheet to cls
def ApplyParams(cls, sheet, setMsg):
    flds = cls.__dict__
    for sl in sheet:
        sel = params.Sel(handle=sl)
        for nm, val in sel.Params:
            flnm = nm.split('.')[1]
            # print("name: %s, value: %s\n" % (flnm, val))
            if flnm in flds:
                cur = getattr(cls, flnm)
                if isinstance(cur, int):
                    setattr(cls, flnm, int(val))
                elif isinstance(cur, float):
                    setattr(cls, flnm, float(val))
                else:
                    setattr(cls, flnm, val)
                if setMsg:
                    print("Field named: %s set to value: %s\n" % (flnm, val))
            else:
                print("ApplyParams error: field: %s not found in class\n" % flnm)
                

# classviews is a dictionary of classviews -- needed for callbacks
classviews = {}

def HasTagValue(tags, tag, value):
    """ returns true if given tag has given value """
    if not tag in tags:
        return False
    tv = tags[tag]
    return tv == value

def GoObjView(val, nm, frame, ctxt, tags):
    """
    returns a gi.Widget representing the given value, with given name.
    frame = gi.Frame or layout to add widgets to -- also callback recv
    ctxt = context for this object (e.g., name of owning struct)
    """
    vw = 0
    fnm = ctxt + ":" + nm
    if isinstance(val, bool):
        vw = gi.AddNewCheckBox(frame, fnm)
        vw.SetChecked(val)
        vw.ButtonSig.Connect(frame, SetBoolValCB)
    elif isinstance(val, Enum):
        vw = gi.AddNewComboBox(frame, fnm)
        vw.SetText(nm)
        vw.SetPropStr("padding", "2px")
        vw.SetPropStr("margin", "2px")
        ItemsFromEnum(vw, val)
        vw.ComboSig.Connect(frame, SetEnumCB)
    elif isinstance(val, go.GoClass):
        vw = gi.AddNewAction(frame, fnm)
        if hasattr(val, "Label"):
            vw.SetText(val.Label())
        else:
            vw.SetText(nm)
        vw.SetPropStr("padding", "2px")
        vw.SetPropStr("margin", "2px")
        vw.SetPropStr("border-radius", "4px")
        vw.ActionSig.Connect(frame, EditGoObjCB)
    elif isinstance(val, pd.DataFrame):
        vw = gi.AddNewAction(frame, fnm)
        vw.SetText(nm)
        vw.SetPropStr("padding", "2px")
        vw.SetPropStr("margin", "2px")
        vw.SetPropStr("border-radius", "4px")
        vw.ActionSig.Connect(frame, EditObjCB)
    elif isinstance(val, ClassViewObj):
        vw = gi.AddNewAction(frame, fnm)
        vw.SetText(nm)
        vw.SetPropStr("padding", "2px")
        vw.SetPropStr("margin", "2px")
        vw.SetPropStr("border-radius", "4px")
        vw.ActionSig.Connect(frame, EditObjCB)
    elif isinstance(val, (int, float)):
        vw = gi.AddNewSpinBox(frame, fnm)
        vw.SetValue(val)
        vw.SpinBoxSig.Connect(frame, SetIntValCB)
    else:
        vw = gi.AddNewTextField(frame, fnm)
        vw.SetText(str(val))
        vw.SetPropStr("min-width", "10em")
        vw.TextFieldSig.Connect(frame, SetStrValCB)
    if HasTagValue(tags, "inactive", "+"):
        vw.SetInactive()
    return vw

def GoObjUpdtView(val, vw, nm):
    """
    updates the given view widget for given value
    """
    if isinstance(val, bool):
        if isinstance(vw, gi.CheckBox):
            svw = gi.CheckBox(vw)
            svw.SetChecked(val)
        else:
            print("epygiv; bool value: %s doesn't have CheckBox widget" % nm)
    elif isinstance(val, Enum):
        if isinstance(vw, gi.ComboBox):
            svw = gi.ComboBox(vw)
            svw.SetCurVal(val.name)
        else:
            print("epygiv; Enum value: %s doesn't have ComboBox widget" % nm)
    elif isinstance(val, go.GoClass):
        pass
    elif isinstance(val, pd.DataFrame):
        pass
    elif isinstance(val, ClassViewObj):
        pass
    elif isinstance(val, (int, float)):
        if isinstance(vw, gi.SpinBox):
            svw = gi.SpinBox(vw)
            svw.SetValue(val)
        else:
            print("epygiv; numerical value: %s doesn't have SpinBox widget" % nm)
    else:
        if isinstance(vw, gi.TextField):
            tvw = gi.TextField(vw)
            tvw.SetText(str(val))
        else:
            print("epygiv; object %s = %s doesn't have expected TextField widget" % (nm, val))
    
def GoObjDialog(vp, obj, title):
    """
    opens a dialog for given Go object, returns dialog
    """
    if obj == 0:
        return
    
    if isinstance(obj, etable.Table):
        dlg = etview.TableViewDialog(vp, obj, giv.DlgOpts(Title=title), go.nil, go.nil)
    elif isinstance(obj, eplot.Plot2D):
        dlg = etview.Plot2DDialog(vp, obj, giv.DlgOpts(Title=title), go.nil, go.nil)
    elif isinstance(obj, etensor.Tensor):
        dlg = etview.TensorGridDialog(vp, obj, giv.DlgOpts(Title=title), go.nil, go.nil)
    elif isinstance(obj, simat.SimMat):
        dlg = etview.SimMatGridDialog(vp, obj, giv.DlgOpts(Title=title), go.nil, go.nil)
    elif isinstance(obj, params.Sets):
        dlg = giv.SliceViewDialogNoStyle(vp, obj, giv.DlgOpts(Title=title), go.nil, go.nil)
    elif isinstance(obj, go.Slice_string):
        dlg = giv.SliceViewDialogNoStyle(vp, obj, giv.DlgOpts(Title=title), go.nil, go.nil)
    else:
        dlg = giv.StructViewDialog(vp, obj, giv.DlgOpts(Title=title), go.nil, go.nil)
    return dlg

def SetIntValCB(recv, send, sig, data):
    vw = gi.SpinBox(handle=send)
    nm = vw.Name()
    # print("spin name:", nm)
    nms = nm.split(':')
    cv = classviews[nms[0]]
    setattr(cv.Class, nms[1], vw.Value)

def EditGoObjCB(recv, send, sig, data):
    vw = gi.Action(handle=send)
    nm = vw.Name()
    nms = nm.split(':')
    cv = classviews[nms[0]]
    fld = getattr(cv.Class, nms[1])
    title = nms[1]
    # print("nm: %s  cv: %s  fld: %s" % (nm, cv, fld))
    return GoObjDialog(vw.Viewport, fld, title)

def EditObjCB(recv, send, sig, data):
    vw = gi.Action(handle=send)
    nm = vw.Name()
    nms = nm.split(':')
    cv = classviews[nms[0]]
    fld = getattr(cv.Class, nms[1])
    tags = cv.FieldTags(nms[1])
    nnm = nm.replace(":", "_")
    return ClassViewDialog(vw.Viewport, fld, nnm, tags, giv.DlgOpts(Title=nnm))

def SetStrValCB(recv, send, sig, data):
    if sig != gi.TextFieldDone:
        return
    vw = gi.TextField(handle=send)
    nm = vw.Name()
    nms = nm.split(':')
    cv = classviews[nms[0]]
    setattr(cv.Class, nms[1], vw.Text())

def SetBoolValCB(recv, send, sig, data):
    if sig != gi.ButtonToggled:
        return
    vw = gi.CheckBox(handle=send)
    nm = vw.Name()
    # print("cb name:", nm)
    nms = nm.split(':')
    cv = classviews[nms[0]]
    setattr(cv.Class, nms[1], vw.IsChecked() != 0)

##############
# Enums

def ItemsFromEnum(cb, enm):
    nms = []
    for en in type(enm):
        nms.append(en.name)
    cb.ItemsFromStringList(go.Slice_string(nms), False, 0)
    cb.SetCurVal(enm.name)
    
def SetEnumCB(recv, send, sig, data):
    vw = gi.ComboBox(handle=send)
    nm = vw.Name()
    nms = nm.split(':')
    idx = vw.CurIndex
    cv = classviews[nms[0]]
    flds = cv.Class.__dict__
    typ = type(flds[nms[1]])
    vl = typ(idx)
    setattr(cv.Class, nms[1], vl)
    
class ClassView(object):
    """
    PyGiClassView provides giv.StructView like editor for python class objects under GoGi.
    Due to limitations on calling python callbacks across threads, you must pass a unique
    name to the constructor, along with a dictionary of tags using the same syntax as the struct
    field tags in Go: https://github.com/goki/gi/wiki/Tags for customizing the view properties.
    (space separated, name:"value")
    """
    def __init__(self, name, tags):
        """ note: essential to provide a distinctive name for each view """
        self.Name = name
        classviews[name] = self
        self.Frame = gi.Frame()
        self.Class = None
        self.Tags = tags
        
    def SetClass(self, cls):
        self.Class = cls
        self.Config()
        
    def AddFrame(self, par):
        """ Add a new gi.Frame for the view to given parent gi object """
        self.Frame = gi.Frame(par.AddNewChild(gi.KiT_Frame(), "classview"))
    
    def FieldTags(self, nm):
        """ returns the parsed dictonary of tags for given field """
        tdict = {}
        if nm in self.Tags:
            ts = self.Tags[nm].split(" ")
            for t in ts:
                nv = t.split(":")
                if len(nv) == 2:
                    tdict[nv[0]] = nv[1].strip('"')
                else:
                    print("ClassView: error in tag formatting for field:", nm, 'should be name:"value", is:', t)
        return tdict

    def Config(self):
        self.Frame.SetStretchMaxWidth()
        self.Frame.SetStretchMaxHeight()
        self.Frame.Lay = gi.LayoutGrid
        self.Frame.Stripes = gi.RowStripes
        self.Frame.SetPropInt("columns", 2)
        updt = self.Frame.UpdateStart()
        self.Frame.SetFullReRender()
        self.Frame.DeleteChildren(True)
        flds = self.Class.__dict__
        self.Views = {}
        for nm, val in flds.items():
            tags = self.FieldTags(nm)
            if HasTagValue(tags, "view", "-"):
                continue
            lbl = gi.Label(self.Frame.AddNewChild(gi.KiT_Label(), "lbl_" + nm))
            lbl.SetText(nm)
            vw = GoObjView(val, nm, self.Frame, self.Name, tags)
            self.Views[nm] = vw
        self.Frame.UpdateEnd(updt)
        
    def Update(self):
        flds = self.Class.__dict__
        for nm, val in flds.items():
            if nm in self.Views:
                vw = self.Views[nm]
                # print("updating:", nm, "view:", vw)
                GoObjUpdtView(val, vw, nm)

def ClassViewDialog(vp, obj, name, tags, opts):
    """
    ClassViewDialog returns a dialog with ClassView editor for python
    class objects under GoGi.
    opts must be a giv.DlgOpts instance
    """
    dlg = gi.NewStdDialog(opts.ToGiOpts(), opts.Ok, opts.Cancel)
    frame = dlg.Frame()
    prIdx = dlg.PromptWidgetIdx(frame)

    cv = ClassView(name, tags)
    cv.Frame = gi.Frame(frame.InsertNewChild(gi.KiT_Frame(), prIdx+1, "cv-frame"))
    cv.SetClass(obj)
    
    # sv.Viewport = dlg.Embed(gi.KiT_Viewport2D).(*gi.Viewport2D)
    # if opts.Inactive {
    #     sv.SetInactive()
    # }

    dlg.UpdateEndNoSig(True)
    dlg.Open(0, 0, vp, go.nil)
    return dlg

    
class ClassViewObj(object):
    """
    this is the base class for any user-defined class that should be displayed
    with its own separate ClassView dialog
    """
    def __init__(self):
        pass

    
