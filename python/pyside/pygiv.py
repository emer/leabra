# Copyright (c) 2019, The GoKi Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, gi, giv, kit, units
from enum import Enum


class ClassViewObj(object):
    """
    ClassViewObj is the base class for Python-defined classes that support a GUI editor (View)
    that functions like the StructView in GoGi.  It maintains a dict of tags for each field
    that determine tooltips and other behavior for the field GUI representation.
    """

    def __init__(self):
        self.Tags = {}
        self.ClassView = 0
        self.ClassViewInline = 0
        self.ClassViewDialog = 0

    def SetTags(self, field, tags):
        self.Tags[field] = tags

    def NewClassView(self, name):
        self.ClassView = ClassView(self, name)
        return self.ClassView

    def UpdateClassView(self):
        if self.ClassView != 0:
            self.ClassView.Update()

    def NewClassViewInline(self, name):
        self.ClassViewInline = ClassViewInline(self, name)
        return self.ClassViewInline

    def UpdateClassViewInline(self):
        if self.ClassViewInline != 0:
            self.ClassViewInline.Update()

    def OpenViewDialog(self, vp, name, tags):
        """opens a new dialog window for this object, or if one already exists, raises it"""
        if self.ClassViewDialog != 0 and self.ClassViewDialog.Win.IsVisible():
            self.ClassViewDialog.Win.Raise()
            return
        self.ClassViewDialog = ClassViewDialog(
            vp, self, name, tags, views.DlgOpts(Title=name)
        )
        return self.ClassViewDialog


class ClassViewInline(object):
    """
    ClassViewInline provides GoGi views.StructViewInline like inline editor for
    python class objects under GoGi.
    Due to limitations on calling python callbacks across threads, you must pass a unique
    name to the constructor.  The object must be a ClassViewObj, with tags using same
    syntax as the struct field tags in Go: https://cogentcore.org/core/gi/wiki/Tags
    for customizing the view properties (space separated, name:"value")
    """

    def __init__(self, obj, name):
        """note: essential to provide a distinctive name for each view"""
        self.Class = obj
        self.Name = name
        classviews[name] = self
        self.Lay = 0
        self.Tags = obj.Tags
        self.Views = {}  # dict of ValueView reps of Go objs
        self.Widgets = {}  # dict of Widget reps of Python objs

    def FieldTags(self, field):
        """returns the full string of tags for given field, empty string if none"""
        if field in self.Tags:
            return self.Tags[field]
        return ""

    def FieldTagValue(self, field, key):
        """returns the value for given key in tags for given field, empty string if none"""
        return views.StructTagValue(key, self.FieldTags(field))

    def Config(self):
        self.Lay = core.Layout()
        self.Lay.InitName(self.Lay, self.Name)
        self.Lay.Lay = core.LayoutHoriz
        self.Lay.SetStretchMaxWidth()
        updt = self.Lay.UpdateStart()
        flds = self.Class.__dict__
        self.Views = {}
        self.Widgets = {}
        for nm, val in flds.items():
            tags = self.FieldTags(nm)
            if (
                HasTagValue(tags, "view", "-")
                or nm == "Tags"
                or nm.startswith("ClassView")
            ):
                continue
            lbl = core.Label(self.Lay.AddNewChild(core.KiT_Label(), "lbl_" + nm))
            lbl.Redrawable = True
            lbl.SetProperty("horizontal-align", "left")
            lbl.SetText(nm)
            dsc = self.FieldTagValue(nm, "desc")
            if dsc != "":
                lbl.Tooltip = dsc
            if isinstance(val, go.GoClass):
                fnm = self.Name + ":" + nm
                if kit.IfaceIsNil(val):
                    print(
                        "Field %s is Nil in ClassView for obj: %s"
                        % (fnm, str(self.Class))
                    )
                    continue
                vv = views.ToValueView(val, tags)
                views.SetSoloValueIface(vv, val)
                vw = self.Lay.AddNewChild(vv.WidgetType(), fnm)
                vv.ConfigWidget(vw)
                self.Views[nm] = vv
                self.Widgets[nm] = vw
                # todo: vv.ViewSig.Connect?
            else:
                vw = PyObjView(val, nm, self.Lay, self.Name, tags)
                self.Widgets[nm] = vw
        self.Lay.UpdateEnd(updt)

    def Update(self):
        updt = self.Lay.UpdateStart()
        flds = self.Class.__dict__
        for nm, val in flds.items():
            if nm in self.Views:
                vv = self.Views[nm]
                views.SetSoloValueIface(
                    vv, val
                )  # always update in case it might have changed
                vv.UpdateWidget()
            elif nm in self.Widgets:
                vw = self.Widgets[nm]
                PyObjUpdateView(val, vw, nm)
        self.Lay.UpdateEnd(updt)


class ClassView(object):
    """
    ClassView provides GoGi views.StructView like editor for python class objects under GoGi.
    Due to limitations on calling python callbacks across threads, you must pass a unique
    name to the constructor.  The object must be a ClassViewObj, with tags using same
    syntax as the struct field tags in Go: https://cogentcore.org/core/gi/wiki/Tags
    for customizing the view properties (space separated, name:"value")
    """

    def __init__(self, obj, name):
        """note: essential to provide a distinctive name for each view"""
        self.Class = obj
        self.Name = name
        classviews[name] = self
        self.Frame = 0
        self.Tags = obj.Tags
        self.Views = {}  # dict of ValueView reps of Go objs
        self.Widgets = {}  # dict of Widget reps of Python objs

    def AddFrame(self, par):
        """Add a new core.Frame for the view to given parent gi object"""
        self.Frame = core.Frame(par.AddNewChild(core.KiT_Frame(), "classview"))

    def FieldTags(self, field):
        """returns the full string of tags for given field, empty string if none"""
        if field in self.Tags:
            return self.Tags[field]
        return ""

    def FieldTagValue(self, field, key):
        """returns the value for given key in tags for given field, empty string if none"""
        return views.StructTagValue(key, self.FieldTags(field))

    def Config(self):
        self.Frame.SetStretchMaxWidth()
        self.Frame.SetStretchMaxHeight()
        self.Frame.Lay = core.LayoutGrid
        self.Frame.Stripes = core.RowStripes
        self.Frame.SetPropInt("columns", 2)
        updt = self.Frame.UpdateStart()
        self.Frame.SetFullReRender()
        self.Frame.DeleteChildren(True)
        flds = self.Class.__dict__
        self.Views = {}
        self.Widgets = {}
        for nm, val in flds.items():
            tags = self.FieldTags(nm)
            if (
                HasTagValue(tags, "view", "-")
                or nm == "Tags"
                or nm.startswith("ClassView")
            ):
                continue
            lbl = core.Label(self.Frame.AddNewChild(core.KiT_Label(), "lbl_" + nm))
            lbl.SetText(nm)
            dsc = self.FieldTagValue(nm, "desc")
            if dsc != "":
                lbl.Tooltip = dsc
            if isinstance(val, go.GoClass):
                fnm = self.Name + ":" + nm
                if kit.IfaceIsNil(val):
                    print(
                        "Field %s is Nil in ClassView for obj: %s"
                        % (fnm, str(self.Class))
                    )
                    continue
                vv = views.ToValueView(val, tags)
                views.SetSoloValueIface(vv, val)
                vw = self.Frame.AddNewChild(vv.WidgetType(), fnm)
                vv.ConfigWidget(vw)
                self.Views[nm] = vv
                self.Widgets[nm] = vw
                # todo: vv.ViewSig.Connect?
            else:
                vw = PyObjView(val, nm, self.Frame, self.Name, tags)
                self.Widgets[nm] = vw
        self.Frame.UpdateEnd(updt)

    def Update(self):
        updt = self.Frame.UpdateStart()
        flds = self.Class.__dict__
        for nm, val in flds.items():
            if nm in self.Views:
                vv = self.Views[nm]
                views.SetSoloValueIface(
                    vv, val
                )  # always update in case it might have changed
                vv.UpdateWidget()
            elif nm in self.Widgets:
                vw = self.Widgets[nm]
                PyObjUpdateView(val, vw, nm)
        self.Frame.UpdateEnd(updt)


def ClassViewDialog(vp, obj, name, tags, opts):
    """
    ClassViewDialog returns a dialog with ClassView editor for python
    class objects under GoGi.
    opts must be a views.DlgOpts instance
    """
    dlg = core.NewStdDialog(opts.ToGiOpts(), opts.Ok, opts.Cancel)
    frame = dlg.Frame()
    prIndex = dlg.PromptWidgetIndex(frame)

    cv = obj.NewClassView(name)
    cv.Frame = core.Frame(
        frame.InsertNewChild(core.KiT_Frame(), prIndex + 1, "cv-frame")
    )
    cv.Config()

    # sv.Viewport = dlg.Embed(core.KiT_Viewport2D).(*core.Viewport2D)
    # if opts.Inactive {
    #     sv.SetInactive()
    # }

    dlg.UpdateEndNoSig(True)
    dlg.Open(0, 0, vp, go.nil)
    return dlg


# classviews is a dictionary of classviews -- needed for callbacks
classviews = {}


def TagValue(tags, key):
    """returns tag value for given key"""
    return views.StructTagValue(key, tags)


def HasTagValue(tags, key, value):
    """returns true if given key has given value"""
    tval = views.StructTagValue(key, tags)
    return tval == value


def PyObjView(val, nm, frame, ctxt, tags):
    """
    PyObjView returns a core.Widget representing the given Python value,
    with given name.
    frame = core.Frame or layout to add widgets to -- also callback recv
    ctxt = context for this object (e.g., name of owning struct)
    """
    vw = 0
    fnm = ctxt + ":" + nm
    if isinstance(val, Enum):
        vw = core.AddNewComboBox(frame, fnm)
        vw.SetText(nm)
        vw.SetPropStr("padding", "2px")
        vw.SetPropStr("margin", "2px")
        ItemsFromEnum(vw, val)
        vw.ComboSig.Connect(frame, SetEnumCB)
    elif isinstance(val, ClassViewObj):
        if HasTagValue(tags, "view", "inline"):
            sv = val.NewClassViewInline(ctxt + "_" + nm)  # new full name
            sv.Config()
            frame.AddChild(sv.Lay)
            vw = sv.Lay
        else:
            vw = core.AddNewAction(frame, fnm)
            vw.SetText(nm)
            vw.SetPropStr("padding", "2px")
            vw.SetPropStr("margin", "2px")
            vw.SetPropStr("border-radius", "4px")
            vw.ActionSig.Connect(frame, EditObjCB)
    elif isinstance(val, bool):
        vw = core.AddNewCheckBox(frame, fnm)
        vw.SetChecked(val)
        vw.ButtonSig.Connect(frame, SetBoolValCB)
    elif isinstance(val, (int, float)):
        vw = core.AddNewSpinBox(frame, fnm)
        vw.SetValue(val)
        if isinstance(val, int):
            vw.SpinBoxSig.Connect(frame, SetIntValCB)
            vw.Step = 1
        else:
            vw.SpinBoxSig.Connect(frame, SetFloatValCB)
        mv = TagValue(tags, "min")
        if mv != "":
            vw.SetMin(float(mv))
        mv = TagValue(tags, "max")
        if mv != "":
            vw.SetMax(float(mv))
        mv = TagValue(tags, "step")
        if mv != "":
            vw.Step = float(mv)
        mv = TagValue(tags, "format")
        if mv != "":
            vw.Format = mv
    else:
        vw = core.AddNewTextField(frame, fnm)
        vw.SetText(str(val))
        vw.SetPropStr("min-width", "10em")
        vw.TextFieldSig.Connect(frame, SetStrValCB)
        mv = TagValue(tags, "width")
        if mv != "":
            vw.SetProperty("width", mv + "ch")
    if HasTagValue(tags, "inactive", "+"):
        vw.SetInactive()
    return vw


def PyObjUpdateView(val, vw, nm):
    """
    updates the given view widget for given value
    """
    if isinstance(val, Enum):
        if isinstance(vw, core.ComboBox):
            svw = core.ComboBox(vw)
            svw.SetCurValue(val.name)
        else:
            print("epygiv; Enum value: %s doesn't have ComboBox widget" % nm)
    elif isinstance(val, go.GoClass):
        pass
    elif isinstance(val, ClassViewObj):
        val.UpdateClassViewInline()
        val.UpdateClassView()
    elif isinstance(val, bool):
        if isinstance(vw, core.CheckBox):
            svw = core.CheckBox(vw)
            svw.SetChecked(val)
        else:
            print("epygiv; bool value: %s doesn't have CheckBox widget" % nm)
    elif isinstance(val, (int, float)):
        if isinstance(vw, core.SpinBox):
            svw = core.SpinBox(vw)
            svw.SetValue(val)
        else:
            print("epygiv; numerical value: %s doesn't have SpinBox widget" % nm)
    else:
        if isinstance(vw, core.TextField):
            tvw = core.TextField(vw)
            tvw.SetText(str(val))
        else:
            print(
                "epygiv; object %s = %s doesn't have expected TextField widget"
                % (nm, val)
            )


def SetIntValCB(recv, send, sig, data):
    vw = core.SpinBox(handle=send)
    nm = vw.Name()
    nms = nm.split(":")
    cv = classviews[nms[0]]
    setattr(cv.Class, nms[1], int(vw.Value))


def SetFloatValCB(recv, send, sig, data):
    vw = core.SpinBox(handle=send)
    nm = vw.Name()
    nms = nm.split(":")
    cv = classviews[nms[0]]
    setattr(cv.Class, nms[1], float(vw.Value))


def EditObjCB(recv, send, sig, data):
    vw = core.Action(handle=send)
    nm = vw.Name()
    nms = nm.split(":")
    cv = classviews[nms[0]]
    fld = getattr(cv.Class, nms[1])
    tags = cv.FieldTags(nms[1])
    nnm = nm.replace(":", "_")
    return fld.OpenViewDialog(vw.Viewport, nnm, tags)


def SetStrValCB(recv, send, sig, data):
    if sig != core.TextFieldDone:
        return
    vw = core.TextField(handle=send)
    nm = vw.Name()
    nms = nm.split(":")
    cv = classviews[nms[0]]
    setattr(cv.Class, nms[1], vw.Text())


def SetBoolValCB(recv, send, sig, data):
    if sig != core.ButtonToggled:
        return
    vw = core.CheckBox(handle=send)
    nm = vw.Name()
    # print("cb name:", nm)
    nms = nm.split(":")
    cv = classviews[nms[0]]
    setattr(cv.Class, nms[1], vw.IsChecked() != 0)


##############
# Enums


def ItemsFromEnum(cb, enm):
    nms = []
    typ = type(enm)
    nnm = (
        typ.__name__ + "N"
    )  # common convention of using the type name + N for last item in list
    for en in typ:
        if en.name != nnm:
            nms.append(en.name)
    cb.ItemsFromStringList(go.Slice_string(nms), False, 0)
    cb.SetCurValue(enm.name)


def SetEnumCB(recv, send, sig, data):
    vw = core.ComboBox(handle=send)
    nm = vw.Name()
    nms = nm.split(":")
    idx = vw.CurIndex
    cv = classviews[nms[0]]
    flds = cv.Class.__dict__
    typ = type(flds[nms[1]])
    vl = typ(idx)
    setattr(cv.Class, nms[1], vl)
