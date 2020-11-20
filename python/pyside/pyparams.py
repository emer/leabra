# Copyright (c) 2019, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from leabra import go, params

def ApplyParams(cls, sheet, setMsg):
    """
    ApplyParams applies params.Sheet to cls
    """
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
                

