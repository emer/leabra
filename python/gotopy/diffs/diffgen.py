#!/usr/local/bin/python3

# Copyright (c) 2020, The Emergent Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os
import diff_match_patch as dmp_module

dmp = dmp_module.diff_match_patch()

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
    
def json_dump(fnm, val):
    with open(fnm,"w") as f:
        json.dump(f, val)
    
src = read_as_string("ra25-p1-src.py")
trg = read_as_string("ra25-p1-trg.py")

patch = dmp.patch_make(src, trg)

raw = read_as_string("ra25-raw.py")

fixes = dmp.patch_apply(patch, raw)

txt = fixes[0]

txt = txt.replace("leabra.LeabraLayer(", "leabra.Layer(")
txt = txt.replace(".AsLeabra()", "")

write_string("ra25-p1-out.py", txt)

