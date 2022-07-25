#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time, pdb, argparse, subprocess, pickle, os, gzip, glob, json

from SyncNetInstance import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description="SyncNet")
parser.add_argument(
    "--initial_model",
    type=str,
    default=f"{os.path.dirname(__file__)}/data/syncnet_v2.model",
    help="",
)
parser.add_argument("--batch_size", type=int, default="20", help="")
parser.add_argument("--vshift", type=int, default="15", help="")
parser.add_argument("--data_dir", type=str, default="data/work", help="")
parser.add_argument("--videofile", type=str, default="", help="")
parser.add_argument("--reference", type=str, default="", help="")
parser.add_argument("--output_path", type=str, default=None)
opt = parser.parse_args()

setattr(opt, "avi_dir", os.path.join(opt.data_dir, "pyavi"))
setattr(opt, "tmp_dir", os.path.join(opt.data_dir, "pytmp"))
setattr(opt, "work_dir", os.path.join(opt.data_dir, "pywork"))
setattr(opt, "crop_dir", os.path.join(opt.data_dir, "pycrop"))


# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance()

s.loadParameters(opt.initial_model)
print("Model %s loaded." % opt.initial_model)

flist = glob.glob(os.path.join(opt.crop_dir, opt.reference, "0*.avi"))
flist.sort()

# ==================== GET OFFSETS ====================

dists = []
for idx, fname in enumerate(flist):
    offset, conf, dist = s.evaluate(opt, videofile=fname)
    eval_dict = dict(offset=offset.item(), conf=conf.item())

    if opt.output_path is not None:
        with open(opt.output_path, "a") as f:
            f.write(f"{json.dumps(eval_dict)}\n")
    dists.append(dist)

# ==================== PRINT RESULTS TO FILE ====================

with open(os.path.join(opt.work_dir, opt.reference, "activesd.pckl"), "wb") as fil:
    pickle.dump(dists, fil)
