"""Generate ncnn fixtures from the shared TinyNet.

Requires: torch, pnnx (pip install pnnx ncnn).
pnnx traces the TorchScript module and writes ncnn .param (text) + .bin:
  tinynet.ncnn.param / tinynet.ncnn.bin   straight pnnx output (Input has no dims)
  tinynet_shaped.ncnn.param               same graph with Input dims 0=w 1=h 2=c set
                                          (the form classic onnx2ncnn/caffe2ncnn emit)
  tinynet_shaped.ncnn.param.bin           binary encoding of the shaped param, using
                                          the ncnn2mem layout (magic 7767517 as int32,
                                          layer type index, blob indexes, id/value pairs, -233)
Run from anywhere: python make_ncnn.py
"""
import os
import shutil
import struct
import sys
import tempfile

import subprocess
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tinynet import EXAMPLE, build  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "ncnn")
os.makedirs(OUT, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    pt = os.path.join(tmp, "tinynet.pt")
    torch.jit.trace(build(), EXAMPLE).save(pt)
    # The pnnx CLI (installed by the pnnx wheel) is more robust on Windows
    # than pnnx.convert(), whose generated Python breaks on backslash paths.
    subprocess.run(["pnnx", pt, "inputshape=[1,3,32,32]", "fp16=0"], cwd=tmp, check=True)
    for f in ("tinynet.ncnn.param", "tinynet.ncnn.bin"):
        shutil.copy(os.path.join(tmp, f), os.path.join(OUT, f))

# ncnn layer_type_enum order (src/CMakeLists.txt ncnn_add_layer order), prefix.
LAYER_TYPES = (
    "AbsVal ArgMax BatchNorm Bias BNLL Concat Convolution Crop Deconvolution Dropout Eltwise ELU Embed Exp "
    "Flatten InnerProduct Input Log LRN MemoryData MVN Pooling Power PReLU Proposal Reduction ReLU Reshape "
    "ROIPooling Scale Sigmoid Slice Softmax Split SPP TanH Threshold Tile RNN LSTM BinaryOp UnaryOp "
    "ConvolutionDepthWise Padding Squeeze ExpandDims Normalize Permute PriorBox DetectionOutput Interp"
).split()


def to_param_bin(text):
    lines = [l.split() for l in text.strip().splitlines()]
    layer_count, blob_count = map(int, lines[1])
    blobs = {}
    out = [struct.pack("<iii", 7767517, layer_count, blob_count)]
    for cols in lines[2:]:
        typ, _name, nin, nout = cols[0], cols[1], int(cols[2]), int(cols[3])
        ins, outs = cols[4:4 + nin], cols[4 + nin:4 + nin + nout]
        for b in ins + outs:
            blobs.setdefault(b, len(blobs))
        out.append(struct.pack("<iii", LAYER_TYPES.index(typ), nin, nout))
        out.append(struct.pack(f"<{nin + nout}i", *[blobs[b] for b in ins + outs]))
        for kv in cols[4 + nin + nout:]:
            k, v = kv.split("=")
            k = int(k)
            if k <= -23300:
                vals = v.split(",")[1:]
                out.append(struct.pack("<ii", k, len(vals)))
                for x in vals:
                    out.append(struct.pack("<f", float(x)) if ("." in x or "e" in x) else struct.pack("<i", int(x)))
            else:
                out.append(struct.pack("<i", k))
                out.append(struct.pack("<f", float(v)) if ("." in v or "e" in v) else struct.pack("<i", int(v)))
        out.append(struct.pack("<i", -233))
    return b"".join(out)


with open(os.path.join(OUT, "tinynet.ncnn.param")) as f:
    text = f.read()
shaped = text.replace(" 0 1 in0\n", " 0 1 in0 0=32 1=32 2=3\n", 1)
with open(os.path.join(OUT, "tinynet_shaped.ncnn.param"), "w", newline="\n") as f:
    f.write(shaped)
with open(os.path.join(OUT, "tinynet_shaped.ncnn.param.bin"), "wb") as f:
    f.write(to_param_bin(shaped))
print("wrote", os.listdir(OUT))
