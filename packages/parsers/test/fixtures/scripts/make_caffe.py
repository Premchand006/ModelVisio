"""Generate Caffe fixtures for TinyNet (see tinynet.py) without needing Caffe.

Writes, next to the downloaded BVLC/SqueezeNet prototxts in ../caffe/:
  tinynet.prototxt       deploy graph in protobuf text format
  tinynet.caffemodel     binary NetParameter (modern `layer`, field 100) with the
                         real TinyNet weights (BN kept as BatchNorm + Scale)
  tinynet_v1.caffemodel  legacy V1 NetParameter (`layers`, field 2, enum types,
                         net-level input/input_dim, legacy num/channels/height/width blobs)

The protobuf wire encoding is hand-rolled from BVLC caffe.proto field numbers.
Requires: torch.
"""
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from tinynet import build  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "caffe")


# ── wire format ───────────────────────────────────────────────────────────────
def varint(n):
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def key(field, wt):
    return varint((field << 3) | wt)


def f_varint(field, v):
    return key(field, 0) + varint(int(v))


def f_bytes(field, b):
    return key(field, 2) + varint(len(b)) + b


def f_str(field, s):
    return f_bytes(field, s.encode())


def f_float(field, v):
    return key(field, 5) + struct.pack("<f", v)


def packed_floats(field, arr):
    return f_bytes(field, struct.pack("<%df" % len(arr), *arr))


def packed_varints(field, arr):
    return f_bytes(field, b"".join(varint(int(x)) for x in arr))


def blob_shape(dims):
    return packed_varints(1, dims)  # BlobShape.dim = 1 (packed int64)


def blob(t):
    flat = t.detach().flatten().tolist()
    return f_bytes(7, blob_shape(list(t.shape))) + packed_floats(5, flat)  # shape=7, data=5


def legacy_blob(t):
    dims = [1] * (4 - t.dim()) + list(t.shape)
    return (f_varint(1, dims[0]) + f_varint(2, dims[1]) + f_varint(3, dims[2]) + f_varint(4, dims[3])
            + packed_floats(5, t.detach().flatten().tolist()))


def layer(name, type_, bottom=(), top=(), blobs=(), params=b""):
    b = f_str(1, name) + f_str(2, type_)
    for x in bottom:
        b += f_str(3, x)
    for x in top:
        b += f_str(4, x)
    for t in blobs:
        b += f_bytes(7, blob(t))
    return f_bytes(100, b + params)


def conv_param(num_output, k, stride=1, pad=0):
    p = f_varint(1, num_output) + f_varint(4, k)
    if stride != 1:
        p += f_varint(6, stride)
    if pad:
        p += f_varint(3, pad)
    return f_bytes(106, p)


def main():
    m = build()
    sd = {k: v for k, v in m.state_dict().items()}
    ones = torch.ones(1)

    # ── modern caffemodel ──
    net = f_str(1, "TinyNet")
    net += layer("data", "Input", top=["data"], params=f_bytes(143, f_bytes(1, blob_shape([1, 3, 32, 32]))))
    net += layer("conv1", "Convolution", ["data"], ["conv1"], [sd["conv1.weight"], sd["conv1.bias"]], conv_param(8, 3, pad=1))
    net += layer("bn1", "BatchNorm", ["conv1"], ["conv1"], [sd["bn1.running_mean"], sd["bn1.running_var"], ones])
    net += layer("scale1", "Scale", ["conv1"], ["conv1"], [sd["bn1.weight"], sd["bn1.bias"]], f_bytes(142, f_varint(4, 1)))
    net += layer("relu1", "ReLU", ["conv1"], ["conv1"])
    net += layer("pool1", "Pooling", ["conv1"], ["pool1"], params=f_bytes(121, f_varint(1, 0) + f_varint(2, 2) + f_varint(3, 2)))
    net += layer("conv2", "Convolution", ["pool1"], ["conv2"], [sd["conv2.weight"], sd["conv2.bias"]], conv_param(8, 3, pad=1))
    net += layer("relu2", "ReLU", ["conv2"], ["conv2"])
    net += layer("add", "Eltwise", ["conv2", "pool1"], ["add"], params=f_bytes(110, f_varint(1, 1)))
    net += layer("conv3", "Convolution", ["add"], ["conv3"], [sd["conv3.weight"], sd["conv3.bias"]], conv_param(16, 1, stride=2))
    net += layer("relu3", "ReLU", ["conv3"], ["conv3"])
    net += layer("gap", "Pooling", ["conv3"], ["gap"], params=f_bytes(121, f_varint(1, 1) + f_varint(12, 1)))
    net += layer("fc", "InnerProduct", ["gap"], ["fc"], [sd["fc.weight"], sd["fc.bias"]], f_bytes(117, f_varint(1, 10)))
    net += layer("prob", "Softmax", ["fc"], ["prob"])
    with open(os.path.join(OUT, "tinynet.caffemodel"), "wb") as f:
        f.write(net)

    # ── legacy V1 caffemodel: conv1 -> relu -> pool -> fc -> softmax ──
    def v1(name, enum, bottom=(), top=(), blobs=(), params=b""):
        b = f_str(4, name) + f_varint(5, enum)
        for x in bottom:
            b += f_str(2, x)
        for x in top:
            b += f_str(3, x)
        for t in blobs:
            b += f_bytes(6, legacy_blob(t))
        return f_bytes(2, b + params)

    # The V1 fc reads the 8x16x16 pool output directly, so it needs its own 10x2048 weight.
    torch.manual_seed(1)
    fc_w = torch.randn(10, 8 * 16 * 16) * 0.01
    old = f_str(1, "TinyNetV1") + f_str(3, "data") + b"".join(f_varint(4, d) for d in [1, 3, 32, 32])
    old += v1("conv1", 4, ["data"], ["conv1"], [sd["conv1.weight"], sd["conv1.bias"]],
              f_bytes(10, f_varint(1, 8) + f_varint(4, 3) + f_varint(3, 1)))
    old += v1("relu1", 18, ["conv1"], ["conv1"])
    old += v1("pool1", 17, ["conv1"], ["pool1"], params=f_bytes(19, f_varint(1, 0) + f_varint(2, 2) + f_varint(3, 2)))
    old += v1("fc", 14, ["pool1"], ["fc"], [fc_w, sd["fc.bias"]], f_bytes(17, f_varint(1, 10)))
    old += v1("prob", 20, ["fc"], ["prob"])
    with open(os.path.join(OUT, "tinynet_v1.caffemodel"), "wb") as f:
        f.write(old)

    # ── deploy prototxt (same topology as tinynet.caffemodel) ──
    txt = '''name: "TinyNet"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 3 dim: 32 dim: 32 } } }
layer { name: "conv1" type: "Convolution" bottom: "data" top: "conv1"
  convolution_param { num_output: 8 kernel_size: 3 pad: 1 } }
layer { name: "bn1" type: "BatchNorm" bottom: "conv1" top: "conv1" }
layer { name: "scale1" type: "Scale" bottom: "conv1" top: "conv1" scale_param { bias_term: true } }
layer { name: "relu1" type: "ReLU" bottom: "conv1" top: "conv1" }
layer { name: "pool1" type: "Pooling" bottom: "conv1" top: "pool1" pooling_param { pool: MAX kernel_size: 2 stride: 2 } }
layer { name: "conv2" type: "Convolution" bottom: "pool1" top: "conv2"
  convolution_param { num_output: 8 kernel_size: 3 pad: 1 } }
layer { name: "relu2" type: "ReLU" bottom: "conv2" top: "conv2" }
layer { name: "add" type: "Eltwise" bottom: "conv2" bottom: "pool1" top: "add" eltwise_param { operation: SUM } }
layer { name: "conv3" type: "Convolution" bottom: "add" top: "conv3"
  convolution_param { num_output: 16 kernel_size: 1 stride: 2 } }
layer { name: "relu3" type: "ReLU" bottom: "conv3" top: "conv3" }
layer { name: "gap" type: "Pooling" bottom: "conv3" top: "gap" pooling_param { pool: AVE global_pooling: true } }
layer { name: "fc" type: "InnerProduct" bottom: "gap" top: "fc" inner_product_param { num_output: 10 } }
layer { name: "prob" type: "Softmax" bottom: "fc" top: "prob" }
'''
    with open(os.path.join(OUT, "tinynet.prototxt"), "w", newline="\n") as f:
        f.write(txt)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
