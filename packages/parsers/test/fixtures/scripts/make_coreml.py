"""Generate Core ML fixtures for TinyNet (see tinynet.py).

Writes to ../coreml/:
  tinynet_nn.mlmodel          NeuralNetwork (spec v4) via ct.convert(convert_to="neuralnetwork")
  tinynet_classifier.mlmodel  NeuralNetworkClassifier (ClassifierConfig, 10 labels)
  tinynet_mlprogram.mlmodel   ML Program (MIL) spec whose weights live in weight.bin
  tinynet.mlpackage.zip       the zipped .mlpackage directory (Manifest.json +
                              Data/com.apple.CoreML/model.mlmodel + weights/weight.bin)

Requires: torch, coremltools (tested with 9.0). On Windows/Linux coremltools ships
without its native MIL blob writer and ModelPackage libraries, so this script
provides a small pure-Python writer for the documented MIL blob storage format
(64-byte header, then per blob a 64-byte metadata record with sentinel
0xDEADBEEF followed by 64-byte aligned data) and assembles the .mlpackage itself.
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tinynet import EXAMPLE, build  # noqa: E402

import coremltools as ct  # noqa: E402
from coremltools.converters.mil.backend.mil import load as mil_load  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "coreml")
os.makedirs(OUT, exist_ok=True)

# MILBlob BlobDataType
DT = {"fp16": 1, "f32": 2, "u8": 3, "i8": 4, "bf16": 5, "i16": 6, "u16": 7, "i32": 14, "u32": 15}


class PyBlobWriter:
    """Pure-Python stand-in for coremltools.libmilstoragepython._BlobStorageWriter."""

    def __init__(self, path):
        self.path = path
        self.buf = bytearray(struct.pack("<II", 0, 2) + b"\0" * 56)  # count, version=2, reserved
        self.count = 0
        self._flush()

    def _write(self, dtype, arr):
        data = np.ascontiguousarray(arr).tobytes()
        while len(self.buf) % 64:
            self.buf += b"\0"
        meta = len(self.buf)
        data_off = meta + 64
        self.buf += struct.pack("<IIQQQ", 0xDEADBEEF, DT[dtype], len(data), data_off, 0) + b"\0" * 32
        self.buf += data
        self.count += 1
        struct.pack_into("<I", self.buf, 0, self.count)
        self._flush()
        return meta

    def _flush(self):
        with open(self.path, "wb") as f:
            f.write(self.buf)

    def write_float_data(self, a): return self._write("f32", a.astype(np.float32))
    def write_fp16_data(self, a): return self._write("fp16", a.astype(np.uint16))
    def write_uint8_data(self, a): return self._write("u8", a.astype(np.uint8))
    def write_int8_data(self, a): return self._write("i8", a.astype(np.int8))
    def write_int16_data(self, a): return self._write("i16", a.astype(np.int16))
    def write_uint16_data(self, a): return self._write("u16", a.astype(np.uint16))
    def write_int32_data(self, a): return self._write("i32", a.astype(np.int32))
    def write_uint32_data(self, a): return self._write("u32", a.astype(np.uint32))


class _Captured(Exception):
    pass


def main():
    traced = torch.jit.trace(build(), EXAMPLE)
    inputs = [ct.TensorType(name="input", shape=(1, 3, 32, 32))]

    # 1) NeuralNetwork
    nn = ct.convert(traced, inputs=inputs, convert_to="neuralnetwork")
    nn.save(os.path.join(OUT, "tinynet_nn.mlmodel"))

    # 2) NeuralNetworkClassifier
    labels = [f"class_{i}" for i in range(10)]
    clf = ct.convert(traced, inputs=inputs, convert_to="neuralnetwork", classifier_config=ct.ClassifierConfig(labels))
    clf.save(os.path.join(OUT, "tinynet_classifier.mlmodel"))

    # 3) ML Program: capture the proto + weights dir before coremltools tries to
    #    build the package with its (unavailable) native ModelPackage library.
    mil_load.BlobWriter = PyBlobWriter
    captured = {}

    def fake_create(proto, weights_dir=None, package_dir=None):
        captured["proto"] = proto
        captured["weights"] = os.path.join(weights_dir, "weight.bin")
        tmp = tempfile.mkdtemp()
        shutil.copy(captured["weights"], os.path.join(tmp, "weight.bin"))
        captured["weights"] = os.path.join(tmp, "weight.bin")
        raise _Captured()

    ct.models.model._create_mlpackage = fake_create
    ct.models.utils._create_mlpackage = fake_create
    try:
        ct.convert(traced, inputs=inputs, convert_to="mlprogram", skip_model_load=True)
    except _Captured:
        pass
    spec = captured["proto"]
    raw = spec.SerializeToString()
    with open(os.path.join(OUT, "tinynet_mlprogram.mlmodel"), "wb") as f:
        f.write(raw)
    with open(captured["weights"], "rb") as f:
        weights = f.read()

    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "7A1B2C3D-0000-4000-8000-000000000001": {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            },
            "7A1B2C3D-0000-4000-8000-000000000002": {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Weights",
                "name": "weights",
                "path": "com.apple.CoreML/weights",
            },
        },
        "rootModelIdentifier": "7A1B2C3D-0000-4000-8000-000000000001",
    }
    with zipfile.ZipFile(os.path.join(OUT, "tinynet.mlpackage.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("tinynet.mlpackage/Manifest.json", json.dumps(manifest, indent=2))
        z.writestr("tinynet.mlpackage/Data/com.apple.CoreML/model.mlmodel", raw)
        z.writestr("tinynet.mlpackage/Data/com.apple.CoreML/weights/weight.bin", weights)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
