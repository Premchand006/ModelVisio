"""Generate the MNN fixture (tinynet.mnn) from the shared TinyNet via ONNX.

Requires: torch, onnx, MNN (pip install --no-deps MNN; provides `mnnconvert`).
Run from anywhere: python make_mnn.py
"""
import os
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from tinynet import EXAMPLE, build  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "mnn")
os.makedirs(OUT, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    onnx_path = os.path.join(tmp, "tinynet.onnx")
    torch.onnx.export(build(), EXAMPLE, onnx_path, input_names=["input"], output_names=["probs"], opset_version=13)
    subprocess.run(
        ["mnnconvert", "-f", "ONNX", "--modelFile", onnx_path, "--MNNModel", os.path.join(OUT, "tinynet.mnn"), "--bizCode", "modelvisio"],
        check=True,
    )
print("wrote", os.listdir(OUT))
