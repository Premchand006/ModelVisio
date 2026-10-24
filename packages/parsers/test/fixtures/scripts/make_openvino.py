"""Generate OpenVINO IR fixtures (tinynet.xml/.bin, fp32 and fp16-compressed).

Requires: torch, onnx, openvino (pip install openvino).
Run from anywhere: python make_openvino.py
"""
import os
import sys
import tempfile

import openvino as ov
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tinynet import EXAMPLE, build  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "openvino")
os.makedirs(OUT, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    onnx_path = os.path.join(tmp, "tinynet.onnx")
    torch.onnx.export(build(), EXAMPLE, onnx_path, input_names=["input"], output_names=["probs"], opset_version=13)
    model = ov.convert_model(onnx_path)
    ov.save_model(model, os.path.join(OUT, "tinynet.xml"), compress_to_fp16=False)
    ov.save_model(model, os.path.join(OUT, "tinynet_fp16.xml"), compress_to_fp16=True)
print("wrote", os.listdir(OUT))
