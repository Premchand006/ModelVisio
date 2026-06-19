// Conversion-script templates for the Converter panel. Each takes the current
// option set and returns a copy-pasteable script.
export type ConvOpts = {
  opset: number; sz: number; dyn: boolean; i8: boolean; fp16: boolean;
  inp: string; out: string; prec: string; plat: string;
};

export const CONV_SCRIPTS: Record<string, (o: ConvOpts) => string> = {
  "pt→onnx": (o) => `from ultralytics import YOLO\nmodel = YOLO("${o.inp || "yolo26n.pt"}")\nmodel.export(format="onnx", opset=${o.opset || 18}, simplify=True, imgsz=${o.sz || 640}, dynamic=${o.dyn ? "True" : "False"})`,
  "pt→tflite": (o) => `from ultralytics import YOLO\nmodel = YOLO("${o.inp || "yolo26n.pt"}")\nmodel.export(format="tflite", imgsz=${o.sz || 640}, int8=${o.i8 ? "True" : "False"})`,
  "onnx→tensorrt": (o) => `# TensorRT conversion\nimport subprocess\ncmd = ["trtexec", f"--onnx=${o.inp || "model.onnx"}", f"--saveEngine=${o.out || "model.engine"}", "--${o.prec || "fp16"}", "--workspace=4096"${o.dyn ? ', "--minShapes=images:1x3x640x640", "--optShapes=images:1x3x640x640", "--maxShapes=images:8x3x640x640"' : ""}]\nsubprocess.run(cmd, check=True)\nprint("✓ TensorRT engine built")`,
  "onnx→openvino": (o) => `from openvino.tools import mo\nfrom openvino.runtime import serialize\nmodel = mo.convert_model("${o.inp || "model.onnx"}", compress_to_fp16=True, input_shape=[1,3,${o.sz || 640},${o.sz || 640}])\nserialize(model, "${o.out || "model.xml"}")\nprint("✓ OpenVINO IR saved")`,
  "onnx→rknn": (o) => `from rknn.api import RKNN\nrknn = RKNN()\nrknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform="${o.plat || "rk3588"}")\nrknn.load_onnx(model="${o.inp || "model.onnx"}")\nrknn.build(do_quantization=${o.i8 ? "True" : "False"}${o.i8 ? ', dataset="calib_list.txt"' : ""})\nrknn.export_rknn("${o.out || "model.rknn"}")\nprint("✓ RKNN exported")`,
  "onnx→ncnn": (o) => `# ONNX → ncnn\nimport subprocess\nsubprocess.run(["onnx2ncnn", "${o.inp || "model.onnx"}", "${o.out || "model"}.param", "${o.out || "model"}.bin"], check=True)\n# Optimize\nsubprocess.run(["ncnnoptimize", "${o.out || "model"}.param", "${o.out || "model"}.bin", "${o.out || "model"}_opt.param", "${o.out || "model"}_opt.bin", "${o.prec === "fp16" ? "1" : "0"}"], check=True)\nprint("✓ ncnn model exported")`,
  "pt→coreml": (o) => `from ultralytics import YOLO\nmodel = YOLO("${o.inp || "yolo26n.pt"}")\nmodel.export(format="coreml", imgsz=${o.sz || 640}, half=${o.fp16 ? "True" : "False"}, nms=True)\nprint("✓ CoreML .mlpackage exported")`,
  "onnx→onnx-fp16": (o) => `# Proper FP16 ONNX (Cast nodes at IO boundaries)\nimport onnx\nfrom onnxconverter_common import float16\nm = onnx.load("${o.inp || "model.onnx"}")\nm16 = float16.convert_float_to_float16(m, keep_io_types=True)\nonnx.save(m16, "${o.out || "model_fp16.onnx"}")\nprint("✓ FP16 ONNX saved")`,
  "onnx→tfjs": (o) => `# ONNX → TensorFlow SavedModel → TF.js\nimport subprocess\nsubprocess.run(["onnx2tf", "-i", "${o.inp || "model.onnx"}", "-o", "saved_model"], check=True)\nsubprocess.run(["tensorflowjs_converter", "--input_format=tf_saved_model", "saved_model", "tfjs_model"], check=True)\nprint("✓ TF.js model in tfjs_model/")`,
  "pt→torchscript": (o) => `from ultralytics import YOLO\nmodel = YOLO("${o.inp || "yolo26n.pt"}")\nmodel.export(format="torchscript", imgsz=${o.sz || 640})\nprint("✓ TorchScript exported")`,
  "pt→executorch": (o) => `from ultralytics import YOLO\nmodel = YOLO("${o.inp || "yolo26n.pt"}")\nmodel.export(format="executorch", imgsz=${o.sz || 640})\nprint("✓ ExecuTorch package exported")`,
};
