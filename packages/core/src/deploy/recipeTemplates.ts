// Quick-start deploy recipes shown by the DeployRecipe panel, extracted from the
// component (unchanged text) so the deployment agent can reuse them headlessly.

export const DEPLOY_RECIPE_TARGETS: { id: string; n: string }[] = [
  { id: "orin_nano", n: "Jetson Orin Nano" },
  { id: "hailo8", n: "Hailo-8" },
  { id: "rk3588", n: "RK3588" },
];

/** All quick-start recipes for a model file, keyed by hardware id. */
export function deployRecipes(modelFile: string): Record<string, string> {
  return {
    orin_nano: `# --- Jetson Orin Nano Deploy Package ---
# 1. Export to TensorRT
trtexec --onnx=${modelFile} --saveEngine=model.engine --fp16 --workspace=4096

# 2. Dockerfile
cat > Dockerfile << 'EOF'
FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime
COPY model.engine /app/model.engine
COPY inference.py /app/inference.py
WORKDIR /app
RUN pip install opencv-python-headless numpy
CMD ["python3", "inference.py"]
EOF

# 3. inference.py
cat > inference.py << 'PYEOF'
import tensorrt as trt
import numpy as np, cv2, pycuda.driver as cuda, pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("model.engine","rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
# ... allocate buffers, run inference
print("YOLO26 running on Jetson Orin Nano")
PYEOF

# 4. Benchmark
trtexec --loadEngine=model.engine --warmUp=1000 --iterations=1000 --avgRuns=100`,
    hailo8: `# --- Hailo-8 Deploy Package ---
pip install hailo_sdk_client

# 1. Convert with Hailo DFC
hailo parser onnx ${modelFile} --end-node-names output0
hailo optimize --hw-arch hailo8 --batch-size 1
hailo compile --hw-arch hailo8

# 2. Run with HailoRT
cat > inference.py << 'EOF'
from hailo_platform import HailoRT
target = HailoRT.Device()
hef = HailoRT.HEF("model.hef")
network_group = target.configure(hef)
# ... run inference
print("YOLO26 running on Hailo-8")
EOF`,
    rk3588: `# --- RK3588 NPU Deploy ---
pip install rknn-toolkit2

# 1. Convert
python3 << 'EOF'
from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
rknn.load_onnx(model="${modelFile}")
rknn.build(do_quantization=True, dataset="calib_list.txt")
rknn.export_rknn("model.rknn")
EOF

# 2. Deploy
adb push model.rknn /data/
adb shell "cd /data && rknn_yolo_demo model.rknn test.jpg"`,
  };
}

/** The quick-start recipe for one hardware id, or undefined if none exists. */
export function deployRecipeText(hw: string, modelFile: string): string | undefined {
  return deployRecipes(modelFile)[hw];
}
