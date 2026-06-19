import { useState } from "react";
import type { Model } from "@modelvisio/parsers";
import { useT } from "../theme/ThemeContext";

export function DeployRecipe({ model }: { model: Model }) {
  const t = useT();
  const [hw, setHw] = useState("orin_nano");
  const modelFile = model?.name || "model.onnx";

  const recipes: Record<string, string> = {
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
  const targets = [{ id: "orin_nano", n: "Jetson Orin Nano" }, { id: "hailo8", n: "Hailo-8" }, { id: "rk3588", n: "RK3588" }];

  return <div style={{ display: "flex", flexDirection: "column", gap: 8, height: "100%" }}>
    <div style={{ fontSize: 13, fontWeight: 700, color: t.t0 }}>Deployment Recipe Generator</div>
    <div style={{ display: "flex", gap: 4 }}>{targets.map((tg) => <button key={tg.id} type="button" onClick={() => setHw(tg.id)} style={{ padding: "4px 10px", borderRadius: 4, border: `1px solid ${hw === tg.id ? t.acc3 : t.bdr}`, background: hw === tg.id ? t.acc3 + "18" : "transparent", color: hw === tg.id ? t.acc3 : t.t2, fontSize: 10, cursor: "pointer", fontWeight: 500 }}>{tg.n}</button>)}</div>
    <div style={{ flex: 1, overflow: "auto", position: "relative" }}>
      <button type="button" onClick={() => navigator.clipboard?.writeText(recipes[hw] || "")} style={{ position: "absolute", top: 6, right: 6, padding: "3px 8px", borderRadius: 3, border: `1px solid ${t.bdr}`, background: t.bg2, color: t.t2, fontSize: 9, cursor: "pointer", zIndex: 2 }}>Copy All</button>
      <pre style={{ background: t.bg, border: `1px solid ${t.bdr}`, borderRadius: 6, padding: 12, fontSize: 10.5, color: t.suc, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.5, whiteSpace: "pre-wrap", height: "100%", margin: 0, overflow: "auto" }}>{recipes[hw] || "Coming soon."}</pre>
    </div>
  </div>;
}
