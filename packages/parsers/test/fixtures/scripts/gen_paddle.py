"""Generate the tiny PaddlePaddle fixture (legacy ProgramDesc .pdmodel).

Run with a Python that has paddlepaddle installed (tested with 3.3.1, CPU):
    python gen_paddle.py <out_dir>
Paddle 3.x saves the PIR .json format by default; FLAGS_enable_pir_api=0
forces the classic protobuf ProgramDesc that parsePaddle reads.
Writes tiny_cnn.pdmodel (+ .pdiparams) and tiny_cnn.expected.json with the
framework-reported parameter count for the test to compare against.
"""
import json
import os
import sys

os.environ["FLAGS_enable_pir_api"] = "0"
import paddle  # noqa: E402
import paddle.nn as nn  # noqa: E402

paddle.seed(0)


class TinyCNN(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2D(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2D(2, 2)
        self.dw = nn.Conv2D(8, 8, 3, padding=1, groups=8)
        self.pw = nn.Conv2D(8, 16, 1)
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.pw(self.dw(x)))
        x = paddle.flatten(self.gap(x), 1)
        return nn.functional.softmax(self.fc(x))


out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
net = TinyCNN()
net.eval()
spec = [paddle.static.InputSpec([None, 3, 32, 32], "float32", "image")]
path = os.path.join(out_dir, "tiny_cnn")
paddle.jit.save(net, path, input_spec=spec)

# Total = every state tensor (incl. BN running mean/var), i.e. what .pdiparams holds.
total = sum(int(p.numel()) for p in net.state_dict().values())
trainable = sum(int(p.numel()) for p in net.parameters() if not p.stop_gradient)
with open(os.path.join(out_dir, "tiny_cnn.expected.json"), "w") as f:
    json.dump({"total_params": total, "trainable_params": trainable, "paddle": paddle.__version__}, f, indent=2)
print("total", total, "trainable", trainable, os.listdir(out_dir))
