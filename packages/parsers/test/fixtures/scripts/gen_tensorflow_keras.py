"""Generate the TensorFlow / Keras fixtures.

Run with a Python that has tensorflow-cpu installed (tested: TF 2.21.0,
Keras 3.12.4; optional tf_keras 2.x for the Keras 2 dialect):
    python gen_tensorflow_keras.py <out_dir>

Writes:
  keras/tiny_functional.keras   Keras 3 archive, functional model with a residual
                                Add, Concatenate, BN, depthwise/separable convs
  keras/tiny_functional.h5      same model, legacy HDF5 (Keras 3 config dialect)
  keras/tiny_sequential.keras   Keras 3 Sequential (Dense/LSTM/Embedding stack)
  keras/tiny_k2.h5              Keras 2 (tf_keras) functional HDF5, if available
  tensorflow/tiny_frozen.pb     frozen GraphDef of tiny_functional (vars → Const)
  tensorflow/tiny_frozen.pbtxt  the same GraphDef in text format
  tensorflow/saved_model.pb     TF2 SavedModel (tf.Module with a conv + dense)
  expected.json                 framework-reported param counts / shapes
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import keras  # noqa: E402
from keras import layers  # noqa: E402
from tensorflow.python.framework.convert_to_constants import (  # noqa: E402
    convert_variables_to_constants_v2,
)

out = sys.argv[1] if len(sys.argv) > 1 else "."
kdir = os.path.join(out, "keras")
tdir = os.path.join(out, "tensorflow")
os.makedirs(kdir, exist_ok=True)
os.makedirs(tdir, exist_ok=True)
keras.utils.set_random_seed(0)
expected = {"tf": tf.__version__, "keras": keras.__version__}


def functional():
    inp = keras.Input(shape=(32, 32, 3), name="image")
    x = layers.Conv2D(8, 3, padding="same", name="stem_conv")(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    y = layers.DepthwiseConv2D(3, padding="same", name="dw")(x)
    y = layers.Conv2D(8, 1, activation="relu", name="pw")(y)
    x = layers.Add(name="residual")([x, y])
    x = layers.MaxPooling2D(2, name="pool")(x)
    a = layers.SeparableConv2D(8, 3, padding="same", name="sep")(x)
    b = layers.Conv2D(4, 3, strides=2, padding="valid", name="strided")(x)
    b = layers.UpSampling2D(2, name="up")(b)
    b = layers.ZeroPadding2D(((0, 2), (0, 2)), name="zpad")(b)
    x = layers.Concatenate(name="concat")([a, b])
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.1, name="drop")(x)
    o = layers.Dense(10, activation="softmax", name="logits")(x)
    return keras.Model(inp, o, name="tiny_functional")


m = functional()
m.save(os.path.join(kdir, "tiny_functional.keras"))
m.save(os.path.join(kdir, "tiny_functional.h5"))
expected["tiny_functional"] = {
    "params": int(m.count_params()),
    "layers": [l.name for l in m.layers],
    "per_layer": {l.name: int(l.count_params()) for l in m.layers},
    "output_shapes": {l.name: list(l.output.shape) for l in m.layers},
}

seq = keras.Sequential(
    [
        keras.Input(shape=(12,), dtype="int32"),
        layers.Embedding(100, 16, name="emb"),
        layers.LSTM(8, return_sequences=True, name="lstm"),
        layers.Bidirectional(layers.GRU(4), name="bigru"),
        layers.Dense(3, name="out"),
    ],
    name="tiny_sequential",
)
seq.save(os.path.join(kdir, "tiny_sequential.keras"))
expected["tiny_sequential"] = {
    "params": int(seq.count_params()),
    "per_layer": {l.name: int(l.count_params()) for l in seq.layers},
    "output_shapes": {l.name: list(l.output.shape) for l in seq.layers},
}

# Frozen GraphDef of the functional model (variables folded into Const nodes).
fn = tf.function(lambda x: m(x, training=False))
cf = fn.get_concrete_function(tf.TensorSpec([1, 32, 32, 3], tf.float32, name="image"))
frozen = convert_variables_to_constants_v2(cf)
gd = frozen.graph.as_graph_def()
with open(os.path.join(tdir, "tiny_frozen.pb"), "wb") as f:
    f.write(gd.SerializeToString())
tf.io.write_graph(gd, tdir, "tiny_frozen.pbtxt", as_text=True)
expected["tiny_frozen"] = {
    "ops": [n.op for n in gd.node],
    "names": [n.name for n in gd.node],
    "params": int(m.count_params()),
    "output": frozen.outputs[0].name,
    "output_shape": frozen.outputs[0].shape.as_list(),
}


class TinyModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.k = tf.Variable(tf.random.normal([3, 3, 3, 4]), name="conv_kernel")
        self.b = tf.Variable(tf.zeros([4]), name="conv_bias")
        self.w = tf.Variable(tf.random.normal([4, 5]), name="dense_w")

    @tf.function(input_signature=[tf.TensorSpec([None, 16, 16, 3], tf.float32, name="x")])
    def __call__(self, x):
        y = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, self.k, 1, "SAME"), self.b))
        y = tf.reduce_mean(y, axis=[1, 2])
        return {"probs": tf.nn.softmax(tf.matmul(y, self.w))}


mod = TinyModule()
sm_dir = os.path.join(out, "_savedmodel")
tf.saved_model.save(mod, sm_dir, signatures={"serving_default": mod.__call__})
with open(os.path.join(sm_dir, "saved_model.pb"), "rb") as f:
    sm = f.read()
with open(os.path.join(tdir, "saved_model.pb"), "wb") as f:
    f.write(sm)
expected["saved_model"] = {"params": 3 * 3 * 3 * 4 + 4 + 4 * 5}

try:
    import tf_keras as k2

    i2 = k2.Input(shape=(8, 8, 1), name="inp")
    c = k2.layers.Conv2D(4, 3, name="c1")(i2)
    c = k2.layers.BatchNormalization(name="bn")(c)
    f1 = k2.layers.Flatten(name="flat")(c)
    d1 = k2.layers.Dense(6, activation="relu", name="d1")(f1)
    d2 = k2.layers.Dense(6, name="d2")(f1)
    s = k2.layers.Add(name="sum")([d1, d2])
    o2 = k2.layers.Dense(2, name="head")(s)
    m2 = k2.Model(i2, o2, name="k2_model")
    m2.save(os.path.join(kdir, "tiny_k2.h5"))
    expected["tiny_k2"] = {
        "params": int(m2.count_params()),
        "keras": k2.__version__,
        "per_layer": {l.name: int(l.count_params()) for l in m2.layers},
    }
except ImportError:
    print("tf_keras not installed; skipping the Keras 2 fixture")

with open(os.path.join(out, "expected.json"), "w") as f:
    json.dump(expected, f, indent=1)
for d in (kdir, tdir):
    for n in sorted(os.listdir(d)):
        print(d, n, os.path.getsize(os.path.join(d, n)))
