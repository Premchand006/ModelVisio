"""Generate MLIR fixtures (StableHLO) by lowering a tiny JAX CNN.

Requires: jax[cpu] (use a separate venv; it may want a newer numpy than torch).
Run from anywhere: python make_mlir.py

Model: conv3x3(3->4, SAME) + bias -> relu -> flatten -> dense(4*8*8 -> 10) + bias,
input 1x3x8x8 (NCHW), weights baked in as constants (seed 0).
Writes tiny_cnn.stablehlo.mlir (pretty form) and tiny_cnn.generic.mlir (generic form).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "mlir")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0)
W1 = jnp.asarray(rng.standard_normal((4, 3, 3, 3)).astype(np.float32) * 0.1)
B1 = jnp.asarray(rng.standard_normal((4,)).astype(np.float32) * 0.1)
W2 = jnp.asarray(rng.standard_normal((4 * 8 * 8, 10)).astype(np.float32) * 0.05)
B2 = jnp.asarray(rng.standard_normal((10,)).astype(np.float32) * 0.1)


def model(x):
    y = jax.lax.conv_general_dilated(x, W1, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))
    y = jax.nn.relu(y + B1[None, :, None, None])
    y = y.reshape((1, -1))
    return y @ W2 + B2


lowered = jax.jit(model).lower(jnp.zeros((1, 3, 8, 8), jnp.float32))
pretty = lowered.as_text()
with open(os.path.join(OUT, "tiny_cnn.stablehlo.mlir"), "w", newline="\n") as f:
    f.write(pretty)
ir = lowered.compiler_ir("stablehlo")
generic = ir.operation.get_asm(print_generic_op_form=True, large_elements_limit=None)
with open(os.path.join(OUT, "tiny_cnn.generic.mlir"), "w", newline="\n") as f:
    f.write(generic)
print("wrote", os.listdir(OUT))
