// A hand-encoded, minimal ONNX model (Y = Identity(X), X: float32[1]). The
// benchmark creates a session from it first so that runtime start-up (fetching
// and compiling the WebAssembly binary, acquiring a WebGPU device or WebNN
// context) is measured separately from the user's model's session creation.
// Encoding it by hand keeps protobufjs out of the benchmark worker bundle.
// Pure: no React, no DOM.

const enc = new TextEncoder();

function varint(n: number): number[] {
  const out: number[] = [];
  let v = n;
  while (v > 0x7f) {
    out.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return out;
}

/** Protobuf field: wire type 0 (varint). */
const vint = (field: number, n: number) => [...varint((field << 3) | 0), ...varint(n)];
/** Protobuf field: wire type 2 (length-delimited bytes / string / message). */
const ld = (field: number, body: number[] | Uint8Array) => [...varint((field << 3) | 2), ...varint(body.length), ...body];
const str = (field: number, s: string) => ld(field, enc.encode(s));

/** ValueInfoProto { name, type: TypeProto { tensor_type { elem_type, shape { dim { dim_value } } } } } */
function valueInfo(name: string, elemType: number, dims: number[]): number[] {
  const shape = dims.flatMap((d) => ld(1, vint(1, d)));
  const tensorType = [...vint(1, elemType), ...ld(2, shape)];
  return [...str(1, name), ...ld(2, ld(1, tensorType))];
}

/** Build the bytes of `Y = Identity(X)` with float32[1] I/O. */
export function tinyIdentityModel(opset = 13): Uint8Array {
  const node = [...str(1, "X"), ...str(2, "Y"), ...str(3, "identity"), ...str(4, "Identity")];
  const graph = [...ld(1, node), ...str(2, "warmup"), ...ld(11, valueInfo("X", 1, [1])), ...ld(12, valueInfo("Y", 1, [1]))];
  const opsetId = [...str(1, ""), ...vint(2, opset)];
  const model = [...vint(1, 8 /* ir_version */), ...str(2, "modelvisio-bench"), ...ld(7, graph), ...ld(8, opsetId)];
  return new Uint8Array(model);
}
