// Shapes of the protobufjs-decoded ONNX messages we read. These mirror the
// proto in proto.ts. int64 fields arrive as `Long | number` — normalize with
// num()/numArr() before use.

export type Long = { toNumber(): number; low: number; high: number };
export type I64 = number | Long;

export interface RawDimension {
  dim_value?: I64;
  dim_param?: string;
}
export interface RawTensorShape {
  dim?: RawDimension[];
}
export interface RawTensorType {
  elem_type?: number;
  shape?: RawTensorShape;
}
export interface RawTypeProto {
  tensor_type?: RawTensorType;
}
export interface RawValueInfo {
  name?: string;
  type?: RawTypeProto;
}
export interface RawTensor {
  dims?: I64[];
  data_type?: number;
  float_data?: number[];
  int32_data?: number[];
  int64_data?: I64[];
  double_data?: number[];
  raw_data?: Uint8Array;
  name?: string;
  data_location?: number; // 0 = default (inline), 1 = external
  external_data?: { key?: string; value?: string }[];
}
export interface RawAttribute {
  name?: string;
  f?: number;
  i?: I64;
  s?: Uint8Array;
  t?: RawTensor;
  floats?: number[];
  ints?: I64[];
  strings?: Uint8Array[];
  type?: number;
}
export interface RawNode {
  input?: string[];
  output?: string[];
  name?: string;
  op_type?: string;
  attribute?: RawAttribute[];
  domain?: string;
}
export interface RawGraph {
  node?: RawNode[];
  name?: string;
  initializer?: RawTensor[];
  input?: RawValueInfo[];
  output?: RawValueInfo[];
  value_info?: RawValueInfo[];
}
export interface RawOpsetId {
  domain?: string;
  version?: I64;
}
export interface RawModel {
  ir_version?: I64;
  producer_name?: string;
  graph?: RawGraph;
  opset_import?: RawOpsetId[];
}

/** Normalize a protobuf int64 (Long or number) to a JS number. */
export function num(x: I64 | undefined | null): number {
  if (x == null) return 0;
  if (typeof x === "number") return x;
  if (typeof x.toNumber === "function") return x.toNumber();
  return Number(x);
}

export function numArr(xs: I64[] | undefined): number[] {
  return (xs ?? []).map(num);
}

// TensorProto.DataType → label used in the normalized model.
export const DT_NAME: Record<number, string> = {
  0: "undefined",
  1: "float32",
  2: "uint8",
  3: "int8",
  4: "uint16",
  5: "int16",
  6: "int32",
  7: "int64",
  8: "string",
  9: "bool",
  10: "float16",
  11: "float64",
  12: "uint32",
  13: "uint64",
  14: "complex64",
  15: "complex128",
  16: "bfloat16",
};

// Bytes per element for raw_data interpretation.
export const DT_BYTES: Record<number, number> = {
  1: 4, // float32
  2: 1, // uint8
  3: 1, // int8
  4: 2, // uint16
  5: 2, // int16
  6: 4, // int32
  7: 8, // int64
  9: 1, // bool
  10: 2, // float16
  11: 8, // float64
  12: 4, // uint32
  13: 8, // uint64
  16: 2, // bfloat16
};
