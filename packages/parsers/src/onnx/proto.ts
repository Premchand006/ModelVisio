import protobuf from "protobufjs";

// A faithful subset of onnx.proto (proto3). Field numbers MUST match the
// official ONNX spec — they are stable across opsets, so decoding any real
// .onnx file works. We include only the messages the parser reads.
//
// Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
const ONNX_PROTO = `
syntax = "proto3";
package onnx;

message StringStringEntryProto {
  string key = 1;
  string value = 2;
}

message TensorShapeProto {
  message Dimension {
    int64 dim_value = 1;
    string dim_param = 2;
    string denotation = 3;
  }
  repeated Dimension dim = 1;
}

message TypeProto {
  message Tensor {
    int32 elem_type = 1;
    TensorShapeProto shape = 2;
  }
  Tensor tensor_type = 1;
}

message TensorProto {
  enum DataType {
    UNDEFINED = 0;
    FLOAT = 1;
    UINT8 = 2;
    INT8 = 3;
    UINT16 = 4;
    INT16 = 5;
    INT32 = 6;
    INT64 = 7;
    STRING = 8;
    BOOL = 9;
    FLOAT16 = 10;
    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;
    COMPLEX128 = 15;
    BFLOAT16 = 16;
  }
  enum DataLocation {
    DEFAULT = 0;
    EXTERNAL = 1;
  }
  message Segment {
    int64 begin = 1;
    int64 end = 2;
  }
  repeated int64 dims = 1;
  int32 data_type = 2;
  Segment segment = 3;
  repeated float float_data = 4 [packed = true];
  repeated int32 int32_data = 5 [packed = true];
  repeated bytes string_data = 6;
  repeated int64 int64_data = 7 [packed = true];
  string name = 8;
  bytes raw_data = 9;
  repeated double double_data = 10 [packed = true];
  repeated uint64 uint64_data = 11 [packed = true];
  string doc_string = 12;
  repeated StringStringEntryProto external_data = 13;
  DataLocation data_location = 14;
}

message SparseTensorProto {
  TensorProto values = 1;
  TensorProto indices = 2;
  repeated int64 dims = 3;
}

message AttributeProto {
  enum AttributeType {
    UNDEFINED = 0;
    FLOAT = 1;
    INT = 2;
    STRING = 3;
    TENSOR = 4;
    GRAPH = 5;
    FLOATS = 6;
    INTS = 7;
    STRINGS = 8;
    TENSORS = 9;
    GRAPHS = 10;
    SPARSE_TENSOR = 11;
    SPARSE_TENSORS = 12;
    TYPE_PROTO = 13;
    TYPE_PROTOS = 14;
  }
  string name = 1;
  float f = 2;
  int64 i = 3;
  bytes s = 4;
  TensorProto t = 5;
  GraphProto g = 6;
  repeated float floats = 7;
  repeated int64 ints = 8;
  repeated bytes strings = 9;
  repeated TensorProto tensors = 10;
  repeated GraphProto graphs = 11;
  string doc_string = 13;
  TypeProto tp = 14;
  repeated TypeProto type_protos = 15;
  AttributeType type = 20;
  string ref_attr_name = 21;
  SparseTensorProto sparse_tensor = 22;
  repeated SparseTensorProto sparse_tensors = 23;
}

message ValueInfoProto {
  string name = 1;
  TypeProto type = 2;
  string doc_string = 3;
}

message NodeProto {
  repeated string input = 1;
  repeated string output = 2;
  string name = 3;
  string op_type = 4;
  repeated AttributeProto attribute = 5;
  string doc_string = 6;
  string domain = 7;
}

message GraphProto {
  repeated NodeProto node = 1;
  string name = 2;
  repeated TensorProto initializer = 5;
  string doc_string = 10;
  repeated ValueInfoProto input = 11;
  repeated ValueInfoProto output = 12;
  repeated ValueInfoProto value_info = 13;
  repeated SparseTensorProto sparse_initializer = 15;
}

message OperatorSetIdProto {
  string domain = 1;
  int64 version = 2;
}

message ModelProto {
  int64 ir_version = 1;
  string producer_name = 2;
  string producer_version = 3;
  string domain = 4;
  int64 model_version = 5;
  string doc_string = 6;
  GraphProto graph = 7;
  repeated OperatorSetIdProto opset_import = 8;
  repeated StringStringEntryProto metadata_props = 14;
}
`;

let cached: protobuf.Type | null = null;

/** Lazily compiles the embedded schema and returns the ModelProto type. */
export function modelProtoType(): protobuf.Type {
  if (!cached) {
    const root = protobuf.parse(ONNX_PROTO, { keepCase: true }).root;
    cached = root.lookupType("onnx.ModelProto");
  }
  return cached;
}
