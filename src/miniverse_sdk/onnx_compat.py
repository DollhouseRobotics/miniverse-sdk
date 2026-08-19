"""Bounded, dependency-free TensorRT compatibility scan of ONNX checkpoints.

TensorRT and TensorRT-RTX deterministically reject some graphs at build time
that ONNX Runtime executes correctly, so a checkpoint can be valid for
Miniverse yet never receive its optimized derivatives. This module statically
resolves those documented builder limits from the serialized ModelProto
without loading tensor payloads into memory, letting user agents lint a
checkpoint on any machine before uploading it. The scan is a best-effort
pre-flight: the server-side compiler remains authoritative.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from io import BytesIO
from typing import Any, BinaryIO

from .onnx_metadata import (
    MAX_METADATA_ENTRY_BYTES,
    OnnxMetadataError,
    _discard,
    _entry_fields,
    _read_varint,
    validate_contract,
)

TENSORRT_MAX_TOPK_K = 3840
MAX_STRING_BYTES = 64 * 1024
MAX_GRAPH_DEPTH = 8
_INTEGER_TENSOR_FORMATS = {6: "<i", 7: "<q", 12: "<I", 13: "<Q"}
_KEPT_TENSOR_BYTES = 64


@dataclass(frozen=True)
class CompatFinding:
    code: str
    severity: str
    op_type: str
    node: str
    message: str
    hint: str


@dataclass(frozen=True)
class ModelScan:
    precision: str
    findings: tuple[CompatFinding, ...]


class _Bounded:
    """A strict window over an underlying stream; reads never cross the bound."""

    __slots__ = ("_source", "remaining")

    def __init__(self, source: Any, length: int):
        self._source = source
        self.remaining = length

    def read(self, count: int) -> bytes:
        if count <= 0 or not self.remaining:
            return b""
        chunk = self._source.read(min(count, self.remaining))
        if not chunk:
            raise OnnxMetadataError("ONNX graph is truncated")
        self.remaining -= len(chunk)
        return chunk


def _read_exact(source: Any, count: int) -> bytes:
    parts: list[bytes] = []
    remaining = count
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise OnnxMetadataError("ONNX graph is truncated")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _length(source: Any, maximum: int | None = None) -> int:
    value = _read_varint(source)
    assert value is not None
    if maximum is not None and value > maximum:
        raise OnnxMetadataError("ONNX record exceeds its bounded size")
    return value


def _string(source: Any) -> str:
    raw = _read_exact(source, _length(source, MAX_STRING_BYTES))
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OnnxMetadataError("ONNX identifier must be UTF-8") from error


def _skip_value(source: Any, wire: int) -> None:
    if wire == 0:
        _read_varint(source)
    elif wire == 1:
        _discard(source, 8)
    elif wire == 2:
        _discard(source, _length(source))
    elif wire == 5:
        _discard(source, 4)
    else:
        raise OnnxMetadataError(f"ONNX uses unsupported protobuf wire type {wire}")


def _packed_varints(source: Any, length: int) -> tuple[int, ...]:
    packed = BytesIO(_read_exact(source, length))
    values: list[int] = []
    while True:
        value = _read_varint(packed, allow_eof=True)
        if value is None:
            return tuple(values)
        values.append(value)


@dataclass(frozen=True)
class _Node:
    op_type: str
    label: str
    inputs: tuple[str, ...]
    attributes: dict[str, Any]


class _GraphState:
    """Nodes plus every small integer constant usable as a static operand.

    Constant names are collected across nested subgraphs into one flat map;
    ONNX requires graph-unique tensor names, and a shadowing collision in a
    subgraph could at worst misresolve a limit operand for a warning-level
    finding, never crash the scan.
    """

    def __init__(self) -> None:
        self.nodes: list[_Node] = []
        self.constants: dict[str, tuple[int, ...]] = {}


def _parse_tensor(source: _Bounded) -> tuple[str, tuple[int, ...] | None]:
    name = ""
    data_type = 0
    raw: bytes | None = None
    varint_values: list[int] = []
    while True:
        tag = _read_varint(source, allow_eof=True)
        if tag is None:
            break
        number, wire = tag >> 3, tag & 7
        if number == 2 and wire == 0:
            data_type = _length(source)
        elif number == 8 and wire == 2:
            name = _string(source)
        elif number in (5, 7, 11) and wire == 2:
            length = _length(source)
            if length <= _KEPT_TENSOR_BYTES * 10:
                varint_values.extend(_packed_varints(source, length))
            else:
                _discard(source, length)
        elif number in (5, 7, 11) and wire == 0:
            varint_values.append(_length(source))
        elif number == 9 and wire == 2:
            length = _length(source)
            if length <= _KEPT_TENSOR_BYTES:
                raw = _read_exact(source, length)
            else:
                _discard(source, length)
        else:
            _skip_value(source, wire)
    if raw is not None and data_type in _INTEGER_TENSOR_FORMATS:
        item = _INTEGER_TENSOR_FORMATS[data_type]
        if len(raw) % struct.calcsize(item) == 0:
            return name, tuple(value for (value,) in struct.iter_unpack(item, raw))
    if varint_values:
        return name, tuple(varint_values)
    return name, None


def _parse_attribute(source: _Bounded, state: _GraphState, depth: int) -> tuple[str, Any]:
    name = ""
    value: Any = None
    while True:
        tag = _read_varint(source, allow_eof=True)
        if tag is None:
            return name, value
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 2:
            name = _string(source)
        elif number == 3 and wire == 0:
            value = _length(source)
        elif number == 5 and wire == 2:
            _, tensor = _parse_tensor(_Bounded(source, _length(source)))
            value = tensor
        elif number in (6, 11) and wire == 2:
            _parse_graph(_Bounded(source, _length(source)), state, depth + 1)
        elif number == 8 and wire == 2:
            value = _packed_varints(source, _length(source, _KEPT_TENSOR_BYTES * 10))
        elif number == 8 and wire == 0:
            value = (*(value if isinstance(value, tuple) else ()), _length(source))
        else:
            _skip_value(source, wire)


def _parse_node(source: _Bounded, state: _GraphState, depth: int) -> None:
    inputs: list[str] = []
    outputs: list[str] = []
    op_type = ""
    node_name = ""
    attributes: dict[str, Any] = {}
    while True:
        tag = _read_varint(source, allow_eof=True)
        if tag is None:
            break
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 2:
            inputs.append(_string(source))
        elif number == 2 and wire == 2:
            outputs.append(_string(source))
        elif number == 3 and wire == 2:
            node_name = _string(source)
        elif number == 4 and wire == 2:
            op_type = _string(source)
        elif number == 5 and wire == 2:
            name, value = _parse_attribute(_Bounded(source, _length(source)), state, depth)
            if name:
                attributes[name] = value
        else:
            _skip_value(source, wire)
    state.nodes.append(_Node(op_type=op_type, label=node_name, inputs=tuple(inputs), attributes=attributes))
    if op_type == "Constant" and outputs:
        for key in ("value", "value_int", "value_ints"):
            value = attributes.get(key)
            if isinstance(value, int):
                value = (value,)
            if isinstance(value, tuple):
                state.constants[outputs[0]] = value
                break


def _parse_graph(source: _Bounded, state: _GraphState, depth: int) -> None:
    if depth > MAX_GRAPH_DEPTH:
        raise OnnxMetadataError("ONNX graph nesting exceeds the supported depth")
    while True:
        tag = _read_varint(source, allow_eof=True)
        if tag is None:
            return
        number, wire = tag >> 3, tag & 7
        if number == 1 and wire == 2:
            _parse_node(_Bounded(source, _length(source)), state, depth)
        elif number == 5 and wire == 2:
            name, values = _parse_tensor(_Bounded(source, _length(source)))
            if name and values is not None:
                state.constants[name] = values
        else:
            _skip_value(source, wire)


def _topk_findings(node: _Node, state: _GraphState, opset: int) -> list[CompatFinding]:
    label = node.label or "unnamed TopK node"
    k: int | None = None
    if opset and opset < 10:
        value = node.attributes.get("k")
        k = value if isinstance(value, int) else None
    elif len(node.inputs) > 1:
        values = state.constants.get(node.inputs[1])
        if isinstance(values, tuple) and len(values) == 1:
            k = values[0]
    if k is None:
        return [CompatFinding(
            code="tensorrt_topk_dynamic_k", severity="warning", op_type=node.op_type, node=label,
            message=f"TopK K is not statically resolvable; TensorRT requires K <= {TENSORRT_MAX_TOPK_K} at build time",
            hint="export K as a graph constant so the TensorRT compiler can verify it; see `miniverse agent-help onnx`",
        )]
    if k > TENSORRT_MAX_TOPK_K:
        return [CompatFinding(
            code="tensorrt_topk_k_limit", severity="warning", op_type=node.op_type, node=label,
            message=f"TopK K={k} exceeds the TensorRT maximum of {TENSORRT_MAX_TOPK_K}; TensorRT and TensorRT-RTX will reject this graph and Miniverse cannot derive optimized artifacts",
            hint="a full sort lowers to TopK over the entire axis; top-p sampling does not need one — re-export with threshold-bisection nucleus selection and a vocabulary-order inverse CDF (see `miniverse agent-help onnx`); capping K changes sampling behavior",
        )]
    return []


_NODE_RULES = {"TopK": _topk_findings}


def scan_model(source: BinaryIO) -> ModelScan:
    """Stream one ONNX ModelProto, returning its precision and compatibility findings."""
    metadata: dict[str, str] = {}
    opsets: dict[str, int] = {}
    state = _GraphState()
    while True:
        tag = _read_varint(source, allow_eof=True)
        if tag is None:
            break
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise OnnxMetadataError("ONNX contains an invalid protobuf field")
        if number == 7 and wire == 2:
            _parse_graph(_Bounded(source, _length(source)), state, 1)
        elif number == 8 and wire == 2:
            entry = BytesIO(_read_exact(source, _length(source, MAX_STRING_BYTES)))
            domain, version = "", None
            while True:
                entry_tag = _read_varint(entry, allow_eof=True)
                if entry_tag is None:
                    break
                entry_number, entry_wire = entry_tag >> 3, entry_tag & 7
                if entry_number == 1 and entry_wire == 2:
                    domain = _read_exact(entry, _length(entry)).decode("utf-8", errors="replace")
                elif entry_number == 2 and entry_wire == 0:
                    version = _length(entry)
                else:
                    _skip_value(entry, entry_wire)
            if version is not None:
                opsets[domain] = version
        elif number == 14 and wire == 2:
            fields = _entry_fields(_read_exact(source, _length(source, MAX_METADATA_ENTRY_BYTES)))
            if 1 in fields and 2 in fields:
                try:
                    key = fields[1].decode("utf-8")
                    value = fields[2].decode("utf-8")
                except UnicodeDecodeError as error:
                    raise OnnxMetadataError("ONNX metadata must be UTF-8") from error
                if key in metadata:
                    raise OnnxMetadataError(f"ONNX metadata contains duplicate key {key!r}")
                metadata[key] = value
        else:
            _skip_value(source, wire)
    precision = validate_contract(metadata)
    opset = opsets.get("", 0)
    findings: list[CompatFinding] = []
    for node in state.nodes:
        rule = _NODE_RULES.get(node.op_type)
        if rule is not None:
            findings.extend(rule(node, state, opset))
    return ModelScan(precision=precision, findings=tuple(findings))
