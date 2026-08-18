"""Bounded extraction of canonical Miniverse metadata from an ONNX ModelProto."""

from __future__ import annotations

import json
from typing import BinaryIO

ONNX_METADATA_KEY = "com.dollhouserobotics.miniverse.simulation_contract"
ONNX_SCHEMA_KEY = "com.dollhouserobotics.miniverse.simulation_contract_schema_version"
ONNX_SCHEMA_VERSION = "0.2"
SUPPORTED_PRECISIONS = ("fp32", "fp16", "bf16")
MAX_METADATA_ENTRY_BYTES = 512 * 1024


class OnnxMetadataError(ValueError):
    pass


def _read_varint(source: BinaryIO, *, allow_eof: bool = False) -> int | None:
    value = 0
    for index in range(10):
        raw = source.read(1)
        if not raw:
            if allow_eof and index == 0:
                return None
            raise OnnxMetadataError("ONNX metadata contains a truncated protobuf varint")
        byte = raw[0]
        value |= (byte & 0x7F) << (index * 7)
        if not byte & 0x80:
            return value
    raise OnnxMetadataError("ONNX metadata contains an invalid protobuf varint")


def _discard(source: BinaryIO, count: int) -> None:
    remaining = count
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise OnnxMetadataError("ONNX metadata contains a truncated protobuf value")
        remaining -= len(chunk)


def _entry_fields(data: bytes) -> dict[int, bytes]:
    from io import BytesIO

    source = BytesIO(data)
    result: dict[int, bytes] = {}
    while True:
        tag = _read_varint(source, allow_eof=True)
        if tag is None:
            return result
        number, wire = tag >> 3, tag & 7
        if number == 0 or wire != 2:
            raise OnnxMetadataError("ONNX metadata entry uses an unsupported protobuf field")
        length = _read_varint(source)
        assert length is not None
        value = source.read(length)
        if len(value) != length:
            raise OnnxMetadataError("ONNX metadata entry is truncated")
        if number in (1, 2):
            result[number] = value


def read_miniverse_precision(source: BinaryIO) -> str:
    """Read the required v0.2 precision without loading the ONNX graph into memory."""
    metadata: dict[str, str] = {}
    while True:
        tag = _read_varint(source, allow_eof=True)
        if tag is None:
            break
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise OnnxMetadataError("ONNX contains an invalid protobuf field")
        if wire == 0:
            _read_varint(source)
        elif wire == 1:
            _discard(source, 8)
        elif wire == 2:
            length = _read_varint(source)
            assert length is not None
            if number != 14:
                _discard(source, length)
                continue
            if length > MAX_METADATA_ENTRY_BYTES:
                raise OnnxMetadataError("ONNX metadata entry exceeds 512 KiB")
            entry = source.read(length)
            if len(entry) != length:
                raise OnnxMetadataError("ONNX metadata entry is truncated")
            fields = _entry_fields(entry)
            if 1 in fields and 2 in fields:
                try:
                    key = fields[1].decode("utf-8")
                    value = fields[2].decode("utf-8")
                except UnicodeDecodeError as error:
                    raise OnnxMetadataError("ONNX metadata must be UTF-8") from error
                if key in metadata:
                    raise OnnxMetadataError(f"ONNX metadata contains duplicate key {key!r}")
                metadata[key] = value
        elif wire == 5:
            _discard(source, 4)
        else:
            raise OnnxMetadataError(f"ONNX uses unsupported protobuf wire type {wire}")

    if metadata.get(ONNX_SCHEMA_KEY) != ONNX_SCHEMA_VERSION:
        raise OnnxMetadataError(f"ONNX checkpoint must use Miniverse contract schema {ONNX_SCHEMA_VERSION}")
    raw_contract = metadata.get(ONNX_METADATA_KEY)
    if raw_contract is None:
        raise OnnxMetadataError("ONNX checkpoint is missing its embedded Miniverse contract")
    try:
        contract = json.loads(raw_contract)
    except json.JSONDecodeError as error:
        raise OnnxMetadataError("ONNX Miniverse contract is invalid JSON") from error
    if not isinstance(contract, dict) or contract.get("schemaVersion") != ONNX_SCHEMA_VERSION:
        raise OnnxMetadataError(f"ONNX checkpoint must use Miniverse contract schema {ONNX_SCHEMA_VERSION}")
    precision = contract.get("precision")
    if precision not in SUPPORTED_PRECISIONS:
        raise OnnxMetadataError(
            "ONNX inference precision is required and must be one of: " + ", ".join(SUPPORTED_PRECISIONS)
        )
    return str(precision)
