"""Bounded extraction of canonical Miniverse metadata from an ONNX ModelProto."""

from __future__ import annotations

import json
from typing import BinaryIO

ONNX_METADATA_KEY = "com.dollhouserobotics.miniverse.simulation_contract"
ONNX_SCHEMA_KEY = "com.dollhouserobotics.miniverse.simulation_contract_schema_version"
ONNX_SCHEMA_VERSION = "0.3"
SUPPORTED_PRECISIONS = ("fp32", "fp16", "bf16")
MAX_METADATA_ENTRY_BYTES = 512 * 1024

# Static per-simulator observation capability table, mirrored from the runtime
# engine adapters. "conditional" providers depend on a runtime probe (Isaac
# contact/scene-query frontends) and are verified again at session start.
SHARED_PROVIDERS = ("command", "previousAction", "simTime", "constant", "history", "deterministicRandom")
SIMULATOR_PROVIDERS: dict[str, dict[str, str]] = {
    "mujoco": {
        **{provider: "fulfilled" for provider in SHARED_PROVIDERS},
        **{provider: "fulfilled" for provider in (
            "jointPosition", "jointVelocity", "jointEffort", "bodyPose", "bodyLinearVelocity",
            "bodyAngularVelocity", "projectedGravity", "contacts", "surfaceQuery",
        )},
    },
    "isaac-sim": {
        **{provider: "fulfilled" for provider in SHARED_PROVIDERS},
        **{provider: "fulfilled" for provider in (
            "jointPosition", "jointVelocity", "jointEffort", "bodyPose", "bodyLinearVelocity",
            "bodyAngularVelocity", "projectedGravity",
        )},
        "contacts": "conditional",
        "surfaceQuery": "conditional",
    },
}
SIMULATOR_PROVIDERS["isaac-sim-gpu-physx"] = dict(SIMULATOR_PROVIDERS["isaac-sim"])
# Isaac's tensor frontend only exposes net link contact forces.
ISAAC_CONTACT_COMPONENTS = ("netForce", "impulse")


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
    """Read the required v0.3 precision without loading the ONNX graph into memory."""
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

    return validate_contract(metadata)


def parse_contract(metadata: dict[str, str]) -> dict:
    """Validate collected ONNX metadata_props and return the embedded contract."""
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
    return contract


def validate_contract(metadata: dict[str, str]) -> str:
    """Backwards-compatible precision accessor over parse_contract."""
    return str(parse_contract(metadata)["precision"])


def fulfillment_report(contract: dict) -> dict:
    """Per-input provider fulfillment against each simulator's capability set.

    Authoring-time preview of the session-composition check: every slice's
    provider is resolved against the static capability table so an author sees
    an unservable observation before uploading, not at session start.
    """
    state_inputs = {str(value.get("inputName", "")) for value in contract.get("stateBindings", []) if isinstance(value, dict)}
    report: dict[str, list[dict]] = {}
    unfulfillable: list[str] = []
    entries = [
        *(({"input": str(item.get("name", "")), "slices": item.get("slices", [])},) for item in contract.get("inputs", []) if isinstance(item, dict) and str(item.get("name", "")) not in state_inputs),
        *(({"input": f"historyBuffer:{item.get('id')}", "slices": [item.get("source", {})]},) for item in contract.get("historyBuffers", []) if isinstance(item, dict)),
    ]
    for (entry,) in entries:
        rows = []
        for value in entry["slices"]:
            if not isinstance(value, dict):
                continue
            provider = str(value.get("provider", ""))
            component = value.get("component")
            row: dict = {"provider": provider}
            if component:
                row["component"] = str(component)
            for simulator, table in SIMULATOR_PROVIDERS.items():
                status = table.get(provider, "unfulfillable")
                if provider == "history":
                    status = "fulfilled"
                if provider == "contacts" and simulator.startswith("isaac") and component and str(component) not in ISAAC_CONTACT_COMPONENTS:
                    status = "unfulfillable"
                row[simulator] = status
                if status == "unfulfillable":
                    unfulfillable.append(f"{entry['input']}:{provider}@{simulator}")
            rows.append(row)
        report[entry["input"]] = rows
    return {"inputs": report, "unfulfillable": sorted(set(unfulfillable))}
