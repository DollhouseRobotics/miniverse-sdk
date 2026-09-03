"""Canonical heightfield authoring and local inspection."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

EXTENSION = "DHR_robotics_rollout"
MAX_HEIGHTFIELD_SAMPLES = 1024 * 1024
COORDINATE_FRAME = {
    "source": {"handedness": "right", "up": "+Z", "forward": "+X", "units": "m"},
    "target": {"handedness": "right", "up": "+Y", "forward": "+Z", "units": "m"},
    "sourceToGltf": [0, -1, 0, 0, 0, 1, 1, 0, 0],
}


class TerrainValidationError(ValueError):
    pass


@dataclass(frozen=True)
class HeightfieldInspection:
    id: str
    grid_sha256: str
    width: int
    height: int
    origin: tuple[float, float, float]
    xy_resolution: tuple[float, float]
    vertical_scale: float
    vertical_offset: float
    out_of_bounds: str
    out_of_bounds_value: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gridSha256": self.grid_sha256,
            "width": self.width,
            "height": self.height,
            "origin": list(self.origin),
            "xyResolution": list(self.xy_resolution),
            "verticalScale": self.vertical_scale,
            "verticalOffset": self.vertical_offset,
            "outOfBounds": self.out_of_bounds,
            **({"outOfBoundsValue": self.out_of_bounds_value} if self.out_of_bounds_value is not None else {}),
        }


def load_height_array(path: str | Path) -> tuple[int, int, list[float]]:
    """Load a finite row-major 2-D .npy or JSON height array."""
    source = Path(path)
    try:
        if source.suffix.lower() == ".npy":
            import numpy as np

            array = np.load(source, allow_pickle=False)
            if array.ndim != 2:
                raise TerrainValidationError("height array must be two-dimensional")
            rows = array.tolist()
        elif source.suffix.lower() == ".json":
            rows = json.loads(source.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        else:
            raise TerrainValidationError("height array must use .npy or .json")
    except TerrainValidationError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise TerrainValidationError(f"could not load height array: {error}") from error
    if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(row, list) for row in rows):
        raise TerrainValidationError("height array must contain at least two rows")
    width = len(rows[0])
    if width < 2 or any(len(row) != width for row in rows):
        raise TerrainValidationError("height array rows must have one shared width of at least two")
    height = len(rows)
    if width * height > MAX_HEIGHTFIELD_SAMPLES:
        raise TerrainValidationError(f"height array exceeds the {MAX_HEIGHTFIELD_SAMPLES}-sample limit")
    values: list[float] = []
    for row in rows:
        for raw in row:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                raise TerrainValidationError("height array contains a non-finite or non-numeric sample")
            values.append(float(raw))
    return width, height, values


def build_heightfield_glb(
    *,
    terrain_id: str,
    width: int,
    height: int,
    heights: Sequence[float],
    xy_resolution: Sequence[float],
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    vertical_scale: float = 1.0,
    vertical_offset: float = 0.0,
    out_of_bounds: str = "error",
    out_of_bounds_value: float | None = None,
) -> bytes:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", terrain_id):
        raise TerrainValidationError("terrain id is invalid")
    width = _dimension(width, "width")
    height = _dimension(height, "height")
    values = _finite(heights, "height samples")
    cell = _tuple(xy_resolution, 2, "XY resolution")
    terrain_origin = _tuple(origin, 3, "origin")
    scale = _number(vertical_scale, "vertical scale")
    offset = _number(vertical_offset, "vertical offset")
    if width < 2 or height < 2 or width * height != len(values) or len(values) > MAX_HEIGHTFIELD_SAMPLES:
        raise TerrainValidationError("heightfield dimensions do not match its bounded sample array")
    if any(value <= 0 for value in cell):
        raise TerrainValidationError("XY resolution must be positive")
    if out_of_bounds not in {"error", "clamp", "constant"}:
        raise TerrainValidationError("out-of-bounds behavior must be error, clamp, or constant")
    outside = None if out_of_bounds_value is None else _number(out_of_bounds_value, "out-of-bounds value")
    if (out_of_bounds == "constant") != (outside is not None):
        raise TerrainValidationError("out-of-bounds value is required only for constant behavior")
    binary = struct.pack("<" + "f" * len(values), *values)
    accessor = {
        "bufferView": 0, "componentType": 5126, "count": len(values), "type": "SCALAR",
        "min": [min(values)], "max": [max(values)],
    }
    surface: dict[str, Any] = {
        "id": terrain_id, "type": "heightfield", "nodeId": "world", "width": width, "height": height,
        "heightsAccessor": 0, "cellSize": cell, "scale": scale, "offset": offset,
        "origin": terrain_origin, "outOfBounds": out_of_bounds,
    }
    if outside is not None:
        surface["outOfBoundsValue"] = outside
    extension = {
        "schemaVersion": "0.1", "assetRole": "rollout", "rolloutId": f"environment/{terrain_id}",
        "sourceTrainingStep": 0,
        "animation": {"index": None, "durationSeconds": 0.0, "clock": "rollout-seconds"},
        "skeleton": {"skeletonId": "environment", "rootNodeId": "world"},
        "coordinateFrame": COORDINATE_FRAME,
        "bodies": [], "joints": [], "visualBindings": [], "tracks": [],
        "environment": {"surfaces": [surface], "objects": []},
        "collisionShapes": [], "colliders": [], "gizmos": [], "labels": [],
        "timeAxes": [], "metrics": [], "parameters": {}, "provenance": {},
    }
    document = {
        "asset": {"version": "2.0", "generator": "Miniverse heightfield environment v1"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"name": "world"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "accessors": [accessor], "animations": [], "extensionsUsed": [EXTENSION],
        "extensions": {EXTENSION: extension},
    }
    return _encode_glb(document, binary)


def inspect_heightfield_glb(data: bytes) -> HeightfieldInspection:
    document, binary = _parse_glb(data)
    extension = document.get("extensions", {}).get(EXTENSION)
    if not isinstance(extension, dict) or extension.get("schemaVersion") != "0.1" or extension.get("coordinateFrame") != COORDINATE_FRAME:
        raise TerrainValidationError("terrain GLB is missing the canonical DHR heightfield contract")
    environment = extension.get("environment", {})
    surfaces = environment.get("surfaces", []) if isinstance(environment, dict) else []
    if not isinstance(surfaces, list) or len(surfaces) != 1 or not isinstance(surfaces[0], dict) or surfaces[0].get("type") != "heightfield":
        raise TerrainValidationError("terrain GLB must contain exactly one heightfield")
    if set(environment) != {"surfaces", "objects"}:
        raise TerrainValidationError("terrain GLB environment contains unsupported fields")
    surface = surfaces[0]
    allowed_surface_fields = {
        "id", "type", "nodeId", "width", "height", "heightsAccessor", "cellSize",
        "scale", "offset", "origin", "outOfBounds", "outOfBoundsValue",
    }
    if set(surface) - allowed_surface_fields or surface.get("nodeId") != "world":
        raise TerrainValidationError("terrain GLB heightfield contains unsupported fields or is not world-anchored")
    values = _read_float_accessor(document, binary, surface.get("heightsAccessor"))
    terrain_id = str(surface.get("id", ""))
    raw_width, raw_height = surface.get("width"), surface.get("height")
    if (not isinstance(raw_width, int) or isinstance(raw_width, bool)
            or not isinstance(raw_height, int) or isinstance(raw_height, bool)):
        raise TerrainValidationError("terrain GLB dimensions must be integers")
    width, height = raw_width, raw_height
    cell = _tuple(surface.get("cellSize", ()), 2, "XY resolution")
    origin = _tuple(surface.get("origin", (0, 0, 0)), 3, "origin")
    scale = _number(surface.get("scale", 1), "vertical scale")
    offset = _number(surface.get("offset", 0), "vertical offset")
    behavior = str(surface.get("outOfBounds", "error"))
    outside = surface.get("outOfBoundsValue")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", terrain_id) or width < 2 or height < 2 or width * height != len(values) or len(values) > MAX_HEIGHTFIELD_SAMPLES:
        raise TerrainValidationError("terrain GLB heightfield identity or dimensions are invalid")
    if (extension.get("assetRole") != "rollout"
            or extension.get("rolloutId") != f"environment/{terrain_id}"
            or extension.get("sourceTrainingStep") != 0
            or extension.get("skeleton") != {"skeletonId": "environment", "rootNodeId": "world"}):
        raise TerrainValidationError("terrain GLB environment identity is invalid")
    if any(value <= 0 for value in cell) or behavior not in {"error", "clamp", "constant"}:
        raise TerrainValidationError("terrain GLB sampling metadata is invalid")
    if behavior == "constant":
        outside = _number(outside, "out-of-bounds value")
    elif outside is not None:
        raise TerrainValidationError("out-of-bounds value is valid only for constant behavior")
    forbidden = [name for name in ("bodies", "joints", "visualBindings", "tracks", "collisionShapes", "colliders", "gizmos", "labels", "timeAxes", "metrics") if extension.get(name)]
    if environment.get("objects") or forbidden or document.get("meshes") or document.get("animations") or extension.get("parameters") or extension.get("provenance"):
        raise TerrainValidationError("terrain GLB must be data-only; Miniverse derives render and collision meshes")
    identity = json.dumps({
        "width": width, "height": height, "heights": tuple(values), "origin": tuple(origin),
        "xyResolution": tuple(cell), "verticalScale": scale, "verticalOffset": offset,
    }, sort_keys=True, separators=(",", ":")).encode()
    return HeightfieldInspection(
        terrain_id, hashlib.sha256(identity).hexdigest(), width, height,
        tuple(origin), tuple(cell), scale, offset, behavior, outside,
    )


def _stable_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _pad(data: bytes, byte: bytes) -> bytes:
    return data + byte * ((-len(data)) % 4)


def _encode_glb(document: dict[str, Any], binary: bytes) -> bytes:
    json_chunk = _pad(_stable_json(document), b" ")
    bin_chunk = _pad(binary, b"\0")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk + struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk


def _parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < 20 or struct.unpack_from("<III", data, 0) != (0x46546C67, 2, len(data)):
        raise TerrainValidationError("terrain is not a valid GLB 2.0 file")
    offset, document, binary = 12, None, b""
    while offset < len(data):
        if offset + 8 > len(data):
            raise TerrainValidationError("terrain GLB has a truncated chunk")
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + length]
        if len(chunk) != length:
            raise TerrainValidationError("terrain GLB has a truncated chunk")
        offset += length
        if kind == 0x4E4F534A:
            try:
                document = json.loads(chunk.decode("utf-8").rstrip(" \0"))
            except (UnicodeError, ValueError) as error:
                raise TerrainValidationError(f"terrain GLB has an invalid JSON document: {error}") from error
        elif kind == 0x004E4942:
            binary = chunk
    if not isinstance(document, dict):
        raise TerrainValidationError("terrain GLB has no JSON document")
    return document, binary


def _read_float_accessor(document: dict[str, Any], binary: bytes, raw_index: Any) -> list[float]:
    try:
        accessor = document["accessors"][int(raw_index)]
        view = document["bufferViews"][int(accessor["bufferView"])]
        count = int(accessor["count"])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise TerrainValidationError("heightfield references an invalid accessor") from error
    if accessor.get("componentType") != 5126 or accessor.get("type") != "SCALAR" or accessor.get("sparse") is not None or view.get("byteStride") not in {None, 4}:
        raise TerrainValidationError("heightfield samples must use a non-sparse scalar FLOAT accessor")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    length = count * 4
    if count <= 0 or start < 0 or start + length > len(binary):
        raise TerrainValidationError("heightfield accessor exceeds its embedded buffer")
    return _finite(struct.unpack_from("<" + "f" * count, binary, start), "height samples")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TerrainValidationError(f"{label} must be finite")
    return float(value)


def _finite(values: Sequence[Any], label: str) -> list[float]:
    return [_number(value, label) for value in values]


def _tuple(values: Sequence[Any], width: int, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != width:
        raise TerrainValidationError(f"{label} must contain {width} values")
    return _finite(values, label)


def _dimension(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise TerrainValidationError(f"heightfield {label} must be an integer of at least two")
    return value
