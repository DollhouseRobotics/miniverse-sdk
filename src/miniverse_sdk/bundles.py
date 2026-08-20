"""Safe, streaming inspection of canonical Miniverse .dhsim archives."""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .onnx_compat import CompatFinding, scan_model
from .onnx_metadata import OnnxMetadataError

MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
MAX_ENTRIES = 64
MAX_COMPRESSION_RATIO = 200
SIMULATORS = {"mujoco", "isaac-sim", "isaac-sim-gpu-physx"}


class BundleValidationError(ValueError):
    """A stable, user-actionable bundle validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AssetInspection:
    kind: str
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class BundleInspection:
    path: str
    archive_sha256: str
    archive_bytes: int
    bundle_id: str
    name: str
    primary_simulator: str
    program_sha256: str
    program_source: str
    assets: tuple[AssetInspection, ...]
    model_precisions: dict[str, str]
    model_findings: dict[str, tuple[CompatFinding, ...]]
    manifest: dict[str, Any]

    def as_dict(self, include_manifest: bool = False) -> dict[str, Any]:
        value = asdict(self)
        if not include_manifest:
            value.pop("manifest")
            value.pop("program_source")
        return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(archive: zipfile.ZipFile, name: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with archive.open(name) as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BundleValidationError("invalid_manifest", f"{label} must be an object")
    return value


def _string(value: Any, label: str, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise BundleValidationError("invalid_manifest", f"{label} must be a non-empty string of at most {maximum} characters")
    return value


def _hash(value: Any, label: str) -> str:
    text = _string(value, label, 64)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BundleValidationError("invalid_manifest", f"{label} must be a lowercase SHA-256 digest")
    return text


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > MAX_ENTRIES:
        raise BundleValidationError("unsafe_archive", f"bundle must contain between 1 and {MAX_ENTRIES} ZIP entries")
    members: dict[str, zipfile.ZipInfo] = {}
    folded: set[str] = set()
    expanded = 0
    for info in infos:
        name = info.filename
        pure = PurePosixPath(name)
        if not name or pure.is_absolute() or ".." in pure.parts or "\\" in name or name.endswith("/"):
            raise BundleValidationError("unsafe_archive", f"bundle contains unsafe or non-file path {name!r}")
        if name in members or name.casefold() in folded:
            raise BundleValidationError("unsafe_archive", f"bundle contains duplicate or case-colliding path {name!r}")
        mode = info.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            raise BundleValidationError("unsafe_archive", f"bundle contains symbolic link {name!r}")
        if info.flag_bits & 0x1:
            raise BundleValidationError("unsafe_archive", f"bundle contains encrypted entry {name!r}")
        if (name == "bundle.json" and info.file_size > 128 * 1024) or (name == "policy.py" and info.file_size > 64 * 1024):
            raise BundleValidationError("unsafe_archive", f"bundle entry {name!r} exceeds its size limit")
        expanded += info.file_size
        if expanded > MAX_EXPANDED_BYTES:
            raise BundleValidationError("unsafe_archive", "bundle expanded size exceeds 4 GiB")
        if (not info.compress_size and info.file_size) or (info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO):
            raise BundleValidationError("unsafe_archive", f"bundle entry {name!r} exceeds the compression-ratio limit")
        members[name] = info
        folded.add(name.casefold())
    return members


def inspect_bundle(path: str | Path) -> BundleInspection:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise BundleValidationError("archive_missing", f"bundle does not exist: {bundle_path}")
    archive_bytes = bundle_path.stat().st_size
    if archive_bytes <= 0 or archive_bytes > MAX_ARCHIVE_BYTES:
        raise BundleValidationError("archive_size", "bundle archive must be between 1 byte and 2 GiB")
    archive_sha256 = sha256_file(bundle_path)
    try:
        archive = zipfile.ZipFile(bundle_path)
    except zipfile.BadZipFile as error:
        raise BundleValidationError("invalid_zip", "bundle is not a valid ZIP archive") from error
    with archive:
        members = _safe_members(archive)
        required = {"bundle.json", "policy.py", "scene.glb"}
        missing = required - members.keys()
        if missing:
            raise BundleValidationError("missing_member", f"bundle is missing {', '.join(sorted(missing))}")
        try:
            manifest = json.loads(archive.read("bundle.json"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise BundleValidationError("invalid_manifest", "bundle.json must be valid UTF-8 JSON") from error
        manifest = _object(manifest, "bundle manifest")
        if "policyBindings" in manifest:
            raise BundleValidationError("invalid_manifest", "policyBindings was removed; bundle controllers always construct model inputs from SimData")
        bundle_id = _string(manifest.get("id"), "bundle id", 128)
        name = _string(manifest.get("name", bundle_id), "bundle name", 160)
        simulator = _string(manifest.get("primarySimulator"), "primarySimulator", 32)
        if simulator not in SIMULATORS:
            raise BundleValidationError("invalid_manifest", "primarySimulator is unsupported")
        program = _object(manifest.get("program"), "program")
        program_hash = _hash(program.get("sourceSha256"), "program.sourceSha256")
        scene = _object(manifest.get("scene"), "scene")
        expected: dict[str, tuple[str, str]] = {
            "policy.py": ("program", program_hash),
            "scene.glb": ("scene", _hash(scene.get("sha256"), "scene.sha256")),
        }
        models = manifest.get("models")
        if not isinstance(models, list) or not 1 <= len(models) <= 8:
            raise BundleValidationError("invalid_manifest", "bundle must declare between one and eight models")
        model_ids: set[str] = set()
        for index, raw in enumerate(models):
            model = _object(raw, f"model {index}")
            model_id = _string(model.get("id"), f"model {index} id", 128)
            if model_id in model_ids:
                raise BundleValidationError("invalid_manifest", f"duplicate model id {model_id!r}")
            model_ids.add(model_id)
            expected[f"models/{model_id}.onnx"] = ("model", _hash(model.get("sha256"), f"model {model_id} sha256"))
        primary_model = manifest.get("primaryModel")
        if primary_model not in model_ids:
            raise BundleValidationError("invalid_manifest", "primaryModel must name a declared model")
        for kind, archive_name, manifest_key in (("robot", "robot/mjcf.zip", "robot"), ("visual", "visual/robot.glb", "visual")):
            if manifest.get(manifest_key) is not None:
                asset = _object(manifest[manifest_key], manifest_key)
                expected[archive_name] = (kind, _hash(asset.get("sha256"), f"{manifest_key}.sha256"))
                declared_bytes = asset.get("bytes")
                if not isinstance(declared_bytes, int) or declared_bytes <= 0:
                    raise BundleValidationError("invalid_manifest", f"{manifest_key}.bytes must be a positive integer")
        undeclared = set(members) - set(expected) - {"bundle.json"}
        missing_assets = set(expected) - set(members)
        if missing_assets:
            raise BundleValidationError("missing_member", f"bundle is missing {', '.join(sorted(missing_assets))}")
        if undeclared:
            raise BundleValidationError("undeclared_member", f"bundle contains undeclared members: {', '.join(sorted(undeclared))}")
        assets: list[AssetInspection] = []
        model_precisions: dict[str, str] = {}
        model_findings: dict[str, tuple[CompatFinding, ...]] = {}
        for archive_name, (kind, expected_hash) in expected.items():
            actual_hash, size = _sha256_stream(archive, archive_name)
            if actual_hash != expected_hash:
                raise BundleValidationError("hash_mismatch", f"{archive_name} does not match its declared SHA-256")
            if kind in {"robot", "visual"} and size != int(manifest[kind]["bytes"]):
                raise BundleValidationError("size_mismatch", f"{archive_name} does not match its declared byte length")
            if kind == "model":
                try:
                    with archive.open(archive_name) as source:
                        scanned = scan_model(source)
                except OnnxMetadataError as error:
                    raise BundleValidationError("invalid_model_metadata", f"{archive_name}: {error}") from error
                model_id = PurePosixPath(archive_name).stem
                model_precisions[model_id] = scanned.precision
                if scanned.findings:
                    model_findings[model_id] = scanned.findings
            assets.append(AssetInspection(kind=kind, path=archive_name, sha256=actual_hash, bytes=size))
        try:
            program_source = archive.read("policy.py").decode("utf-8")
        except UnicodeDecodeError as error:
            raise BundleValidationError("invalid_program", "policy.py must be valid UTF-8") from error
        return BundleInspection(
            path=str(bundle_path), archive_sha256=archive_sha256, archive_bytes=archive_bytes,
            bundle_id=bundle_id, name=name, primary_simulator=simulator,
            program_sha256=program_hash, program_source=program_source,
            assets=tuple(assets), model_precisions=model_precisions, model_findings=model_findings, manifest=manifest,
        )
