"""Safe, streaming inspection of canonical Miniverse .mini archives."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from .onnx_compat import CompatFinding, scan_model
from .onnx_metadata import OnnxMetadataError

MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
MAX_ENTRIES = 64
MAX_COMPRESSION_RATIO = 200
SIMULATORS = {"mujoco", "isaac-sim-cpu-physx", "isaac-sim-gpu-physx"}
SOURCE_FIELDS = {"version", "id", "name", "description", "primarySimulator", "compatibleSimulators", "environment", "embodiment", "models", "program", "commands", "ui", "gizmos", "webModules", "metadata"}


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
    heightfield: dict[str, Any] | None = None


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
        value["assets"] = [
            {key: item for key, item in asset.items() if item is not None}
            for asset in value["assets"]
        ]
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


def _validate_body_dynamics_overrides(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) > 64:
        raise BundleValidationError("invalid_manifest", "embodiment.bodyDynamicsOverrides must be an array with at most 64 entries")
    seen: set[str] = set()
    fields = {"linearDamping", "angularDamping", "maximumLinearVelocity", "maximumAngularVelocity"}
    for index, raw in enumerate(value):
        override = _object(raw, f"body dynamics override {index}")
        if set(override) != {"bodyGroup", "bodyIds", "set"}:
            raise BundleValidationError("invalid_manifest", f"body dynamics override {index} must contain only bodyGroup, bodyIds, and set")
        group = _string(override.get("bodyGroup"), f"body dynamics override {index} bodyGroup", 128)
        body_ids = override.get("bodyIds")
        if not isinstance(body_ids, list) or not 1 <= len(body_ids) <= 128 or any(not isinstance(item, str) for item in body_ids) or len(set(body_ids)) != len(body_ids):
            raise BundleValidationError("invalid_manifest", f"body dynamics override {group} requires unique bodyIds")
        for body_id_value in body_ids:
            body_id = _string(body_id_value, f"body dynamics override {group} body id", 128)
            if body_id in seen:
                raise BundleValidationError("invalid_manifest", f"embodiment body {body_id} is overridden more than once")
            seen.add(body_id)
        updates = _object(override.get("set"), f"body dynamics override {group} set")
        if not updates or set(updates) - fields:
            raise BundleValidationError("invalid_manifest", f"body dynamics override {group} set contains no values or unsupported fields")
        for field, raw_value in updates.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)) or float(raw_value) < 0:
                raise BundleValidationError("invalid_manifest", f"body dynamics override {group} {field} must be a finite non-negative number")
            if field.startswith("maximum") and float(raw_value) == 0:
                raise BundleValidationError("invalid_manifest", f"body dynamics override {group} {field} must be greater than zero")


def _hash(value: Any, label: str) -> str:
    text = _string(value, label, 64)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BundleValidationError("invalid_manifest", f"{label} must be a lowercase SHA-256 digest")
    return text


def _environment_path(value: Any, label: str = "environment.path") -> str:
    path = _string(value, label, 300)
    parts = path.split("/")
    if not path.startswith("environment/") or "\\" in path or any(not part or part in {".", ".."} for part in parts) or not path.endswith(".glb"):
        raise BundleValidationError("invalid_manifest", f"{label} must be a safe relative .glb path")
    return path


def _mjcf_path(value: Any, label: str) -> str:
    path = _string(value, label, 300)
    parts = path.split("/")
    if "\\" in path or any(not part or part in {".", ".."} for part in parts) or PurePosixPath(path).suffix.lower() not in {".xml", ".mjcf"}:
        raise BundleValidationError("invalid_manifest", f"{label} must be a safe relative MJCF path")
    return path


def _resolve_mjcf(base: PurePosixPath, value: str, label: str) -> str:
    if not value or "\\" in value or "\x00" in value or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value) or PurePosixPath(value).is_absolute():
        raise BundleValidationError("invalid_embodiment", f"{label} must be a local relative path")
    parts = list(base.parts)
    for part in PurePosixPath(value).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise BundleValidationError("invalid_embodiment", f"{label} escapes the embodiment directory")
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _compile_embodiment(archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], declaration: dict[str, Any]) -> tuple[str, bytes]:
    if declaration.get("kind") != "mjcf":
        raise BundleValidationError("invalid_manifest", "embodiment.kind must be mjcf")
    full_entrypoint = _mjcf_path(declaration.get("path"), "embodiment.path")
    if not full_entrypoint.startswith("embodiment/"):
        raise BundleValidationError("invalid_manifest", "embodiment.path must be under embodiment/")
    available = {name.removeprefix("embodiment/"): archive.read(name) for name in members if name.startswith("embodiment/")}
    entrypoint = full_entrypoint.removeprefix("embodiment/")
    if entrypoint not in available:
        raise BundleValidationError("missing_member", "bundle embodiment entrypoint is missing")
    pending = [entrypoint]
    selected: dict[str, bytes] = {}
    meshdir = ""
    texturedir = ""
    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        if relative not in available:
            raise BundleValidationError("missing_member", f"bundle embodiment dependency is missing: embodiment/{relative}")
        data = available[relative]
        if not data or len(data) > 256 * 1024 * 1024:
            raise BundleValidationError("invalid_embodiment", f"embodiment/{relative} has an invalid byte length")
        selected[relative] = data
        if len(selected) > 4096:
            raise BundleValidationError("invalid_embodiment", "embodiment contains too many files")
        if PurePosixPath(relative).suffix.lower() not in {".xml", ".mjcf"}:
            continue
        try:
            document = ElementTree.fromstring(data)
        except ElementTree.ParseError as error:
            raise BundleValidationError("invalid_embodiment", f"embodiment/{relative} is invalid MJCF XML") from error
        compiler = document.find("compiler")
        if compiler is not None:
            meshdir = str(compiler.get("meshdir", "")).strip()
            texturedir = str(compiler.get("texturedir", "")).strip()
        parent = PurePosixPath(relative).parent
        for include in document.iter("include"):
            pending.append(_resolve_mjcf(parent, str(include.get("file", "")).strip(), "MJCF include"))
        for element in document.iter():
            name = str(element.get("file", "")).strip()
            if not name or element.tag == "include":
                continue
            directory = meshdir if element.tag in {"mesh", "skin"} else texturedir if element.tag in {"texture", "hfield"} else ""
            base = PurePosixPath(entrypoint).parent
            if directory:
                base = PurePosixPath(_resolve_mjcf(base, directory, f"MJCF {element.tag} directory"))
            pending.append(_resolve_mjcf(base, name, f"MJCF {element.tag} asset"))
    if selected != available:
        extras = sorted(set(available) - set(selected))
        raise BundleValidationError("undeclared_member", f"bundle contains unused embodiment members: {', '.join('embodiment/' + name for name in extras)}")
    entries = [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)} for name, data in sorted(selected.items())]
    source = {"apiVersion": "dhr.mjcf-asset-set/v1", "entrypoint": entrypoint, "files": entries}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    manifest = {**source, "sourceHash": hashlib.sha256(canonical(source)).hexdigest()}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as compiled:
        for name, data in [("dhr-mjcf-assets.json", canonical(manifest)), *sorted(selected.items())]:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            compiled.writestr(info, data)
    return full_entrypoint, output.getvalue()


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
        required = {"bundle.json", "policy.py"}
        missing = required - members.keys()
        if missing:
            raise BundleValidationError("missing_member", f"bundle is missing {', '.join(sorted(missing))}")
        try:
            manifest = json.loads(archive.read("bundle.json"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise BundleValidationError("invalid_manifest", "bundle.json must be valid UTF-8 JSON") from error
        manifest = _object(manifest, "bundle manifest")
        removed = sorted(set(manifest) & {"policyBindings", "scene", "seed", "robot", "visual", "primaryModel", "provenance"})
        if removed:
            raise BundleValidationError("invalid_manifest", f"bundle fields were removed: {', '.join(removed)}")
        raw_embodiment = manifest.get("embodiment")
        if isinstance(raw_embodiment, dict) and "dynamicsOverrides" in raw_embodiment:
            raise BundleValidationError("invalid_manifest", "embodiment.dynamicsOverrides was removed; author actuator gains and limits in the embodiment MJCF")
        from .validation import validate_bundle_manifest

        schema_errors = validate_bundle_manifest(manifest)
        if schema_errors:
            error = BundleValidationError("invalid_manifest_schema", f"bundle.json has {len(schema_errors)} schema error(s)")
            error.details = [issue.as_dict() for issue in schema_errors]
            raise error
        unknown = sorted(set(manifest) - SOURCE_FIELDS)
        if unknown:
            raise BundleValidationError("invalid_manifest", f"bundle manifest contains unknown fields: {', '.join(unknown)}")
        if manifest.get("version") != "v1":
            raise BundleValidationError("invalid_manifest", "bundle version must be v1")
        bundle_id = _string(manifest.get("id"), "bundle id", 128)
        name = _string(manifest.get("name"), "bundle name", 160)
        simulator = _string(manifest.get("primarySimulator"), "primarySimulator", 32)
        if simulator not in SIMULATORS:
            raise BundleValidationError("invalid_manifest", "primarySimulator is unsupported")
        compatible = manifest.get("compatibleSimulators", [])
        if not isinstance(compatible, list) or any(value not in SIMULATORS or value == simulator for value in compatible) or len(set(compatible)) != len(compatible):
            raise BundleValidationError("invalid_manifest", "compatibleSimulators must contain distinct supported non-primary simulators")
        embodiment = _object(manifest.get("embodiment"), "embodiment")
        if set(embodiment) - {"kind", "path", "appearance", "bodyDynamicsOverrides"}:
            raise BundleValidationError("invalid_manifest", "embodiment accepts kind, path, appearance, and bodyDynamicsOverrides")
        _validate_body_dynamics_overrides(embodiment.get("bodyDynamicsOverrides"))
        embodiment_path, embodiment_archive = _compile_embodiment(archive, members, embodiment)
        program = _object(manifest.get("program"), "program")
        if set(program) != {"apiVersion", "entrypoint"} or program.get("apiVersion") != "dhr.python-policy/v1":
            raise BundleValidationError("invalid_manifest", "program must contain only apiVersion and entrypoint")
        _string(program.get("entrypoint"), "program.entrypoint", 200)
        expected: dict[str, str] = {"policy.py": "program"}
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
            if set(model) != {"id"}:
                raise BundleValidationError("invalid_manifest", f"model {model_id} accepts only id; hashes and providers are derived by Miniverse")
            expected[f"models/{model_id}.onnx"] = "model"
        environment = manifest.get("environment")
        if environment is not None:
            if not isinstance(environment, dict) or set(environment) != {"kind", "path"} or environment.get("kind") != "glb":
                raise BundleValidationError("invalid_manifest", "environment must contain exactly kind=glb and path")
            environment_path = _environment_path(environment.get("path"))
            if environment_path in expected or environment_path == "bundle.json":
                raise BundleValidationError("invalid_manifest", "environment.path collides with a reserved bundle member")
            expected[environment_path] = "scene"
        embodiment_members = {name for name in members if name.startswith("embodiment/")}
        undeclared = set(members) - set(expected) - embodiment_members - {"bundle.json"}
        missing_assets = set(expected) - set(members)
        if missing_assets:
            raise BundleValidationError("missing_member", f"bundle is missing {', '.join(sorted(missing_assets))}")
        if undeclared:
            raise BundleValidationError("undeclared_member", f"bundle contains undeclared members: {', '.join(sorted(undeclared))}")
        assets: list[AssetInspection] = []
        model_precisions: dict[str, str] = {}
        model_findings: dict[str, tuple[CompatFinding, ...]] = {}
        program_hash = ""
        for archive_name, kind in expected.items():
            actual_hash, size = _sha256_stream(archive, archive_name)
            heightfield = None
            if kind == "program":
                program_hash = actual_hash
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
            if kind == "scene":
                from .terrain import TerrainValidationError, inspect_heightfield_glb

                try:
                    heightfield = inspect_heightfield_glb(archive.read(archive_name)).as_dict()
                except TerrainValidationError as error:
                    raise BundleValidationError("invalid_heightfield", f"{archive_name}: {error}") from error
            assets.append(AssetInspection(kind=kind, path=archive_name, sha256=actual_hash, bytes=size, heightfield=heightfield))
        assets.insert(1, AssetInspection(kind="embodiment", path=embodiment_path, sha256=hashlib.sha256(embodiment_archive).hexdigest(), bytes=len(embodiment_archive)))
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
