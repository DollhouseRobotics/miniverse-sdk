"""Local inspection for immutable challenge-definition bundles."""

from __future__ import annotations

import ast
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .bundles import MAX_ARCHIVE_BYTES, BundleValidationError, _compile_embodiment, _safe_members, _validate_environment_mjcf, sha256_file

ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SIMULATORS = {"mujoco", "isaac-sim-cpu-physx", "isaac-sim-gpu-physx"}


@dataclass(frozen=True)
class ChallengeBundleInspection:
    path: str
    archive_sha256: str
    archive_bytes: int
    set_id: str
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "archiveSha256": self.archive_sha256,
            "archiveBytes": self.archive_bytes,
            "setId": self.set_id,
            "version": self.manifest["version"],
            "kind": self.manifest["kind"],
            "challengeIds": [value["id"] for value in self.manifest["challenges"]],
        }


def _fail(message: str, code: str = "invalid_challenge_manifest") -> None:
    raise BundleValidationError(code, message)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value):
        _fail(f"{label} must be a portable non-empty ID")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (positive and value <= 0):
        _fail(f"{label} must be a {'positive ' if positive else ''}finite number")
    return float(value)


def inspect_challenge_bundle(path: str | Path) -> ChallengeBundleInspection:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise BundleValidationError("archive_missing", f"bundle does not exist: {bundle_path}")
    archive_bytes = bundle_path.stat().st_size
    if archive_bytes <= 0 or archive_bytes > MAX_ARCHIVE_BYTES:
        raise BundleValidationError("archive_size", "bundle archive must be between 1 byte and 2 GiB")
    try:
        archive = zipfile.ZipFile(bundle_path)
    except zipfile.BadZipFile as error:
        raise BundleValidationError("invalid_zip", "bundle is not a valid ZIP archive") from error
    with archive:
        members = _safe_members(archive)
        missing = {"bundle.json", "challenge.py"} - members.keys()
        if missing:
            raise BundleValidationError("missing_member", f"bundle is missing {', '.join(sorted(missing))}")
        forbidden = [name for name in members if name == "policy.py" or name.startswith(("models/", "embodiment/"))]
        if forbidden:
            _fail(f"challenge bundle contains participant-only members: {', '.join(sorted(forbidden))}", "forbidden_member")
        if members["challenge.py"].file_size > 64 * 1024:
            raise BundleValidationError("unsafe_archive", "bundle entry 'challenge.py' exceeds its size limit")
        try:
            manifest = json.loads(archive.read("bundle.json"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise BundleValidationError("invalid_manifest", "bundle.json must be valid UTF-8 JSON") from error
        if not isinstance(manifest, dict) or manifest.get("version") != "v1" or manifest.get("kind") != "challenge":
            _fail('challenge bundle manifest must have version "v1" and kind "challenge"')
        allowed = {"version", "kind", "id", "name", "description", "primarySimulator", "compatibleSimulators", "environment", "challengeProgram", "challenges", "metadata"}
        unknown = sorted(set(manifest) - allowed)
        if unknown:
            _fail(f"challenge bundle manifest contains unknown fields: {', '.join(unknown)}")
        set_id = _identifier(manifest.get("id"), "challenge set id")
        if not isinstance(manifest.get("name"), str) or not manifest["name"]:
            _fail("challenge set name must be a non-empty string")
        simulator = manifest.get("primarySimulator")
        if simulator not in SIMULATORS:
            _fail("primarySimulator is unsupported")
        compatible = manifest.get("compatibleSimulators", [])
        if not isinstance(compatible, list) or len(set(compatible)) != len(compatible) or any(value not in SIMULATORS or value == simulator for value in compatible):
            _fail("compatibleSimulators must contain distinct supported non-primary simulators")
        program = manifest.get("challengeProgram")
        if not isinstance(program, dict) or program != {"apiVersion": "dhr.python-challenge/v1", "source": "challenge.py"}:
            _fail("challengeProgram must select dhr.python-challenge/v1 from challenge.py")
        environment = manifest.get("environment")
        environment_members: set[str] = set()
        if not isinstance(environment, dict) or environment.get("kind") not in {"builtin", "glb", "mjcf"}:
            _fail("environment must select builtin, glb, or mjcf")
        if environment["kind"] == "builtin":
            if set(environment) != {"kind", "id"}:
                _fail("builtin environment must contain exactly kind and id")
            if not isinstance(environment.get("id"), str) or not environment["id"]:
                _fail("environment id must be a non-empty string")
        else:
            if set(environment) != {"kind", "path"} or not isinstance(environment.get("path"), str):
                _fail("authored environment must contain exactly kind and path")
            source = PurePosixPath(environment["path"])
            if source.is_absolute() or ".." in source.parts or not str(source).startswith("environment/") or environment["path"] not in members:
                _fail("environment.path must name a safe member beneath environment/")
            expected_suffix = ".glb" if environment["kind"] == "glb" else ".xml"
            if source.suffix.lower() != expected_suffix:
                _fail(f"{environment['kind']} environment entrypoint must end in {expected_suffix}")
            environment_members = {name for name in members if name.startswith("environment/")}
            if environment["kind"] == "mjcf":
                compiled_path, compiled = _compile_embodiment(archive, members, environment, subtree="environment")
                if compiled_path != environment["path"]:
                    _fail("compiled environment entrypoint did not match environment.path")
                _validate_environment_mjcf(compiled)
        extras = set(members) - {"bundle.json", "challenge.py"} - environment_members
        if extras:
            raise BundleValidationError("undeclared_member", f"bundle contains undeclared members: {', '.join(sorted(extras))}")
        try:
            source = archive.read("challenge.py").decode("utf-8")
            tree = ast.parse(source, filename="challenge.py")
        except (UnicodeDecodeError, SyntaxError) as error:
            raise BundleValidationError("invalid_program", f"challenge.py is not valid UTF-8 Python: {error}") from error
        classes = {
            node.name: {child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for node in tree.body if isinstance(node, ast.ClassDef)
        }
        challenges = manifest.get("challenges")
        if not isinstance(challenges, list) or not challenges:
            _fail("challenges must contain at least one challenge")
        challenge_ids: set[str] = set()
        for index, challenge in enumerate(challenges):
            if not isinstance(challenge, dict):
                _fail(f"challenge {index} must be an object")
            challenge_id = _identifier(challenge.get("id"), f"challenge {index} id")
            if challenge_id in challenge_ids:
                _fail(f"duplicate challenge id {challenge_id!r}")
            challenge_ids.add(challenge_id)
            entrypoint = challenge.get("entrypoint")
            if not isinstance(entrypoint, str) or not entrypoint.startswith("challenge:") or entrypoint[10:] not in classes:
                _fail(f"challenge {challenge_id} entrypoint must name a top-level class in challenge.py")
            if not {"reset", "command", "evaluate"} <= classes[entrypoint[10:]]:
                _fail(f"challenge {challenge_id} evaluator must define reset, command, and evaluate")
            _finite(challenge.get("timeoutSeconds"), f"challenge {challenge_id} timeoutSeconds", positive=True)
            for field in ("commands", "gizmos"):
                values = challenge.get(field, [])
                if not isinstance(values, list):
                    _fail(f"challenge {challenge_id} {field} must be a list")
                ids = [_identifier(value.get("id") if isinstance(value, dict) else None, f"challenge {challenge_id} {field} id") for value in values]
                if len(ids) != len(set(ids)):
                    _fail(f"challenge {challenge_id} has duplicate {field} IDs")
                if field == "commands":
                    for command in values:
                        length = command.get("sliceLength")
                        default = command.get("default")
                        bounds = command.get("range")
                        if isinstance(length, bool) or not isinstance(length, int) or length <= 0 or not isinstance(default, list) or len(default) != length:
                            _fail(f"challenge {challenge_id} command {command['id']} has an invalid shape or default")
                        if not isinstance(bounds, list) or len(bounds) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
                            _fail(f"challenge {challenge_id} command {command['id']} has an invalid range")
                        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1] for value in default):
                            _fail(f"challenge {challenge_id} command {command['id']} default is outside its range")
                        if command.get("frame") not in {"world", "root", "body", "selection"} or command.get("update") not in {"continuous", "on-release"}:
                            _fail(f"challenge {challenge_id} command {command['id']} has an invalid frame or update mode")
            score = challenge.get("score")
            if not isinstance(score, dict) or score.get("order") not in {"higher-is-better", "lower-is-better"}:
                _fail(f"challenge {challenge_id} score.order is invalid")
            decimals = score.get("decimals")
            if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 12:
                _fail(f"challenge {challenge_id} score.decimals must be between 0 and 12")
    return ChallengeBundleInspection(str(bundle_path), sha256_file(bundle_path), archive_bytes, set_id, manifest)
