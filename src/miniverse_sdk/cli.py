"""Command-line entrypoint for miniverse-sdk."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.parse
import webbrowser
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from .api import ApiError, Client
from .bundles import BundleValidationError, inspect_bundle
from .config import OAuthCredential, auth_file, auth_store, credential, delete_oauth_credential, origin, save_oauth_credential
from .validation import ModelValidation, validate_bundle_model_backends, validate_model

TOPICS = {"auth", "bundles", "upload", "sessions", "onnx"}


@dataclass(frozen=True)
class CommandResult:
    value: Any
    exit_code: int = 0


def emit(value: Any, as_json: bool = False) -> None:
    if as_json or isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        print(value)


def agent_help(topic: str | None, all_topics: bool) -> str:
    names = ["index", *sorted(TOPICS)] if all_topics else [topic or "index"]
    parts = []
    root = resources.files("miniverse_sdk.agent_help")
    for name in names:
        parts.append(root.joinpath(f"{name}.md").read_text(encoding="utf-8").strip())
    return "\n\n".join(parts) + "\n"


def auth_login(args: argparse.Namespace) -> dict[str, Any]:
    api_origin = origin(args.origin)
    client = Client(api_origin, None)
    created = client.request_form("/api/auth/device/code", {
        "client_id": "miniverse-cli",
        "scope": "openid profile email offline_access bundles:read bundles:upload bundles:publish",
        "resource": api_origin,
    })
    verification = str(created.get("verification_uri_complete") or created.get("verification_uri") or "")
    device_code = str(created.get("device_code") or "")
    if not verification or not device_code:
        raise ApiError(502, "authorization server returned an incomplete device flow", "oauth_contract_error")
    print(f"Open {verification}", file=sys.stderr)
    if not args.no_browser:
        webbrowser.open(verification)
    interval = max(1, int(created.get("interval", 5)))
    deadline = time.monotonic() + int(created.get("expires_in", 1800))
    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            value = client.request_form("/api/auth/oauth2/token", {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": "miniverse-cli",
            })
        except ApiError as error:
            if error.code == "authorization_pending":
                continue
            if error.code == "slow_down":
                interval += 5
                continue
            raise
        token = value.get("access_token")
        if isinstance(token, str) and token:
            refresh_token = value.get("refresh_token")
            expires_in = value.get("expires_in")
            saved = OAuthCredential(
                access_token=token,
                refresh_token=refresh_token if isinstance(refresh_token, str) and refresh_token else None,
                expires_at=time.time() + float(expires_in) if isinstance(expires_in, (int, float)) else None,
                scope=value.get("scope") if isinstance(value.get("scope"), str) else None,
                origin=api_origin,
            )
            if not saved.renewable:
                raise ApiError(502, "authorization server did not issue a refresh token", "oauth_contract_error")
            storage = save_oauth_credential(saved)
            return {"authenticated": True, "source": "oauth", "storage": storage, "renewable": True}
    raise ApiError(408, "device authorization expired", "expired_token")


def bundle_upload(args: argparse.Namespace) -> dict[str, Any]:
    validated = validate_bundle(args.bundle)
    if validated.exit_code:
        error = BundleValidationError("bundle_validation", "bundle failed local validation")
        error.details = validated.value
        raise error
    inspected = validated.inspection
    api_origin = origin(args.origin)
    saved, source = credential(api_origin)
    client = Client(api_origin, saved)
    prepared = client.request("/api/v1/bundle-imports", {
        "archiveSha256": inspected.archive_sha256,
        "bytes": inspected.archive_bytes,
        "filename": Path(args.bundle).name,
        "idempotencyKey": args.idempotency_key or f"archive:{inspected.archive_sha256}",
        "manifest": inspected.manifest,
        "programSource": inspected.program_source,
        "assets": [asset.__dict__ for asset in inspected.assets],
    })
    transfer = prepared.get("transfer")
    if not isinstance(transfer, dict) or transfer.get("mode") != "single" or not isinstance(transfer.get("url"), str):
        raise ApiError(502, "server returned unsupported upload instructions", "upload_contract_error")
    if not prepared.get("uploaded"):
        client.upload(str(transfer["url"]), Path(args.bundle))
    completed = client.request(f"/api/v1/bundle-imports/{prepared['uploadId']}/complete", {}, method="POST")
    completed["credentialSource"] = source
    if args.no_wait:
        return completed
    status = client.wait_for_import(str(completed["statusUrl"]), args.timeout)
    if status.get("state") in {"failed", "rejected", "cancelled"}:
        raise ApiError(409, str(status.get("error") or "bundle import failed"), str(status.get("code") or "import_failed"))
    return status


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="miniverse", description="Validate and upload Miniverse .dhsim bundles.")
    root.add_argument("--origin", help="Miniverse API origin; defaults to MINIVERSE_ORIGIN or https://miniverse.bot")
    root.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    commands = root.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version")
    version.add_argument("--json", action="store_true", dest="command_json")
    help_command = commands.add_parser("agent-help")
    help_command.add_argument("topic", nargs="?", choices=sorted(TOPICS))
    help_command.add_argument("--all", action="store_true")
    auth = commands.add_parser("auth")
    auth_commands = auth.add_subparsers(dest="auth_command", required=True)
    login = auth_commands.add_parser("login")
    login.add_argument("--no-browser", action="store_true")
    for leaf in (login, auth_commands.add_parser("status"), auth_commands.add_parser("logout")):
        leaf.add_argument("--json", action="store_true", dest="command_json")
    bundle = commands.add_parser("bundle")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    validate = bundle_commands.add_parser("validate")
    validate.add_argument("bundle")
    validate.add_argument("--strict", action="store_true", help="Promote optimization warnings to validation errors")
    validate.add_argument("--json", action="store_true", dest="command_json")
    inspect = bundle_commands.add_parser("inspect")
    inspect.add_argument("bundle")
    inspect.add_argument("--json", action="store_true", dest="command_json")
    upload = bundle_commands.add_parser("upload")
    upload.add_argument("bundle")
    upload.add_argument("--idempotency-key")
    upload.add_argument("--no-wait", action="store_true")
    upload.add_argument("--timeout", type=int, default=3600)
    upload.add_argument("--json", action="store_true", dest="command_json")
    model = commands.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_validate = model_commands.add_parser("validate", help="Validate one ONNX checkpoint and its Miniverse contract")
    model_validate.add_argument("model")
    model_validate.add_argument("--strict", action="store_true", help="Promote optimization warnings to validation errors")
    model_validate.add_argument("--json", action="store_true", dest="command_json")
    status = bundle_commands.add_parser("status")
    status.add_argument("upload_id")
    status.add_argument("--json", action="store_true", dest="command_json")
    list_command = bundle_commands.add_parser("list")
    list_command.add_argument("--json", action="store_true", dest="command_json")
    publish = bundle_commands.add_parser("publish")
    publish.add_argument("bundle_version", help="Bundle version as <id>@<digest>")
    publish.add_argument("--json", action="store_true", dest="command_json")
    return root


@dataclass(frozen=True)
class BundleCommandResult:
    value: dict[str, Any]
    exit_code: int
    inspection: Any


def _model_result(validation: ModelValidation, *, strict: bool, model_id: str | None = None) -> CommandResult:
    errors = [issue.as_dict() for issue in validation.errors]
    warnings = [issue.as_dict() for issue in validation.warnings]
    if strict:
        errors.extend({**warning, "severity": "error", "promotedByStrict": True} for warning in warnings)
    value = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "models": {model_id: validation.as_dict()} if model_id else {},
    }
    if not model_id:
        value.update({"precision": validation.precision, "compatibility": validation.compatibility})
    return CommandResult(value, 2 if errors else 0)


def validate_bundle(path: str | Path, *, strict: bool = False) -> BundleCommandResult:
    inspected = inspect_bundle(path)
    models: dict[str, ModelValidation] = {}
    with tempfile.TemporaryDirectory(prefix="miniverse-sdk-validate-") as directory, zipfile.ZipFile(path) as archive:
        root = Path(directory)
        for value in inspected.manifest.get("models", ()):
            model_id = str(value["id"])
            model_path = root / f"{model_id}.onnx"
            with archive.open(f"models/{model_id}.onnx") as source, model_path.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            models[model_id] = validate_model(model_path)

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for model_id, validation in models.items():
        errors.extend({**issue.as_dict(), "model": model_id} for issue in validation.errors)
        warnings.extend({**issue.as_dict(), "model": model_id} for issue in validation.warnings)
    errors.extend(issue.as_dict() for issue in validate_bundle_model_backends(inspected.manifest, models))
    if strict:
        errors.extend({**warning, "severity": "error", "promotedByStrict": True} for warning in warnings)
    inspection = inspected.as_dict()
    inspection.pop("model_findings", None)
    value = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "models": {model_id: validation.as_dict() for model_id, validation in models.items()},
        **inspection,
    }
    return BundleCommandResult(value, 2 if errors else 0, inspected)


def run(args: argparse.Namespace) -> Any:
    if args.command == "version":
        return {"distribution": "miniverse-sdk", "version": __version__, "cli": "miniverse", "apiVersion": "v1"}
    if args.command == "agent-help":
        return agent_help(args.topic, args.all)
    if args.command == "auth":
        if args.auth_command == "login":
            return auth_login(args)
        if args.auth_command == "logout":
            api_origin = origin(args.origin)
            saved, _ = credential(api_origin)
            if isinstance(saved, OAuthCredential) and saved.refresh_token:
                try:
                    Client(api_origin, None).request_form("/api/auth/oauth2/revoke", {
                        "token": saved.refresh_token,
                        "token_type_hint": "refresh_token",
                        "client_id": saved.client_id,
                    })
                except ApiError:
                    pass
            delete_oauth_credential(api_origin)
            return {"authenticated": False, "source": "none"}
        api_origin = origin(args.origin)
        saved, source = credential(api_origin)
        return {
            "authenticated": bool(saved),
            "source": source,
            **({
                "storage": auth_store(),
                "renewable": isinstance(saved, OAuthCredential) and saved.renewable,
                "expiresAt": saved.expires_at if isinstance(saved, OAuthCredential) else None,
                **({"path": str(auth_file())} if auth_store() == "file" else {}),
            } if saved and source == "oauth" else {}),
        }
    if args.command == "model":
        path = Path(args.model)
        return _model_result(validate_model(path), strict=args.strict, model_id=path.stem)
    if args.command == "bundle":
        if args.bundle_command == "validate":
            validated = validate_bundle(args.bundle, strict=args.strict)
            return CommandResult(validated.value, validated.exit_code)
        if args.bundle_command == "inspect":
            inspected = inspect_bundle(args.bundle)
            return {"ok": True, **inspected.as_dict(include_manifest=True)}
        if args.bundle_command == "upload":
            return bundle_upload(args)
        api_origin = origin(args.origin)
        saved, _ = credential(api_origin)
        client = Client(api_origin, saved)
        if args.bundle_command == "status":
            return client.request(f"/api/v1/bundle-imports/{urllib.parse.quote(args.upload_id)}")
        if args.bundle_command == "list":
            return client.request("/api/v1/bundle-imports")
        bundle_id, separator, digest = args.bundle_version.partition("@")
        if not separator:
            raise BundleValidationError("invalid_bundle_version", "bundle version must be <id>@<digest>")
        return client.request(f"/api/v1/bundles/{urllib.parse.quote(bundle_id)}/{urllib.parse.quote(digest)}/publish", {})
    raise RuntimeError("unreachable command")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        value = run(args)
        exit_code = value.exit_code if isinstance(value, CommandResult) else 0
        if isinstance(value, CommandResult):
            value = value.value
        if isinstance(value, str):
            print(value, end="")
        else:
            emit(value, bool(args.json or getattr(args, "command_json", False)))
        return exit_code
    except BundleValidationError as error:
        details = getattr(error, "details", None)
        if isinstance(details, dict) and {"ok", "errors", "warnings"} <= set(details):
            emit(details, True)
        else:
            issue = {"code": error.code, "severity": "error", "message": str(error)}
            emit({"ok": False, "errors": details if isinstance(details, list) else [issue], "warnings": [], "models": {}}, True)
        return 2
    except ApiError as error:
        emit({"ok": False, "code": error.code, "status": error.status, "error": str(error)}, True)
        return 3
    except (OSError, ValueError) as error:
        emit({"ok": False, "code": "local_error", "error": str(error)}, True)
        return 1
