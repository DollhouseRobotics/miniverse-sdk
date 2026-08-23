"""Command-line entrypoint for miniverse-sdk."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import webbrowser
from importlib import resources
from pathlib import Path
from typing import Any

from dataclasses import asdict

from . import __version__
from .api import ApiError, Client
from .bundles import BundleValidationError, inspect_bundle
from .config import OAuthCredential, auth_file, auth_store, credential, delete_oauth_credential, origin, save_oauth_credential
from .onnx_compat import CompatFinding, scan_model
from .onnx_metadata import OnnxMetadataError

TOPICS = {"auth", "bundles", "upload", "sessions", "onnx"}


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
    inspected = inspect_bundle(args.bundle)
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
    validate.add_argument("--strict", action="store_true", help="Fail when any model compatibility finding is present")
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
    model_validate = model_commands.add_parser("validate", help="Lint one ONNX checkpoint for metadata and TensorRT compatibility")
    model_validate.add_argument("model")
    model_validate.add_argument("--strict", action="store_true", help="Fail when any compatibility finding is present")
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


def _strict_failure(model_findings: dict[str, tuple[CompatFinding, ...]]) -> BundleValidationError:
    details = {model: [asdict(finding) for finding in findings] for model, findings in model_findings.items()}
    total = sum(len(findings) for findings in details.values())
    error = BundleValidationError("model_compatibility", f"strict validation found {total} model compatibility finding(s)")
    error.details = details
    return error


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
        if not path.is_file():
            raise BundleValidationError("model_missing", f"model does not exist: {path}")
        try:
            with path.open("rb") as source:
                scanned = scan_model(source)
        except OnnxMetadataError as error:
            raise BundleValidationError("invalid_model_metadata", str(error)) from error
        if args.strict and scanned.findings:
            raise _strict_failure({path.stem: scanned.findings})
        from .onnx_metadata import compatibility_report

        compatibility = compatibility_report(scanned.contract or {})
        return {
            "ok": not scanned.findings and not compatibility["incompatible"],
            "precision": scanned.precision,
            "findings": [asdict(finding) for finding in scanned.findings],
            "compatibility": compatibility,
        }
    if args.command == "bundle":
        if args.bundle_command in {"validate", "inspect"}:
            inspected = inspect_bundle(args.bundle)
            if getattr(args, "strict", False) and inspected.model_findings:
                raise _strict_failure(inspected.model_findings)
            return {"ok": True, **inspected.as_dict(include_manifest=args.bundle_command == "inspect")}
        api_origin = origin(args.origin)
        saved, _ = credential(api_origin)
        client = Client(api_origin, saved)
        if args.bundle_command == "upload":
            return bundle_upload(args)
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
        if isinstance(value, str):
            print(value, end="")
        else:
            emit(value, bool(args.json or getattr(args, "command_json", False)))
        return 0
    except BundleValidationError as error:
        details = getattr(error, "details", None)
        emit({"ok": False, "code": error.code, "error": str(error), **({"findings": details} if details else {})}, True)
        return 2
    except ApiError as error:
        emit({"ok": False, "code": error.code, "status": error.status, "error": str(error)}, True)
        return 3
    except (OSError, ValueError) as error:
        emit({"ok": False, "code": "local_error", "error": str(error)}, True)
        return 1
