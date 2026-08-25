from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from miniverse_sdk.bundles import BundleValidationError, inspect_bundle
from miniverse_sdk.api import Client
from miniverse_sdk.cli import agent_help, main
from miniverse_sdk.config import OAuthCredential, credential, delete_oauth_credential, load_oauth_credential, save_oauth_credential


def _varint(value: int) -> bytes:
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _metadata_entry(key: str, value: str) -> bytes:
    entry = _field(1, key.encode()) + _field(2, value.encode())
    return _field(14, entry)


def _varint_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _node(op_type: str, inputs: tuple[str, ...] = (), outputs: tuple[str, ...] = (), attributes: bytes = b"") -> bytes:
    encoded = b"".join(_field(1, name.encode()) for name in inputs)
    encoded += b"".join(_field(2, name.encode()) for name in outputs)
    encoded += _field(4, op_type.encode()) + attributes
    return _field(1, encoded)


def _int64_initializer(name: str, *values: int) -> bytes:
    tensor = _varint_field(2, 7) + _field(8, name.encode()) + _field(9, b"".join(struct.pack("<q", value) for value in values))
    return _field(5, tensor)


def _topk_graph(k_value: int | None, *, constant_node: bool = False) -> bytes:
    graph = _node("TopK", inputs=("logits", "k"), outputs=("values", "indices"))
    if k_value is None:
        return graph
    if constant_node:
        tensor = _varint_field(2, 7) + _field(9, struct.pack("<q", k_value))
        attribute = _field(5, _field(1, b"value") + _field(5, tensor))
        return graph + _node("Constant", outputs=("k",), attributes=attribute)
    return graph + _int64_initializer("k", k_value)


def model_with_precision(precision: str = "fp16", graph: bytes | None = None, opset: int = 18) -> bytes:
    contract = json.dumps({"schemaVersion": "0.3", "precision": precision}, separators=(",", ":"))
    return b"".join((
        _field(7, graph or b""),
        _field(8, _field(1, b"") + _varint_field(2, opset)),
        _metadata_entry("com.dollhouserobotics.miniverse.simulation_contract", contract),
        _metadata_entry("com.dollhouserobotics.miniverse.simulation_contract_schema_version", "0.3"),
    ))


def fixture(path: Path, model: bytes | None = None, *, legacy_policy_bindings: bool = False, legacy_dynamics_overrides: bool = False) -> Path:
    program = b"class Policy:\n    pass\n"
    embodiment = b"mjcf-archive"
    model = model_with_precision() if model is None else model
    manifest = {
        "version": "v1", "id": "fixture", "name": "Fixture", "primarySimulator": "mujoco",
        "embodiment": {}, "models": [{"id": "policy"}],
        "program": {"apiVersion": "dhr.python-policy/v1", "entrypoint": "policy:Policy"},
    }
    if legacy_policy_bindings:
        manifest["policyBindings"] = {"source": "embedded-model-contract", "modelId": "policy"}
    if legacy_dynamics_overrides:
        manifest["embodiment"]["dynamicsOverrides"] = []
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle.json", json.dumps(manifest))
        archive.writestr("policy.py", program)
        archive.writestr("embodiment/mjcf.zip", embodiment)
        archive.writestr("models/policy.onnx", model)
    return path


class Handler(BaseHTTPRequestHandler):
    archive = b""
    state = "created"
    token = ""
    conditional = ""
    request_fields = set()

    def log_message(self, *_args):
        pass

    def _json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.headers.get("authorization") != "Bearer test-token":
            return self._json({"error": "unauthorized", "code": "access_required"}, 401)
        length = int(self.headers.get("content-length", "0"))
        value = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/v1/bundles/fixture/revisions":
            Handler.request_fields = set(value)
            Handler.token = value["archiveSha256"]
            return self._json({"revisionId": "brv_" + "1" * 32, "uploaded": False, "transfer": {"mode": "single", "url": f"http://127.0.0.1:{self.server.server_port}/r2/source", "headers": {"content-type": "application/vnd.dhr.simulation-bundle+zip", "if-none-match": "*"}}, "statusUrl": "/api/v1/bundles/fixture/revisions/brv_" + "1" * 32}, 201)
        return self._json({"error": "not found"}, 404)

    def do_PUT(self):
        length = int(self.headers.get("content-length", "0"))
        Handler.archive = self.rfile.read(length)
        Handler.conditional = self.headers.get("if-none-match", "")
        Handler.state = "ready"
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/v1/bundles/fixture/revisions/brv_" + "1" * 32:
            return self._json({"revisionId": "brv_" + "1" * 32, "state": Handler.state, "archiveSha256": Handler.token, "bundleId": "fixture", "bundleDigest": "d" * 64})
        return self._json({"error": "not found"}, 404)


class RefreshHandler(BaseHTTPRequestHandler):
    refreshes = 0

    def log_message(self, *_args):
        pass

    def _json(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        form = dict(item.split("=", 1) for item in self.rfile.read(length).decode().split("&"))
        if self.path != "/api/auth/oauth2/token" or form.get("grant_type") != "refresh_token":
            return self._json({"error": "not found"}, 404)
        RefreshHandler.refreshes += 1
        return self._json({
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 3600,
            "scope": "offline_access bundles:upload",
        })

    def do_GET(self):
        if self.path != "/protected":
            return self._json({"error": "not found"}, 404)
        if self.headers.get("authorization") != "Bearer access-2":
            return self._json({"error": "expired", "code": "access_required"}, 401)
        return self._json({"ok": True})

class CliTest(unittest.TestCase):
    def test_operation_lint_names_supporting_simulators_without_capability_declarations(self):
        from miniverse_sdk.onnx_metadata import compatibility_report

        report = compatibility_report({
            "backends": [{"id": "isaac-sim-cpu-physx"}],
            "inputs": [{"name": "obs", "slices": [{"provider": "contacts", "component": "normalForce"}]}],
        })
        row = report["inputs"]["obs"][0]
        self.assertEqual(row["operation"], "contacts.normalForce")
        self.assertEqual(row["supportedSimulators"], ["mujoco-cpu"])
        self.assertEqual(row["isaac-sim-cpu-physx"], "unsupported")
        self.assertEqual(report["incompatible"], ["obs:contacts.normalForce@isaac-sim-cpu-physx"])

    def test_validate_and_agent_help(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.dhsim")
            inspected = inspect_bundle(path)
            self.assertEqual(inspected.bundle_id, "fixture")
            self.assertEqual(len(inspected.assets), 3)
            self.assertEqual(inspected.model_precisions, {"policy": "fp16"})
        self.assertIn("MINIVERSE_API_TOKEN", agent_help("auth", False))
        self.assertIn("object-create event starts server-side", agent_help("upload", False))

    def test_bundle_rejects_removed_policy_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "legacy.dhsim", legacy_policy_bindings=True)
            with self.assertRaisesRegex(BundleValidationError, "fields were removed: policyBindings"):
                inspect_bundle(path)

    def test_bundle_rejects_removed_actuator_dynamics_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "legacy.dhsim", legacy_dynamics_overrides=True)
            with self.assertRaisesRegex(BundleValidationError, "dynamicsOverrides was removed; author actuator gains and limits"):
                inspect_bundle(path)

    def test_tensorrt_compat_findings(self):
        from io import BytesIO

        from miniverse_sdk.onnx_compat import scan_model

        scanned = scan_model(BytesIO(model_with_precision(graph=_topk_graph(64000))))
        self.assertEqual([finding.code for finding in scanned.findings], ["tensorrt_topk_k_limit"])
        self.assertEqual(scanned.precision, "fp16")
        scanned = scan_model(BytesIO(model_with_precision(graph=_topk_graph(4096, constant_node=True))))
        self.assertEqual([finding.code for finding in scanned.findings], ["tensorrt_topk_k_limit"])
        scanned = scan_model(BytesIO(model_with_precision(graph=_topk_graph(3840))))
        self.assertEqual(scanned.findings, ())
        scanned = scan_model(BytesIO(model_with_precision(graph=_topk_graph(None))))
        self.assertEqual([finding.code for finding in scanned.findings], ["tensorrt_topk_dynamic_k"])
        attribute = _field(5, _field(1, b"k") + _varint_field(3, 64000))
        graph = _node("TopK", inputs=("logits",), outputs=("values", "indices"), attributes=attribute)
        scanned = scan_model(BytesIO(model_with_precision(graph=graph, opset=9)))
        self.assertEqual([finding.code for finding in scanned.findings], ["tensorrt_topk_k_limit"])

    def test_bundle_validate_reports_findings_and_strict_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.dhsim", model=model_with_precision(graph=_topk_graph(64000)))
            inspected = inspect_bundle(path)
            self.assertEqual([finding.code for finding in inspected.model_findings["policy"]], ["tensorrt_topk_k_limit"])
            self.assertEqual(main(["bundle", "validate", str(path), "--json"]), 0)
            self.assertEqual(main(["bundle", "validate", str(path), "--strict"]), 2)
            clean = fixture(Path(directory) / "clean.dhsim", model=model_with_precision(graph=_topk_graph(3840)))
            self.assertEqual(main(["bundle", "validate", str(clean), "--strict"]), 0)

    def test_model_validate_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            path.write_bytes(model_with_precision(graph=_topk_graph(64000)))
            self.assertEqual(main(["model", "validate", str(path), "--json"]), 0)
            self.assertEqual(main(["model", "validate", str(path), "--strict"]), 2)
            path.write_bytes(model_with_precision(graph=_topk_graph(3840)))
            self.assertEqual(main(["model", "validate", str(path), "--strict"]), 0)

    def test_missing_and_invalid_precision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.dhsim")
            with zipfile.ZipFile(path, "r") as archive:
                values = {name: archive.read(name) for name in archive.namelist()}
            for precision in ("tf32", None):
                model = model_with_precision(str(precision)) if precision is not None else _metadata_entry(
                    "com.dollhouserobotics.miniverse.simulation_contract_schema_version", "0.3"
                )
                values["models/policy.onnx"] = model
                values["bundle.json"] = json.dumps({
                    **json.loads(values["bundle.json"]),
                    "models": [{"id": "policy"}],
                }).encode()
                with zipfile.ZipFile(path, "w") as archive:
                    for name, value in values.items():
                        archive.writestr(name, value)
                with self.assertRaises(BundleValidationError) as caught:
                    inspect_bundle(path)
                self.assertEqual(caught.exception.code, "invalid_model_metadata")

    def test_environment_token_is_the_noninteractive_auth_contract(self):
        with patch.dict(os.environ, {"MINIVERSE_API_TOKEN": "environment-token"}, clear=False):
            self.assertEqual(credential(), ("environment-token", "MINIVERSE_API_TOKEN"))

    def test_oauth_file_is_private_and_refresh_rotation_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state" / "auth.json"
            server = ThreadingHTTPServer(("127.0.0.1", 0), RefreshHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            api_origin = f"http://127.0.0.1:{server.server_port}"
            RefreshHandler.refreshes = 0
            try:
                with patch.dict(os.environ, {
                    "MINIVERSE_AUTH_FILE": str(path),
                    "MINIVERSE_AUTH_STORE": "file",
                    "MINIVERSE_API_TOKEN": "",
                }, clear=False):
                    saved = OAuthCredential(
                        access_token="access-1",
                        refresh_token="refresh-1",
                        expires_at=None,
                        origin=api_origin,
                    )
                    self.assertEqual(save_oauth_credential(saved), "file")
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                    self.assertEqual(Client(api_origin, saved).request("/protected"), {"ok": True})
                    rotated = load_oauth_credential(api_origin)
                    self.assertIsNotNone(rotated)
                    self.assertEqual(rotated.access_token, "access-2")
                    self.assertEqual(rotated.refresh_token, "refresh-2")
                    self.assertEqual(RefreshHandler.refreshes, 1)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()

    def test_file_store_does_not_read_legacy_keyring_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            with patch.dict(os.environ, {
                "MINIVERSE_AUTH_FILE": str(path),
                "MINIVERSE_AUTH_STORE": "file",
                "MINIVERSE_API_TOKEN": "",
            }, clear=False), patch("miniverse_sdk.config._keyring_get", return_value="legacy-access") as get_keyring, patch("miniverse_sdk.config._keyring_delete") as delete_keyring:
                self.assertIsNone(load_oauth_credential("https://miniverse.bot"))
                delete_oauth_credential("https://miniverse.bot")
                self.assertFalse(path.exists())
                get_keyring.assert_not_called()
                delete_keyring.assert_not_called()

    def test_rejects_undeclared_and_traversal_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.dhsim")
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("extra.txt", "no")
            with self.assertRaises(BundleValidationError) as caught:
                inspect_bundle(path)
            self.assertEqual(caught.exception.code, "undeclared_member")

    def test_end_to_end_archive_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.dhsim")
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.dict(os.environ, {"MINIVERSE_API_TOKEN": "test-token"}, clear=False):
                    code = main(["--origin", f"http://127.0.0.1:{server.server_port}", "--json", "bundle", "upload", str(path)])
                self.assertEqual(code, 0)
                self.assertEqual(hashlib.sha256(Handler.archive).hexdigest(), Handler.token)
                self.assertEqual(Handler.conditional, "*")
                self.assertEqual(Handler.request_fields, {"archiveSha256", "bytes", "filename", "idempotencyKey"})
            finally:
                server.shutdown()
                thread.join()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
