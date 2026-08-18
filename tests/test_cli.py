from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from miniverse_sdk.bundles import BundleValidationError, inspect_bundle
from miniverse_sdk.cli import agent_help, main
from miniverse_sdk.config import credential


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
    contract = json.dumps({"schemaVersion": "0.2", "precision": precision}, separators=(",", ":"))
    return b"".join((
        _field(7, graph or b""),
        _field(8, _field(1, b"") + _varint_field(2, opset)),
        _metadata_entry("com.dollhouserobotics.miniverse.simulation_contract", contract),
        _metadata_entry("com.dollhouserobotics.miniverse.simulation_contract_schema_version", "0.2"),
    ))


def fixture(path: Path, model: bytes | None = None) -> Path:
    program = b"class Policy:\n    pass\n"
    scene = b"glb-scene"
    model = model_with_precision() if model is None else model
    digest = lambda value: hashlib.sha256(value).hexdigest()
    manifest = {
        "version": "dhr.simulation-bundle/v1", "id": "fixture", "name": "Fixture",
        "primarySimulator": "mujoco", "primaryModel": "policy",
        "scene": {"sha256": digest(scene)},
        "models": [{"id": "policy", "sha256": digest(model)}],
        "program": {"sourceSha256": digest(program)},
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle.json", json.dumps(manifest))
        archive.writestr("policy.py", program)
        archive.writestr("scene.glb", scene)
        archive.writestr("models/policy.onnx", model)
    return path


class Handler(BaseHTTPRequestHandler):
    archive = b""
    state = "created"
    token = ""

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
        if self.path == "/api/v1/bundle-imports":
            Handler.token = value["archiveSha256"]
            return self._json({"uploadId": "bup_fixture", "uploaded": False, "transfer": {"mode": "single", "url": f"http://127.0.0.1:{self.server.server_port}/r2/source"}, "statusUrl": "/api/v1/bundle-imports/bup_fixture"}, 201)
        if self.path.endswith("/complete"):
            Handler.state = "ready"
            return self._json({"uploadId": "bup_fixture", "state": "ready", "statusUrl": "/api/v1/bundle-imports/bup_fixture"}, 202)
        return self._json({"error": "not found"}, 404)

    def do_PUT(self):
        length = int(self.headers.get("content-length", "0"))
        Handler.archive = self.rfile.read(length)
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/v1/bundle-imports/bup_fixture":
            return self._json({"uploadId": "bup_fixture", "state": Handler.state, "archiveSha256": Handler.token, "bundleId": "fixture", "bundleDigest": "d" * 64})
        return self._json({"error": "not found"}, 404)


class CliTest(unittest.TestCase):
    def test_validate_and_agent_help(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.dhsim")
            inspected = inspect_bundle(path)
            self.assertEqual(inspected.bundle_id, "fixture")
            self.assertEqual(len(inspected.assets), 3)
            self.assertEqual(inspected.model_precisions, {"policy": "fp16"})
        self.assertIn("MINIVERSE_API_TOKEN", agent_help("auth", False))
        self.assertIn("server verifies and expands", agent_help("upload", False))

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
                    "com.dollhouserobotics.miniverse.simulation_contract_schema_version", "0.2"
                )
                values["models/policy.onnx"] = model
                values["bundle.json"] = json.dumps({
                    **json.loads(values["bundle.json"]),
                    "models": [{"id": "policy", "sha256": hashlib.sha256(model).hexdigest()}],
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
            finally:
                server.shutdown()
                thread.join()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
