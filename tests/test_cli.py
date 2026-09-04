from __future__ import annotations

import hashlib
import io
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

from miniverse_sdk.bundles import BundleValidationError, _validate_environment_mjcf, inspect_bundle
from miniverse_sdk.api import Client
from miniverse_sdk.cli import agent_help, main, parser
from miniverse_sdk.config import OAuthCredential, credential, delete_oauth_credential, load_oauth_credential, save_oauth_credential
from miniverse_sdk.onnx_metadata import ONNX_HASH_KEY, ONNX_METADATA_KEY, ONNX_SCHEMA_KEY
from miniverse_sdk.terrain import build_heightfield_glb, heightfield_size_warnings, inspect_heightfield_glb


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
    contract = json.dumps({
        "schemaVersion": "0.3",
        "precision": precision,
        "contractHash": "c" * 64,
        "modelSha256": "d" * 64,
        "skeletonId": "fixture",
        "compatibleSceneContractHashes": ["e" * 64],
        "backends": [{"id": "mujoco-cpu", "versionRange": ">=3.3,<4", "providers": ["CPUExecutionProvider"]}],
        "opset": opset,
        "execution": {"kind": "singlePolicyStep"},
        "rates": {"physicsHz": 60, "policyHz": 30, "publishHz": 30, "actionHold": "zero-order-hold", "commandBoundary": "policy", "controlLoop": "policy-then-decimation"},
        "inputs": [{"name": "obs", "dtype": "float32", "shape": [1, 1], "slices": [{"start": 0, "length": 1, "provider": "constant", "value": [0]}]}],
        "outputs": [{"name": "actuator_targets", "dtype": "float32", "shape": [1, 1], "role": "actuatorTargets", "actuators": ["motor"], "clip": [-1, 1], "failsafe": [0]}],
        "commands": [],
        "stateEstimation": {"mode": "simulator-ground-truth"},
    }, separators=(",", ":"))
    return b"".join((
        _field(7, graph or b""),
        _field(8, _field(1, b"") + _varint_field(2, opset)),
        _metadata_entry("com.dollhouserobotics.miniverse.simulation_contract", contract),
        _metadata_entry("com.dollhouserobotics.miniverse.simulation_contract_schema_version", "0.3"),
    ))


def valid_model(precision: str = "fp16", *, topk_k: int | None = None, incompatible: bool = False, opset: int = 18) -> bytes:
    import onnx
    from onnx import TensorProto, helper

    if topk_k is None:
        width = 1
        graph = helper.make_graph(
            [helper.make_node("Identity", ["obs"], ["actuator_targets"])],
            "fixture",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("actuator_targets", TensorProto.FLOAT, [1, width])],
        )
        outputs = [{
            "name": "actuator_targets", "dtype": "float32", "shape": [1, width], "role": "actuatorTargets",
            "actuators": ["motor"], "controlModes": ["position"], "actuatorRanges": [[-1, 1]], "clip": [-1, 1], "failsafe": [0],
        }]
    else:
        width = topk_k
        graph = helper.make_graph(
            [helper.make_node("TopK", ["obs", "k"], ["actuator_targets", "indices"], axis=1)],
            "fixture-topk",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
            [
                helper.make_tensor_value_info("actuator_targets", TensorProto.FLOAT, [1, width]),
                helper.make_tensor_value_info("indices", TensorProto.INT64, [1, width]),
            ],
            [helper.make_tensor("k", TensorProto.INT64, [1], [width])],
        )
        actuators = [f"motor-{index}" for index in range(width)]
        outputs = [
            {
                "name": "actuator_targets", "dtype": "float32", "shape": [1, width], "role": "actuatorTargets",
                "actuators": actuators, "controlModes": ["position"] * width, "actuatorRanges": [[-1, 1]] * width,
                "clip": [-1, 1], "failsafe": [0] * width,
            },
            {"name": "indices", "dtype": "int64", "shape": [1, width], "role": "auxiliary"},
        ]
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    clone = onnx.ModelProto(); clone.CopyFrom(model); del clone.metadata_props[:]
    model_hash = hashlib.sha256(clone.SerializeToString(deterministic=True)).hexdigest()
    source = {
        "start": 0, "length": width,
        **({"provider": "contacts", "component": "normalForce", "ids": ["foot"]} if incompatible else {"provider": "constant", "value": [0] * width}),
    }
    contract = {
        "schemaVersion": "0.3", "precision": precision, "modelSha256": model_hash,
        "skeletonId": "fixture", "compatibleSceneContractHashes": ["e" * 64],
        "backends": [{
            "id": "isaac-sim-cpu-physx" if incompatible else "mujoco-cpu",
            "versionRange": ">=5.1,<5.2" if incompatible else ">=3.3,<4",
            "providers": ["CPUExecutionProvider"],
        }],
        "opset": opset, "execution": {"kind": "singlePolicyStep"},
        "rates": {"physicsHz": 60, "policyHz": 30, "publishHz": 30, "actionHold": "zero-order-hold", "commandBoundary": "policy", "controlLoop": "policy-then-decimation"},
        "inputs": [{"name": "obs", "dtype": "float32", "shape": [1, width], "slices": [source]}],
        "outputs": outputs, "commands": [], "stateEstimation": {"mode": "simulator-ground-truth"},
    }
    contract["contractHash"] = hashlib.sha256(json.dumps(
        contract, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode()).hexdigest()
    helper.set_model_props(model, {
        ONNX_METADATA_KEY: json.dumps(contract, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
        ONNX_SCHEMA_KEY: "0.3",
        ONNX_HASH_KEY: contract["contractHash"],
    })
    return model.SerializeToString(deterministic=True)


def fixture(path: Path, model: bytes | None = None, *, legacy_policy_bindings: bool = False, legacy_dynamics_overrides: bool = False, primary_simulator: str = "mujoco") -> Path:
    program = b"class Policy:\n    pass\n"
    embodiment = b'<mujoco model="fixture"><worldbody><body name="robot"/></worldbody></mujoco>'
    model = valid_model() if model is None else model
    manifest = {
        "version": "v1", "id": "fixture", "name": "Fixture", "primarySimulator": primary_simulator,
        "embodiment": {"kind": "mjcf", "path": "embodiment/robot.xml"}, "models": [{"id": "policy"}],
        "program": {"apiVersion": "dhr.python-policy/v1", "entrypoint": "policy:Policy"},
    }
    if legacy_policy_bindings:
        manifest["policyBindings"] = {"source": "embedded-model-contract", "modelId": "policy"}
    if legacy_dynamics_overrides:
        manifest["embodiment"]["dynamicsOverrides"] = []
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bundle.json", json.dumps(manifest))
        archive.writestr("policy.py", program)
        archive.writestr("embodiment/robot.xml", embodiment)
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
    def test_terrain_build_creates_the_canonical_data_only_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heights = root / "heights.json"
            output = root / "environment" / "terrain.glb"
            heights.write_text("[[0, 1, 2], [3, 4, 5]]")
            self.assertEqual(main([
                "terrain", "build", str(heights), str(output), "--id", "climb-001",
                "--cell-size", "0.25", "0.5", "--origin", "-0.25", "1", "0.1",
                "--vertical-scale", "0.2", "--vertical-offset", "-0.1", "--out-of-bounds", "clamp", "--json",
            ]), 0)
            inspected = inspect_heightfield_glb(output.read_bytes())
            self.assertEqual((inspected.id, inspected.width, inspected.height), ("climb-001", 3, 2))
            self.assertEqual(inspected.xy_resolution, (0.25, 0.5))
            self.assertEqual(main(["terrain", "build", str(heights), str(output), "--cell-size", "1", "1"]), 2)

    def test_terrain_build_accepts_a_two_dimensional_numpy_array(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heights = root / "heights.npy"
            output = root / "environment" / "terrain.glb"
            np.save(heights, np.asarray([[0, 0.1], [0.2, 0.3]], dtype=np.float32))
            self.assertEqual(main([
                "terrain", "build", str(heights), str(output),
                "--id", "numpy-grid", "--cell-size", "0.1", "0.2", "--json",
            ]), 0)
            inspected = inspect_heightfield_glb(output.read_bytes())
            self.assertEqual((inspected.width, inspected.height), (2, 2))
            self.assertEqual(inspected.xy_resolution, (0.1, 0.2))

    def test_bundle_inspection_accepts_a_declared_heightfield_glb_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "terrain.mini")
            with zipfile.ZipFile(path, "r") as archive:
                values = {name: archive.read(name) for name in archive.namelist()}
            manifest = json.loads(values["bundle.json"])
            environment_path = "environment/terrains/steps.glb"
            manifest["environment"] = {"kind": "glb", "path": environment_path}
            values["bundle.json"] = json.dumps(manifest).encode()
            values[environment_path] = build_heightfield_glb(
                terrain_id="steps", width=2, height=2, heights=[0, 0, 0.2, 0.2], xy_resolution=[0.1, 0.1], out_of_bounds="clamp",
            )
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            inspected = inspect_bundle(path)
            terrain = next(asset for asset in inspected.assets if asset.kind == "scene")
            self.assertEqual(terrain.path, environment_path)
            self.assertEqual(terrain.heightfield["id"], "steps")

            manifest["environment"]["path"] = "../steps.glb"
            values["bundle.json"] = json.dumps(manifest).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            with self.assertRaisesRegex(BundleValidationError, "schema error"):
                inspect_bundle(path)

    def test_bundle_embodiment_compiles_the_exact_mjcf_dependency_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.mini")
            with zipfile.ZipFile(path, "r") as archive:
                values = {name: archive.read(name) for name in archive.namelist()}
            values["embodiment/robot.xml"] = b'<mujoco><include file="parts/body.xml"/></mujoco>'
            values["embodiment/parts/body.xml"] = b'<mujocoinclude><worldbody><body name="robot"/></worldbody></mujocoinclude>'
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            inspected = inspect_bundle(path)
            self.assertEqual(next(asset for asset in inspected.assets if asset.kind == "embodiment").path, "embodiment/robot.xml")

            values["embodiment/unused.xml"] = b"<mujoco/>"
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            with self.assertRaisesRegex(BundleValidationError, "unused embodiment members"):
                inspect_bundle(path)

            del values["embodiment/unused.xml"]
            del values["embodiment/parts/body.xml"]
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            with self.assertRaisesRegex(BundleValidationError, "dependency is missing"):
                inspect_bundle(path)

    def test_bundle_embodiment_rejects_dependency_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.mini")
            with zipfile.ZipFile(path, "r") as archive:
                values = {name: archive.read(name) for name in archive.namelist()}
            values["embodiment/robot.xml"] = b'<mujoco><include file="../../escape.xml"/></mujoco>'
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            with self.assertRaisesRegex(BundleValidationError, "escapes the embodiment directory"):
                inspect_bundle(path)

    def test_bundle_mjcf_environment_compiles_its_exact_dependency_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.mini")
            with zipfile.ZipFile(path, "r") as archive:
                values = {name: archive.read(name) for name in archive.namelist()}
            manifest = json.loads(values["bundle.json"])
            manifest["primarySimulator"] = "isaac-sim-cpu-physx"
            manifest["compatibleSimulators"] = ["mujoco", "isaac-sim-gpu-physx"]
            manifest["environment"] = {"kind": "mjcf", "path": "environment/world.xml"}
            values["bundle.json"] = json.dumps(manifest).encode()
            values["environment/world.xml"] = b'<mujoco><include file="parts/swing.xml"/><worldbody><geom type="plane" size="5 5 .1"/></worldbody></mujoco>'
            values["environment/parts/swing.xml"] = b'<mujocoinclude><worldbody><body name="swing"><joint name="hinge"/><geom type="box" size=".1 .1 .5"/></body></worldbody></mujocoinclude>'
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            inspected = inspect_bundle(path)
            scene = next(asset for asset in inspected.assets if asset.kind == "scene")
            self.assertEqual(scene.path, "environment/world.xml")
            self.assertEqual(scene.kind, "scene")

            values["environment/unused.xml"] = b"<mujoco/>"
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            with self.assertRaisesRegex(BundleValidationError, "unused environment members"):
                inspect_bundle(path)

    def test_bundle_mjcf_environment_enforces_portable_names_and_count_limits(self):
        def compiled(xml: str) -> bytes:
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr("world.xml", xml)
            return output.getvalue()

        invalid = {
            "name syntax": ("<mujoco><worldbody><body name='bad name'/></worldbody></mujoco>", "unique stable name"),
            "portable name collision": ("<mujoco><worldbody><body name='a-b'/><body name='a_b'/></worldbody></mujoco>", "remain unique"),
            "body limit": ("<mujoco><worldbody>" + "".join(f"<body name='b{i}'/>" for i in range(4097)) + "</worldbody></mujoco>", "exceeds the supported"),
            "joint limit": ("<mujoco><worldbody><body name='root'>" + "<joint/>" * 8193 + "</body></worldbody></mujoco>", "exceeds the supported"),
            "geom limit": ("<mujoco><worldbody>" + "<geom/>" * 16385 + "</worldbody></mujoco>", "exceeds the supported"),
        }
        for label, (xml, message) in invalid.items():
            with self.subTest(label=label), self.assertRaisesRegex(BundleValidationError, message):
                _validate_environment_mjcf(compiled(xml))

    def test_heightfield_helpers_fail_with_structured_errors_for_malformed_metadata(self):
        from miniverse_sdk.terrain import TerrainValidationError

        with self.assertRaisesRegex(TerrainValidationError, "width must be an integer"):
            build_heightfield_glb(
                terrain_id="steps", width=2.5, height=2, heights=[0, 0, 0, 0], xy_resolution=[1, 1],
            )
        data = build_heightfield_glb(
            terrain_id="steps", width=2, height=2, heights=[0, 0, 0, 0], xy_resolution=[1, 1],
        )
        malformed = data.replace(b'"cellSize":[1.0,1.0]', b'"cellSize":100000000')
        self.assertNotEqual(malformed, data)
        with self.assertRaisesRegex(TerrainValidationError, "XY resolution must contain"):
            inspect_heightfield_glb(malformed)

    def test_large_heightfields_report_portable_size_guidance(self):
        self.assertEqual(heightfield_size_warnings(512, 512), ())
        warnings = heightfield_size_warnings(513, 512)
        self.assertEqual([warning["code"] for warning in warnings], ["large_heightfield"])
        self.assertIn("not a routinely qualified Isaac fleet size", warnings[0]["hint"])

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

    def test_validate_canonical_mini_extension_and_agent_help(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.mini")
            inspected = inspect_bundle(path)
            self.assertEqual(inspected.bundle_id, "fixture")
            self.assertEqual(len(inspected.assets), 3)
            self.assertEqual(inspected.model_precisions, {"policy": "fp16"})
        help_text = agent_help(None, True)
        self.assertIn("MINIVERSE_API_TOKEN", help_text)
        self.assertIn("object-create event starts server-side", help_text)
        self.assertIn(".mini", parser().format_help())
        self.assertIn(".mini", help_text)
        self.assertNotIn(".dhsim", parser().format_help())
        self.assertNotIn(".dhsim", help_text)

    def test_inspect_silently_accepts_legacy_dhsim_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "legacy.dhsim")
            self.assertEqual(inspect_bundle(path).bundle_id, "fixture")

    def test_bundle_rejects_removed_policy_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "legacy.mini", legacy_policy_bindings=True)
            with self.assertRaisesRegex(BundleValidationError, "fields were removed: policyBindings"):
                inspect_bundle(path)

    def test_bundle_rejects_removed_actuator_dynamics_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "legacy.mini", legacy_dynamics_overrides=True)
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
    def test_bundle_validate_reports_findings_and_strict_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.mini", model=valid_model(topk_k=3841))
            inspected = inspect_bundle(path)
            self.assertEqual([finding.code for finding in inspected.model_findings["policy"]], ["tensorrt_topk_k_limit"])
            self.assertEqual(main(["bundle", "validate", str(path), "--json"]), 0)
            self.assertEqual(main(["bundle", "validate", str(path), "--strict"]), 2)
            clean = fixture(Path(directory) / "clean.mini", model=valid_model())
            self.assertEqual(main(["bundle", "validate", str(clean), "--strict"]), 0)

    def test_model_validate_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            path.write_bytes(valid_model(topk_k=3841))
            self.assertEqual(main(["model", "validate", str(path), "--json"]), 0)
            self.assertEqual(main(["model", "validate", str(path), "--strict"]), 2)
            path.write_bytes(valid_model())
            self.assertEqual(main(["model", "validate", str(path), "--strict"]), 0)

    def test_model_validate_fails_simulator_incompatibility_without_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incompatible.onnx"
            path.write_bytes(valid_model(incompatible=True))
            self.assertEqual(main(["model", "validate", str(path), "--json"]), 2)

    def test_bundle_validate_fails_model_incompatibility_without_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(
                Path(directory) / "incompatible.mini",
                model=valid_model(incompatible=True),
                primary_simulator="isaac-sim-cpu-physx",
            )
            self.assertEqual(main(["bundle", "validate", str(path), "--json"]), 2)

    def test_bundle_upload_refuses_local_validation_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(
                Path(directory) / "incompatible.mini",
                model=valid_model(incompatible=True),
                primary_simulator="isaac-sim-cpu-physx",
            )
            with patch("miniverse_sdk.cli.Client") as client:
                self.assertEqual(main(["bundle", "upload", str(path), "--json"]), 2)
            client.assert_not_called()

    def test_bundle_manifest_uses_the_normative_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "missing-name.mini")
            with zipfile.ZipFile(path, "r") as archive:
                values = {name: archive.read(name) for name in archive.namelist()}
            manifest = json.loads(values["bundle.json"])
            del manifest["name"]
            values["bundle.json"] = json.dumps(manifest).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in values.items():
                    archive.writestr(name, value)
            self.assertEqual(main(["bundle", "validate", str(path), "--json"]), 2)

    def test_bundled_schemas_match_repository_authorities(self):
        root = Path(__file__).resolve().parents[1]
        bundled = root / "src" / "miniverse_sdk" / "schemas"
        authoritative = root / "schemas"
        for name in ("simulation-bundle-v1.schema.json", "onnx-simulation-contract-0.3.schema.json"):
            self.assertEqual(json.loads((bundled / name).read_text()), json.loads((authoritative / name).read_text()))

    def test_missing_and_invalid_precision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.mini")
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
            path = fixture(Path(directory) / "fixture.mini")
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("extra.txt", "no")
            with self.assertRaises(BundleValidationError) as caught:
                inspect_bundle(path)
            self.assertEqual(caught.exception.code, "undeclared_member")

    def test_end_to_end_archive_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fixture(Path(directory) / "fixture.mini")
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
