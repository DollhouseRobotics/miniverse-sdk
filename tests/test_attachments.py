from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import onnx
from test_cli import fixture, valid_model

from miniverse_sdk.bundles import BundleValidationError, inspect_bundle
from miniverse_sdk.onnx_metadata import ONNX_HASH_KEY, ONNX_METADATA_KEY

SIMULATORS = ("mujoco", "browser-mujoco", "isaac-sim-cpu-physx", "isaac-sim-gpu-physx")


def embodiment() -> ET.Element:
    return ET.fromstring(
        """<mujoco model="closed_loop_payload">
          <worldbody>
            <body name="base"><freejoint/>
              <body name="left"><site name="left_mount"/></body>
              <body name="right"><body name="payload"><site name="payload_mount"/></body></body>
            </body>
          </worldbody>
          <equality><weld name="secondary_mount" site1="left_mount" site2="payload_mount"/></equality>
        </mujoco>"""
    )


def bundle(path: Path, root: ET.Element, *, simulator: str = "mujoco", extra=None) -> Path:
    model = onnx.load_model_from_string(valid_model())
    metadata = {value.key: value.value for value in model.metadata_props}
    contract = json.loads(metadata[ONNX_METADATA_KEY])
    for backend in ("isaac-sim-cpu-physx", "isaac-sim-gpu-physx"):
        contract["backends"].append(
            {"id": backend, "versionRange": ">=5.1,<5.2", "providers": ["CPUExecutionProvider"]}
        )
    del contract["contractHash"]
    contract["contractHash"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata[ONNX_METADATA_KEY] = json.dumps(contract)
    metadata[ONNX_HASH_KEY] = contract["contractHash"]
    onnx.helper.set_model_props(model, metadata)
    fixture(path, model=model.SerializeToString(deterministic=True), primary_simulator=simulator)
    with zipfile.ZipFile(path) as archive:
        values = {name: archive.read(name) for name in archive.namelist()}
    values["embodiment/robot.xml"] = ET.tostring(root)
    values.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in values.items():
            archive.writestr(name, data)
    return path


class AttachmentValidationTests(unittest.TestCase):
    def test_body_site_and_world_welds_are_valid_for_every_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.mini"
            declarations = (
                {"site1": "left_mount", "site2": "payload_mount"},
                {"body1": "left", "body2": "payload"},
                {"body1": "payload"},
            )
            for simulator in SIMULATORS:
                for declaration in declarations:
                    with self.subTest(simulator=simulator, declaration=declaration):
                        root = embodiment()
                        root.find("equality/weld").attrib = {"name": "attachment", **declaration}
                        inspect_bundle(bundle(path, root, simulator=simulator))

    def test_malformed_endpoint_declarations_are_rejected_for_every_simulator(self):
        cases = (
            ({}, "requires body1 or both site1 and site2"),
            ({"site1": "left_mount"}, "requires both site1 and site2"),
            ({"body2": "payload"}, "requires body1"),
            ({"body1": "left", "site1": "left_mount", "site2": "payload_mount"}, "not both"),
            ({"body1": "left", "body2": "left"}, "body to itself"),
            ({"site1": "left_mount", "site2": "left_mount"}, "site to itself"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.mini"
            for simulator in SIMULATORS:
                for declaration, message in cases:
                    root = embodiment()
                    root.find("equality/weld").attrib = {"name": "attachment", **declaration}
                    with self.subTest(simulator=simulator, declaration=declaration):
                        with self.assertRaisesRegex(BundleValidationError, message) as raised:
                            inspect_bundle(bundle(path, root, simulator=simulator))
                        self.assertEqual(raised.exception.code, "invalid_mjcf_weld")

    def test_welds_in_nonstandard_extension_includes_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = embodiment()
            root.remove(root.find("equality"))
            ET.SubElement(root, "include", file="attachment.inc")
            included = b'<mujocoinclude><equality><weld site1="left_mount"/></equality></mujocoinclude>'
            with self.assertRaisesRegex(BundleValidationError, "both site1 and site2"):
                inspect_bundle(
                    bundle(
                        Path(directory) / "robot.mini",
                        root,
                        extra={"embodiment/attachment.inc": included},
                    )
                )

    def test_backend_specific_equality_behavior_is_left_to_the_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = embodiment()
            weld = root.find("equality/weld")
            weld.set("solref", ".1 1")
            ET.SubElement(root.find("equality"), "connect", site1="left_mount", site2="payload_mount")
            for simulator in SIMULATORS:
                with self.subTest(simulator=simulator):
                    inspect_bundle(bundle(Path(directory) / "robot.mini", root, simulator=simulator))

    def test_invalid_named_weld_id_is_descriptive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = embodiment()
            root.find("equality/weld").set("name", "invalid name")
            with self.assertRaisesRegex(BundleValidationError, "stable ID"):
                inspect_bundle(bundle(Path(directory) / "robot.mini", root))


if __name__ == "__main__":
    unittest.main()
