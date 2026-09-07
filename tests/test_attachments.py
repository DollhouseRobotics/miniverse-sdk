from __future__ import annotations

import hashlib
from importlib import resources, util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from xml.etree import ElementTree as ET
import zipfile

import onnx

from miniverse_sdk.bundles import BundleValidationError, inspect_bundle
from miniverse_sdk.cli import agent_help, main
from miniverse_sdk.mjcf_constraints import _load_mujoco, MjcfConstraintError
from miniverse_sdk.onnx_metadata import ONNX_HASH_KEY, ONNX_METADATA_KEY
from test_cli import fixture, valid_model

HAS_MUJOCO = util.find_spec("mujoco") is not None
ISAAC = ("isaac-sim-cpu-physx", "isaac-sim-gpu-physx")


def example() -> ET.Element:
    # Exercise the actual shipped authoring example so it cannot silently rot.
    guide = resources.files("miniverse_sdk.agent_help").joinpath("attachments.md").read_text()
    return ET.fromstring(guide.split("```xml\n", 1)[1].split("```", 1)[0])


def bundle(path, root, *, primary=ISAAC[0], compatible=(), extra=None, entrypoint="robot.xml"):
    model = onnx.load_model_from_string(valid_model())
    metadata = {value.key: value.value for value in model.metadata_props}
    contract = json.loads(metadata[ONNX_METADATA_KEY])
    for backend in ISAAC:
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
    fixture(path, model=model.SerializeToString(deterministic=True), primary_simulator=primary)
    with zipfile.ZipFile(path) as archive:
        values = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(values["bundle.json"])
    if compatible:
        manifest["compatibleSimulators"] = list(compatible)
    manifest["embodiment"]["path"] = f"embodiment/{entrypoint}"
    del values["embodiment/robot.xml"]
    values["bundle.json"] = json.dumps(manifest).encode()
    values[f"embodiment/{entrypoint}"] = ET.tostring(root)
    values.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in values.items():
            archive.writestr(name, data)
    return path


class AttachmentDependencyTests(unittest.TestCase):
    def test_no_extra_required_for_plain_isaac_includes_or_mujoco_equalities(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"mujoco": None}):
            path = Path(directory) / "robot.mini"
            root = example()
            root.remove(root.find("equality"))
            for backend in ISAAC:
                inspect_bundle(bundle(path, root, primary=backend))
                included = ET.Element("mujoco")
                ET.SubElement(included, "include", file="parts/body.inc")
                inspect_bundle(
                    bundle(path, included, primary=backend, extra={"embodiment/parts/body.inc": ET.tostring(root)})
                )
            for backend in ("mujoco", "browser-mujoco"):
                root = example()
                root.find("equality/weld").set("solref", "0.1 1")
                inspect_bundle(bundle(path, root, primary=backend))

    def test_missing_extra_is_actionable_for_validate_and_stops_upload_before_auth(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"mujoco": None}):
            path = bundle(Path(directory) / "robot.mini", example(), primary="mujoco", compatible=[ISAAC[1]])
            for command in ("validate", "upload"):
                output = io.StringIO()
                with (
                    redirect_stdout(output),
                    patch("miniverse_sdk.cli.credential") as auth,
                    patch("miniverse_sdk.cli.Client") as client,
                ):
                    result = main(["bundle", command, str(path), "--json"])
                self.assertEqual(result, 2)
                issue = json.loads(output.getvalue())["errors"][0]
                self.assertEqual(issue["code"], "mjcf_validation_dependency")
                self.assertIn('uv tool install --force "miniverse-sdk[mjcf]"', issue["message"])
                auth.assert_not_called()
                client.assert_not_called()

    def test_wrong_compiler_version_does_not_silently_change_semantics(self):
        with patch.dict("sys.modules", {"mujoco": type("Compiler", (), {"__version__": "0.0"})()}):
            with self.assertRaises(MjcfConstraintError) as raised:
                _load_mujoco()
        self.assertEqual(raised.exception.code, "mjcf_validation_dependency")
        self.assertIn("3.3.6", str(raised.exception))

    def test_non_xml_includes_are_checked_before_native_compilation(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("miniverse_sdk.mjcf_constraints._load_mujoco") as compiler,
        ):
            path = Path(directory) / "robot.mini"
            root = ET.fromstring('<mujoco><include file="part.inc"/></mujoco>')
            cases = (
                (
                    b'<mujocoinclude><include file="../../outside.xml"/></mujocoinclude>',
                    "escapes the embodiment directory",
                ),
                (b'<mujocoinclude><include file="part.inc"/></mujocoinclude>', "cyclic"),
                (b'<mujocoinclude><include file="missing.inc"/></mujocoinclude>', "dependency is missing"),
                (b'<mujocoinclude><compiler assetdir="/outside"/></mujocoinclude>', "compiler settings"),
                (b"not XML", "invalid MJCF XML"),
            )
            for data, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(BundleValidationError, message):
                    inspect_bundle(bundle(path, root, extra={"embodiment/part.inc": data}))
            compiler.assert_not_called()

    def test_agent_help_topic_is_discoverable(self):
        self.assertIn("agent-help attachments", agent_help(None, False))
        self.assertIn("site1=", agent_help("attachments", False))
        self.assertIn("mjcf_validation_dependency", agent_help(None, True))


@unittest.skipUnless(HAS_MUJOCO, "install miniverse-sdk[mjcf] for native compiler regression tests")
class AttachmentCompilerTests(unittest.TestCase):
    def test_site_body_world_and_inactive_welds_on_both_isaac_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            for backend in ISAAC:
                for attrs in (
                    {"site1": "left_mount", "site2": "payload_mount"},
                    {
                        "body1": "left_arm",
                        "body2": "payload",
                        "anchor": ".1 .2 .3",
                        "relpose": ".4 .5 .6 .7071067811865476 0 0 .7071067811865476",
                    },
                    {"body1": "payload"},
                    {"body1": "left_arm", "body2": "payload", "active": "false"},
                ):
                    with self.subTest(backend=backend, attrs=attrs):
                        root = example()
                        root.find("equality/weld").attrib = attrs
                        path = bundle(Path(directory) / "robot.mini", root, primary=backend)
                        before = path.read_bytes()
                        output = io.StringIO()
                        with redirect_stdout(output):
                            self.assertEqual(main(["bundle", "validate", str(path), "--json"]), 0)
                        self.assertTrue(json.loads(output.getvalue())["ok"])
                        self.assertEqual(path.read_bytes(), before)

    def test_defaults_in_nested_non_xml_includes_and_compatible_simulator_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = example()
            root.find("equality/weld").set("class", "attachment")
            ET.SubElement(root, "include", file="parts/outer.inc")
            extra = {
                "embodiment/model/parts/outer.inc": b'<mujocoinclude><include file="inner.inc"/></mujocoinclude>',
                "embodiment/model/parts/inner.inc": b'<mujocoinclude><default><default class="attachment"><equality solref=".1 1"/></default></default></mujocoinclude>',
            }
            path = bundle(
                Path(directory) / "robot.mini",
                root,
                primary="browser-mujoco",
                compatible=[ISAAC[1]],
                extra=extra,
                entrypoint="model/robot.xml",
            )
            with self.assertRaisesRegex(BundleValidationError, "default solref/solimp") as raised:
                inspect_bundle(path)
            self.assertEqual(raised.exception.code, "unsupported_isaac_equality")
            extra["embodiment/model/parts/inner.inc"] = extra["embodiment/model/parts/inner.inc"].replace(
                b".1 1", b".02 1"
            )
            inspect_bundle(
                bundle(
                    path,
                    root,
                    primary="browser-mujoco",
                    compatible=[ISAAC[1]],
                    extra=extra,
                    entrypoint="model/robot.xml",
                )
            )

    def test_each_custom_solver_parameter_is_rejected_even_when_inactive(self):
        with tempfile.TemporaryDirectory() as directory:
            for attribute, value in (("solref", ".1 1"), ("solimp", ".8 .95 .001 .5 2"), ("torquescale", "0")):
                root = example()
                root.find("equality/weld").set(attribute, value)
                root.find("equality/weld").set("active", "false")
                path = bundle(Path(directory) / "robot.mini", root)
                with (
                    self.subTest(attribute=attribute),
                    self.assertRaisesRegex(BundleValidationError, "default solref/solimp"),
                ):
                    inspect_bundle(path)

    def test_other_equalities_and_independent_payloads_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.mini"
            for kind, attributes in (
                ("connect", {"site1": "left_mount", "site2": "payload_mount"}),
                ("joint", {"joint1": "left_hinge", "joint2": "right_hinge"}),
                ("tendon", {"tendon1": "left_tendon", "tendon2": "right_tendon"}),
            ):
                root = example()
                weld = root.find("equality/weld")
                weld.tag, weld.attrib = kind, attributes
                if kind == "tendon":
                    tendons = ET.SubElement(root, "tendon")
                    for side in ("left", "right"):
                        tendon = ET.SubElement(tendons, "fixed", name=f"{side}_tendon")
                        ET.SubElement(tendon, "joint", joint=f"{side}_hinge", coef="1")
                with self.subTest(kind=kind), self.assertRaisesRegex(BundleValidationError, "not a supported weld"):
                    inspect_bundle(bundle(path, root))
            root = example()
            payload = root.find(".//body[@name='payload']")
            root.find(".//body[@name='right_arm']").remove(payload)
            ET.SubElement(payload, "freejoint", name="payload_free")
            root.find("worldbody").append(payload)
            with self.assertRaisesRegex(BundleValidationError, "single embodiment body tree"):
                inspect_bundle(bundle(path, root))

    def test_missing_endpoint_and_invalid_ids_are_descriptive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = example()
            root.find("equality/weld").set("site2", "missing_mount")
            path = bundle(Path(directory) / "robot.mini", root)
            with self.assertRaisesRegex(BundleValidationError, "could not compile") as raised:
                inspect_bundle(path)
            self.assertEqual(raised.exception.code, "invalid_embodiment")
            root = example()
            root.find("equality/weld").set("name", "invalid name")
            with self.assertRaisesRegex(BundleValidationError, "stable ID"):
                inspect_bundle(bundle(path, root))

    def test_global_disable_and_world_owned_site_preserve_accepted_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = example()
            ET.SubElement(root.find("option"), "flag", equality="disable")
            ET.SubElement(root.find("worldbody"), "site", name="world_mount", pos=".5 0 2")
            root.find("equality/weld").set("site1", "world_mount")
            path = bundle(Path(directory) / "robot.mini", root)
            inspected = inspect_bundle(path)
            mujoco_only = inspect_bundle(bundle(path, root, primary="mujoco"))
            self.assertEqual(inspected.assets[1], mujoco_only.assets[1])

    def test_equality_count_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = example()
            equality = root.find("equality")
            for _ in range(1024):
                ET.SubElement(equality, "weld", body1="left_arm", body2="payload")
            path = bundle(Path(directory) / "robot.mini", root)
            with self.assertRaisesRegex(BundleValidationError, "at most 1024"):
                inspect_bundle(path)


if __name__ == "__main__":
    unittest.main()
