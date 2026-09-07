"""Local admission checks for the platform's closed-loop MJCF weld support.

Use the same MuJoCo compiler as the platform, not a second implementation of
MJCF defaults, sites or reference poses. This module does not simulate, modify
source assets, generate scene hashes, or import the platform runtime.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping
from xml.etree import ElementTree

MAX_WELDS = 1024
MUJOCO_VERSION = "3.3.6"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_INSTALL = 'uv tool install --force "miniverse-sdk[mjcf]"'


class MjcfConstraintError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _load_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as error:
        raise MjcfConstraintError(
            "mjcf_validation_dependency",
            f"Isaac equality validation requires the optional MJCF compiler; run {_INSTALL} and retry. "
            "No Isaac installation or GPU is needed.",
        ) from error
    if mujoco.__version__ != MUJOCO_VERSION:
        raise MjcfConstraintError(
            "mjcf_validation_dependency",
            f"Isaac equality validation requires MuJoCo {MUJOCO_VERSION} to match the platform; run {_INSTALL}.",
        )
    return mujoco


def _unsupported(message: str) -> None:
    raise MjcfConstraintError(
        "unsupported_isaac_equality",
        message + "; use mujoco or browser-mujoco without declaring Isaac compatibility. "
        "See miniverse agent-help attachments.",
    )


def _invalid_weld(message: str) -> None:
    raise MjcfConstraintError("invalid_mjcf_weld", message + ". See miniverse agent-help attachments.")


def _validate_compiled(model: Any, mujoco: Any) -> None:
    if model.neq > MAX_WELDS:
        _invalid_weld(f"Miniverse supports at most {MAX_WELDS} weld constraints")
    occupied = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) for i in range(model.neq)}
    for index in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index)
        label = name or f"weld:{index}"
        if name is None:
            while label in occupied:
                label += ":auto"
        occupied.add(label)
        if model.eq_type[index] != mujoco.mjtEq.mjEQ_WELD:
            _unsupported(f"Isaac equality {label} is not a supported weld")
        if not _ID.fullmatch(label):
            _invalid_weld(f"Weld {label!r} requires a stable ID of at most 128 characters")
        first, second = int(model.eq_obj1id[index]), int(model.eq_obj2id[index])
        frame_values = list(model.eq_data[index])
        if model.eq_objtype[index] == mujoco.mjtObj.mjOBJ_SITE:
            for site in (first, second):
                frame_values.extend(model.site_pos[site])
                frame_values.extend(model.site_quat[site])
            first, second = int(model.site_bodyid[first]), int(model.site_bodyid[second])
        elif model.eq_objtype[index] != mujoco.mjtObj.mjOBJ_BODY:
            _unsupported(f"Isaac weld {label} has an unsupported endpoint type")
        if first == second:
            _invalid_weld(f"Weld {label} cannot attach a body to itself")
        if not all(math.isfinite(float(value)) for value in frame_values):
            _invalid_weld(f"Weld {label} requires finite attachment frames")
        for body in (first, second):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body != 0 and (body_name is None or not _ID.fullmatch(body_name)):
                _invalid_weld(f"Weld {label} endpoints must be named embodiment bodies with stable IDs")
        actual = [*model.eq_solref[index], *model.eq_solimp[index], model.eq_data[index, 10]]
        expected = [0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0, 1.0]
        if not all(math.isclose(float(a), b, rel_tol=0, abs_tol=1e-12) for a, b in zip(actual, expected)):
            _unsupported(f"Isaac weld {label} supports only default solref/solimp and torquescale=1")
    if model.neq and sum(int(model.body_parentid[i]) == 0 for i in range(1, model.nbody)) != 1:
        _unsupported(
            "Isaac weld attachments require a single embodiment body tree. Make a permanently attached "
            "payload a jointless child of its primary attachment body and weld the remaining sites"
        )


def validate_isaac_constraints(
    entrypoint: str,
    files: Mapping[str, bytes],
    documents: Mapping[str, ElementTree.Element],
) -> None:
    """Check an already path-validated MJCF closure for declared Isaac use.

    Ordinary scenes, including includes without equalities, do not require the
    optional compiler. MJCF generators can create equalities, so compile those
    too. Call only after the bundle inspector has resolved the exact closure.
    """
    if not any(
        element.tag in {"equality", "composite", "flexcomp", "attach", "replicate"}
        for document in documents.values()
        for element in document.iter()
    ):
        return
    mujoco = _load_mujoco()
    with tempfile.TemporaryDirectory(prefix="miniverse-sdk-mjcf-") as directory:
        root = Path(directory)
        for name, data in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        try:
            model = mujoco.MjModel.from_xml_path(str(root / entrypoint))
        except ValueError as error:
            raise MjcfConstraintError(
                "invalid_embodiment", f"MJCF equality validation could not compile the embodiment: {error}"
            ) from error
    _validate_compiled(model, mujoco)
