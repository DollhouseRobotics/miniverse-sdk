"""Static validation for closed-loop MJCF weld declarations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from xml.etree import ElementTree

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class MjcfConstraintError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _invalid(message: str) -> None:
    raise MjcfConstraintError("invalid_mjcf_weld", message)


def validate_weld_declarations(documents: Mapping[str, ElementTree.Element]) -> None:
    """Validate the endpoint shape of authored welds without choosing a backend."""
    for document in documents.values():
        for index, weld in enumerate(document.iter("weld")):
            name = weld.get("name")
            label = name or f"weld:{index}"
            if name is not None and not _ID.fullmatch(name):
                _invalid(f"MJCF weld {name!r} requires a stable ID of at most 128 characters")

            body1, body2 = weld.get("body1"), weld.get("body2")
            site1, site2 = weld.get("site1"), weld.get("site2")
            uses_bodies = body1 is not None or body2 is not None
            uses_sites = site1 is not None or site2 is not None

            if uses_bodies and uses_sites:
                _invalid(f"MJCF weld {label} must use body endpoints or site endpoints, not both")
            if uses_sites:
                if not site1 or not site2:
                    _invalid(f"MJCF weld {label} requires both site1 and site2")
                if site1 == site2:
                    _invalid(f"MJCF weld {label} cannot attach a site to itself")
                continue
            if not body1:
                _invalid(f"MJCF weld {label} requires body1 or both site1 and site2")
            if body2 == body1:
                _invalid(f"MJCF weld {label} cannot attach a body to itself")
