"""Small versioned HTTP client used by the Miniverse command."""

from __future__ import annotations

import json
import http.client
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import __version__


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, code: str = "api_error"):
        super().__init__(message)
        self.status = status
        self.code = code


class Client:
    def __init__(self, origin: str, token: str | None):
        self.origin = origin.rstrip("/")
        self.token = token

    def request(self, path: str, value: dict[str, Any] | None = None, method: str | None = None, authenticated: bool = True) -> dict[str, Any]:
        data = json.dumps(value, separators=(",", ":")).encode() if value is not None else None
        headers = {"accept": "application/json", "user-agent": f"miniverse-sdk/{__version__}"}
        if data is not None:
            headers["content-type"] = "application/json"
        if authenticated:
            if not self.token:
                raise ApiError(401, "authentication required; set MINIVERSE_API_TOKEN or run `miniverse auth login`", "authentication_required")
            headers["authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.origin + path, data=data, method=method or ("POST" if data is not None else "GET"), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read(64 * 1024)
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            raw_error = payload.get("error")
            code = payload.get("code") or (raw_error if isinstance(raw_error, str) else None) or "api_error"
            message = payload.get("message") or payload.get("error_description") or (raw_error if isinstance(raw_error, str) else None) or error.reason
            raise ApiError(error.code, str(message), str(code).lower()) from error

    def upload(self, url: str, path: Path) -> None:
        parsed = urllib.parse.urlsplit(url)
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        last_error: Exception | None = None
        for attempt in range(3):
            connection = connection_type(parsed.hostname, parsed.port, timeout=1800)
            try:
                target = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
                connection.putrequest("PUT", target)
                connection.putheader("content-type", "application/vnd.dhr.simulation-bundle+zip")
                connection.putheader("content-length", str(path.stat().st_size))
                connection.putheader("user-agent", f"miniverse-sdk/{__version__}")
                connection.endheaders()
                with path.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        connection.send(chunk)
                response = connection.getresponse()
                response.read()
                if response.status in {200, 201, 204}:
                    return
                last_error = ApiError(response.status, f"archive upload failed: {response.reason}", "upload_failed")
            except (OSError, http.client.HTTPException) as error:
                last_error = error
            finally:
                connection.close()
            if attempt < 2:
                time.sleep(2 ** attempt)
        if isinstance(last_error, ApiError):
            raise last_error
        raise ApiError(0, f"archive upload failed: {last_error}", "upload_failed")

    def wait_for_import(self, status_url: str, timeout: int = 3600) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.request(status_url)
            if value.get("state") in {"ready", "published", "failed", "rejected", "cancelled"}:
                return value
            time.sleep(max(1, min(10, int(value.get("pollAfterSeconds", 2)))))
        raise ApiError(408, "bundle import did not finish before the timeout", "import_timeout")
