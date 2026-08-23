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
from .config import OAuthCredential, credential_lock, load_oauth_credential, save_oauth_credential


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, code: str = "api_error"):
        super().__init__(message)
        self.status = status
        self.code = code


class Client:
    def __init__(self, origin: str, credential: OAuthCredential | str | None):
        self.origin = origin.rstrip("/")
        self.credential = (
            OAuthCredential(access_token=credential, origin=self.origin)
            if isinstance(credential, str)
            else credential
        )

    def request(self, path: str, value: dict[str, Any] | None = None, method: str | None = None, authenticated: bool = True) -> dict[str, Any]:
        data = json.dumps(value, separators=(",", ":")).encode() if value is not None else None
        headers = {"accept": "application/json", "user-agent": f"miniverse-sdk/{__version__}"}
        if data is not None:
            headers["content-type"] = "application/json"
        return self._request(path, data, headers, method or ("POST" if data is not None else "GET"), authenticated)

    def request_form(self, path: str, value: dict[str, str], authenticated: bool = False) -> dict[str, Any]:
        data = urllib.parse.urlencode(value).encode()
        headers = {
            "accept": "application/json",
            "content-type": "application/x-www-form-urlencoded",
            "user-agent": f"miniverse-sdk/{__version__}",
        }
        return self._request(path, data, headers, "POST", authenticated)

    def _request(self, path: str, data: bytes | None, headers: dict[str, str], method: str, authenticated: bool) -> dict[str, Any]:
        if authenticated:
            if not self.credential:
                raise ApiError(401, "authentication required; set MINIVERSE_API_TOKEN or run `miniverse auth login`", "authentication_required")
            if self.credential.renewable and self.credential.expires_at is not None and self.credential.expires_at <= time.time() + 60:
                self._refresh()
            headers = {**headers, "authorization": f"Bearer {self.credential.access_token}"}
        try:
            return self._request_once(path, data, headers, method)
        except ApiError as error:
            if error.status != 401 or not authenticated or not self.credential or not self.credential.renewable:
                raise
            self._refresh(force=True)
            retry_headers = {**headers, "authorization": f"Bearer {self.credential.access_token}"}
            return self._request_once(path, data, retry_headers, method)

    def _request_once(self, path: str, data: bytes | None, headers: dict[str, str], method: str) -> dict[str, Any]:
        request = urllib.request.Request(self.origin + path, data=data, method=method, headers=headers)
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

    def _refresh(self, force: bool = False) -> None:
        if not self.credential or not self.credential.refresh_token:
            raise ApiError(401, "saved login cannot be renewed; run `miniverse auth login` once", "reauthentication_required")
        with credential_lock():
            current = load_oauth_credential(self.origin)
            if current and current.refresh_token:
                if current.access_token != self.credential.access_token and (current.expires_at is None or current.expires_at > time.time() + 60):
                    self.credential = current
                    return
                self.credential = current
            if not force and self.credential.expires_at is not None and self.credential.expires_at > time.time() + 60:
                return
            refresh_token = self.credential.refresh_token
            assert refresh_token is not None
            try:
                value = self.request_form("/api/auth/oauth2/token", {
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": self.credential.client_id,
                })
            except ApiError as error:
                if error.code in {"invalid_grant", "invalid_client"}:
                    raise ApiError(401, "saved login expired or was revoked; run `miniverse auth login`", "reauthentication_required") from error
                raise
            access_token = value.get("access_token")
            if not isinstance(access_token, str) or not access_token:
                raise ApiError(502, "authorization server returned an incomplete refresh response", "oauth_contract_error")
            expires_in = value.get("expires_in")
            rotated = value.get("refresh_token")
            self.credential = OAuthCredential(
                access_token=access_token,
                refresh_token=rotated if isinstance(rotated, str) and rotated else refresh_token,
                expires_at=time.time() + float(expires_in) if isinstance(expires_in, (int, float)) else None,
                scope=value.get("scope") if isinstance(value.get("scope"), str) else self.credential.scope,
                client_id=self.credential.client_id,
                origin=self.origin,
            )
            save_oauth_credential(self.credential)

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
