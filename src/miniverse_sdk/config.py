"""Endpoint handling and durable OAuth credentials for the Miniverse CLI."""

from __future__ import annotations

import json
import os
import platform
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_ORIGIN = "https://miniverse.bot"
SERVICE = "miniverse-sdk"
LEGACY_ACCOUNT = "oauth-access-token"
AUTH_VERSION = 1


@dataclass(frozen=True)
class OAuthCredential:
    access_token: str
    origin: str
    refresh_token: str | None = None
    expires_at: float | None = None
    scope: str | None = None
    client_id: str = "miniverse-cli"

    @property
    def renewable(self) -> bool:
        return bool(self.refresh_token)


def origin(value: str | None = None) -> str:
    return (value or os.environ.get("MINIVERSE_ORIGIN") or DEFAULT_ORIGIN).rstrip("/")


def auth_file() -> Path:
    override = os.environ.get("MINIVERSE_AUTH_FILE")
    if override:
        return Path(override).expanduser()
    home = os.environ.get("MINIVERSE_HOME")
    if home:
        return Path(home).expanduser() / "auth.json"
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Miniverse" / "auth.json"
    if system == "Windows":
        state = os.environ.get("LOCALAPPDATA")
        return (Path(state) if state else Path.home() / "AppData" / "Local") / "Miniverse" / "auth.json"
    state = os.environ.get("XDG_STATE_HOME")
    return (Path(state).expanduser() if state else Path.home() / ".local" / "state") / "miniverse" / "auth.json"


def auth_store() -> str:
    value = os.environ.get("MINIVERSE_AUTH_STORE", "file").strip().lower()
    if value not in {"file", "keyring"}:
        raise RuntimeError("MINIVERSE_AUTH_STORE must be 'file' or 'keyring'")
    return value


def _read_file() -> dict[str, object]:
    path = auth_file()
    try:
        if os.name != "nt" and path.stat().st_mode & 0o077:
            raise RuntimeError(f"Miniverse credential file permissions are too open; run `chmod 600 {path}`")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": AUTH_VERSION, "origins": {}}
    except RuntimeError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not read Miniverse credentials from {path}") from error
    if not isinstance(value, dict) or value.get("version") != AUTH_VERSION or not isinstance(value.get("origins"), dict):
        raise RuntimeError(f"unsupported Miniverse credential file at {path}")
    return value


def _write_file(value: dict[str, object]) -> None:
    path = auth_file()
    parent_existed = path.parent.exists()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not parent_existed:
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
    descriptor, temporary = tempfile.mkstemp(prefix=".auth-", suffix=".tmp", dir=path.parent)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, separators=(",", ":"), sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
        if hasattr(os, "O_DIRECTORY"):
            try:
                parent_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)
            except OSError:
                pass
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _keyring_account(api_origin: str) -> str:
    return f"oauth:{api_origin}"


def _keyring_get(account: str) -> str | None:
    try:
        import keyring  # type: ignore
        return keyring.get_password(SERVICE, account)
    except Exception:
        return None


def _keyring_set(account: str, value: str) -> None:
    try:
        import keyring  # type: ignore
        keyring.set_password(SERVICE, account, value)
    except Exception as error:
        raise RuntimeError("MINIVERSE_AUTH_STORE=keyring requires an available system credential store") from error


def _keyring_delete(account: str) -> None:
    try:
        import keyring  # type: ignore
        keyring.delete_password(SERVICE, account)
    except Exception:
        pass


def _decode_credential(value: object, api_origin: str) -> OAuthCredential | None:
    if not isinstance(value, dict) or not isinstance(value.get("access_token"), str):
        return None
    return OAuthCredential(
        access_token=value["access_token"],
        refresh_token=value.get("refresh_token") if isinstance(value.get("refresh_token"), str) else None,
        expires_at=float(value["expires_at"]) if isinstance(value.get("expires_at"), (int, float)) else None,
        scope=value.get("scope") if isinstance(value.get("scope"), str) else None,
        client_id=value.get("client_id") if isinstance(value.get("client_id"), str) else "miniverse-cli",
        origin=api_origin,
    )


def save_oauth_credential(credential: OAuthCredential) -> str:
    if auth_store() == "keyring":
        _keyring_set(_keyring_account(credential.origin), json.dumps(asdict(credential), separators=(",", ":")))
        return "keyring"
    value = _read_file()
    origins = dict(value["origins"])  # type: ignore[arg-type]
    origins[credential.origin] = asdict(credential)
    _write_file({"version": AUTH_VERSION, "origins": origins})
    return "file"


def load_oauth_credential(api_origin: str | None = None) -> OAuthCredential | None:
    target = origin(api_origin)
    if auth_store() == "keyring":
        serialized = _keyring_get(_keyring_account(target))
        if serialized:
            try:
                return _decode_credential(json.loads(serialized), target)
            except json.JSONDecodeError:
                return None
        return None
    stored_origins = _read_file()["origins"]
    credential = _decode_credential(stored_origins.get(target), target) if isinstance(stored_origins, dict) else None
    if credential:
        return credential
    # One-way migration for the pre-0.2 keyring entry. The old access token is
    # retained as a non-renewable credential until the user signs in once.
    legacy = _keyring_get(LEGACY_ACCOUNT) if target == DEFAULT_ORIGIN else None
    if legacy:
        migrated = OAuthCredential(access_token=legacy, origin=target)
        save_oauth_credential(migrated)
        _keyring_delete(LEGACY_ACCOUNT)
        return migrated
    return None


def delete_oauth_credential(api_origin: str | None = None) -> None:
    target = origin(api_origin)
    if auth_store() == "keyring":
        _keyring_delete(_keyring_account(target))
    else:
        value = _read_file()
        origins = dict(value["origins"])  # type: ignore[arg-type]
        if origins.pop(target, None) is not None:
            _write_file({"version": AUTH_VERSION, "origins": origins})
    _keyring_delete(LEGACY_ACCOUNT)


@contextmanager
def credential_lock() -> Iterator[None]:
    """Serialize refresh-token rotation across local CLI processes."""
    path = auth_file().with_suffix(".lock")
    parent_existed = path.parent.exists()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not parent_existed:
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def credential(api_origin: str | None = None) -> tuple[OAuthCredential | str | None, str]:
    target = origin(api_origin)
    token = os.environ.get("MINIVERSE_API_TOKEN")
    if token:
        return token, "MINIVERSE_API_TOKEN"
    stored = load_oauth_credential(target)
    return (stored, "oauth") if stored else (None, "none")
