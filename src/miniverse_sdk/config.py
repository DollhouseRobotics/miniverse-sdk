"""Credential and endpoint handling without plaintext configuration secrets."""

from __future__ import annotations

import os

DEFAULT_ORIGIN = "https://miniverse.bot"
SERVICE = "miniverse-sdk"
ACCOUNT = "oauth-access-token"


def origin(value: str | None = None) -> str:
    return (value or os.environ.get("MINIVERSE_ORIGIN") or DEFAULT_ORIGIN).rstrip("/")


def save_oauth_token(token: str) -> str:
    try:
        import keyring  # type: ignore
        keyring.set_password(SERVICE, ACCOUNT, token)
        return "keyring"
    except Exception as error:
        raise RuntimeError("OAuth login requires an available system credential store") from error


def load_oauth_token() -> str | None:
    try:
        import keyring  # type: ignore
        value = keyring.get_password(SERVICE, ACCOUNT)
        if value:
            return value
    except Exception:
        return None
    return None


def delete_oauth_token() -> None:
    try:
        import keyring  # type: ignore
        keyring.delete_password(SERVICE, ACCOUNT)
    except Exception:
        pass


def credential() -> tuple[str | None, str]:
    token = os.environ.get("MINIVERSE_API_TOKEN")
    if token:
        return token, "MINIVERSE_API_TOKEN"
    token = load_oauth_token()
    return (token, "oauth") if token else (None, "none")
