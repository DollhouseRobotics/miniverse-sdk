# Authentication

For CI or another non-interactive environment, provide the token only through
the environment:

```bash
MINIVERSE_API_TOKEN=... miniverse bundle upload PATH.dhsim --json
```

Do not pass a token as an argument, write it into repository files, or include
it in logs. `MINIVERSE_API_TOKEN` takes precedence over an OAuth credential.

For a person at a terminal, run `miniverse auth login` once. Complete the
displayed device authorization in the browser. The CLI stores the renewable
OAuth grant in a user-only file and refreshes it automatically, including after
an API response reports that the access token expired.

On Linux the default is `$XDG_STATE_HOME/miniverse/auth.json`, or
`~/.local/state/miniverse/auth.json` when `XDG_STATE_HOME` is unset. Set
`MINIVERSE_HOME` to put `auth.json` in a different persistent directory. An
ephemeral agent or container must mount that directory across runs.

Inspect only non-secret metadata with `miniverse auth status`; remove and revoke
the OAuth credential with `miniverse auth logout`. To opt back into an available
desktop credential service, set `MINIVERSE_AUTH_STORE=keyring`.
