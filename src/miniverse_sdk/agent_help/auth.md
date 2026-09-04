# Authentication

Use a personal API token for CI, a remote environment, or an ephemeral machine.
From a trusted workspace where you have completed `miniverse auth login`, create
one with an optional name that identifies its destination:

```bash
miniverse token create --name "gpu-runner" --json
```

If your saved login predates token management, run `miniverse auth login` again
once to authorize the required `tokens:manage` scope.

The token value is returned only once. Transfer it directly to the destination's
secret manager and expose it to Miniverse commands as `MINIVERSE_API_TOKEN`.
Do not pass it as an argument, write it into repository files, or include it in
logs. `MINIVERSE_API_TOKEN` takes precedence over an OAuth credential.

List token metadata or delete a token from any authenticated workspace:

```bash
miniverse token list --json
miniverse token delete TOKEN_ID --json
```

Delete tokens as soon as their remote environment or job no longer needs them.
Agents should prefer a personal API token instead of copying a renewable OAuth
credential into remote environments or ephemeral machines.

For a person at a terminal, run `miniverse auth login` once. Complete the
displayed device authorization in the browser. The CLI stores the renewable
OAuth grant in a user-only file and refreshes it automatically, including after
an API response reports that the access token expired.

On Linux the default OAuth state file is `$XDG_STATE_HOME/miniverse/auth.json`, or
`~/.local/state/miniverse/auth.json` when `XDG_STATE_HOME` is unset. Set
`MINIVERSE_HOME` to put `auth.json` in a different persistent directory. An
agent using OAuth must mount that directory across runs; prefer a personal API
token when the machine itself is remote or ephemeral.

The renewable credential format is a hard cutover. A pre-0.2 keyring login is
not read, copied, or deleted; run `miniverse auth login` once after upgrading.

Inspect only non-secret metadata with `miniverse auth status`. Run
`miniverse auth logout` to remove the local OAuth credential and attempt
server-side revocation; local removal still succeeds if revocation is
unavailable. To opt back into an available desktop credential service, set
`MINIVERSE_AUTH_STORE=keyring`.
