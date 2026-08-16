# Authentication

For CI or another non-interactive environment, provide the token only through
the environment:

```bash
MINIVERSE_API_TOKEN=... miniverse bundle upload PATH.dhsim --json
```

Do not pass a token as an argument, write it into repository files, or include
it in logs. `MINIVERSE_API_TOKEN` takes precedence over an OAuth credential.

For a person at a terminal, run `miniverse auth login`. Complete the displayed
device authorization in the browser. Inspect only the credential source with
`miniverse auth status`; remove the OAuth credential with `miniverse auth logout`.
