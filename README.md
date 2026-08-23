# Miniverse SDK

The `miniverse-sdk` distribution installs the `miniverse` command for validating
and uploading immutable `.dhsim` simulation bundles.

```bash
uv tool install miniverse-sdk
miniverse agent-help
```

Use `MINIVERSE_API_TOKEN` for CI authentication, or run `miniverse auth login`
once for interactive device authorization. Interactive access and refresh
tokens are stored in a user-only file and renewed automatically. On ephemeral
Linux agents, persist `$XDG_STATE_HOME/miniverse` (or `~/.local/state/miniverse`).
