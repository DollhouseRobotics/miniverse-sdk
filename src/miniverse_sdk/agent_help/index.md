# Miniverse agent guide

You are using `miniverse-sdk` 0.1.0. Miniverse is Dollhouse Robotics' standalone
simulation product. It owns immutable `.dhsim` bundles, browser sessions,
Cloudflare lifecycle control, Modal preprocessing, and the universal Vast fleet.
Do not introduce an Optics dependency.

Start with:

```bash
miniverse version --json
miniverse auth status --json
miniverse bundle validate PATH.dhsim --json
```
Use `MINIVERSE_API_TOKEN` for automation or `miniverse auth login` for an
interactive device flow. Never print credentials or signed transfer URLs.

Read the relevant topic before acting:

```bash
miniverse agent-help auth
miniverse agent-help bundles
miniverse agent-help upload
miniverse agent-help sessions
```
