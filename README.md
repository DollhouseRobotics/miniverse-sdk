# Miniverse SDK

`miniverse-sdk` installs the `miniverse` command for validating and uploading
immutable `.mini` simulation bundles to [Miniverse](https://miniverse.bot). It
also builds canonical heightfield assets from ordinary 2-D `.npy` or JSON
arrays so the viewer, collision backend, and policy queries share one grid.

## Install

```bash
uv tool install miniverse-sdk
miniverse agent-help
```

Use `MINIVERSE_API_TOKEN` for CI authentication, or run:

```bash
miniverse auth login
```

Interactive access and refresh tokens are stored in a user-only file and
renewed automatically. On ephemeral Linux agents, persist
`$XDG_STATE_HOME/miniverse`, or `~/.local/state/miniverse`.

## Heightfield terrain

```bash
miniverse terrain build heights.npy environment/terrain.glb \
  --id experiment-001 --cell-size 0.05 0.05 --out-of-bounds clamp --json
```

Add `"environment": {"kind": "heightfield"}` to `bundle.json`, then pack the
generated file at that exact path. Run `miniverse agent-help terrain` for the
coordinate-frame, origin, scaling, and policy-query contract.

## Agent skill

The repository includes a Miniverse skill at
[`skills/miniverse/SKILL.md`](skills/miniverse/SKILL.md). Install that skill in
your agent, then follow its setup steps. The skill installs or upgrades this
package and reads the versioned instructions bundled with the CLI through
`miniverse agent-help`.

## Develop

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Miniverse server-side import remains authoritative. Local validation is
preflight feedback for bundle authors and agents.

## License

Apache License 2.0. See [LICENSE](LICENSE).
