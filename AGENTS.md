# AGENTS.md

## Purpose

This is the Miniverse SDK/CLI, designed to be used by AI agents, in order to perform operations on the Miniverse robotics
cloud simulation platform. With it, agents can upload miniverse bundles, test them, and more.

This repository owns the public `miniverse-sdk` Python distribution, its
`miniverse` command, package tests, versioned agent help, and the Miniverse
agent skill.

## Development

Use Python 3.10 or newer.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Keep the package version in `pyproject.toml`, `src/miniverse_sdk/__init__.py`,
and `src/miniverse_sdk/agent_help/index.md` synchronized. Package the Apache
2.0 license, JSON schemas, and agent-help Markdown in both the wheel and source
distribution.

The installed CLI owns the current Miniverse bundle and API instructions.
Update `agent-help` with contract changes. Keep `skills/miniverse/SKILL.md`
small enough to install easily and make it direct agents to the installed
`miniverse agent-help` output.

Never commit credentials, bundle archives, checkpoints, signed upload URLs, or
OAuth state.

## Updating `agent-help`

`agent-help` is meant to guide AI agents on how to use the Miniverse SDK. When changes are made to the SDK, `agent-help` must be updated to provide proper guidance, based on the changes.

Do not include any internal platform details in the help. These docs should be geared toward user/agent facing API surfaces and behavior.

Do not include any migration or information about how the SDK used to work before new changes. Docs should only represent the current state of the SDK, in the present tense.

Do not include language about something that cannot be done, unless it's to clarify some limitation that the developer explictly asked you to add. If you want to recommend such copy, you may,
but it must be approved by the developer first.

## Deployment

Use proper semver versions when publishing a new version.

Use `uv` to publish to pypi.
