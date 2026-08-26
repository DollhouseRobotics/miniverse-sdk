# AGENTS.md

      ## Purpose

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
