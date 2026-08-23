# Bundle upload

Run:

```bash
miniverse bundle upload PATH.dhsim --json
```

The CLI handles the upload; the server verifies and expands the archive and
returns a durable import record. Use `--no-wait` to detach and
`miniverse bundle status UPLOAD_ID` to resume observation.

An uploaded bundle is private and ready by default. Publication is a separate,
explicit `miniverse bundle publish ID@DIGEST` action. Never infer publication
or simulator readiness from an HTTP 202, object presence, or provider status.
