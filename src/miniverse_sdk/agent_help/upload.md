# Bundle upload

Run:

```bash
miniverse bundle upload PATH.dhsim --json
```

The CLI uploads the exact archive directly to a short-lived R2 capability. The
server verifies and expands it into internal content-addressed assets, captures
the initial state, validates model derivatives, and returns a durable import
record. Use `--no-wait` to detach and `miniverse bundle status UPLOAD_ID` to
resume observation.

An uploaded bundle is private and ready by default. Publication is a separate,
explicit `miniverse bundle publish ID@DIGEST` action. Never infer publication
or simulator readiness from an HTTP 202, object presence, or provider status.
