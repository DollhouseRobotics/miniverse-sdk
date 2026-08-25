# Bundle upload

Run:

```bash
miniverse bundle upload PATH.dhsim --json
```

The CLI creates a D1-backed bundle revision and uploads the archive directly to
the returned conditional R2 URL. An R2 object-create event starts server-side
inspection and preprocessing; the CLI does not send a completion request or
declare the archive's manifest and asset identities. Use `--no-wait` to detach
and `miniverse bundle status ID@REVISION_ID` to resume observation.

An uploaded revision remains private when it becomes ready. Publication is a
separate, explicit `miniverse bundle publish ID@REVISION_ID` action that moves
the bundle's current-revision pointer. Never infer publication
or simulator readiness from an HTTP 202, object presence, or provider status.
