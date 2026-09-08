# Miniverse agent guide

You are using `miniverse-sdk` 0.6.0. Miniverse is Dollhouse Robotics'
cloud-based robotics physics simulation platform.

Start with:

```bash
miniverse version --json
miniverse auth status --json
miniverse bundle validate PATH.mini --json
```
Use `miniverse auth login` once for a trusted persistent agent workspace; the
CLI refreshes that grant automatically. For CI, remote environments, and
ephemeral machines, create a personal token with `miniverse token create` and
provide it through `MINIVERSE_API_TOKEN`. Never print credentials or signed
transfer URLs except for capturing a newly created token directly into a secret
manager.

Read the relevant topic before acting:

```bash
miniverse agent-help auth
miniverse agent-help bundles
miniverse agent-help environments
miniverse agent-help mcp
miniverse agent-help onnx
miniverse agent-help terrain
miniverse agent-help upload
miniverse agent-help sessions
miniverse agent-help tests
```

Read `onnx` BEFORE exporting a checkpoint: it lists the TensorRT-RTX
compatibility rules (no full-axis sorts, static TopK K, no data-dependent
control flow) that determine whether Miniverse can serve the checkpoint
through optimized inference instead of the ONNX fallback.

Run a test after the first upload of a bundle and whenever you upload major
changes, such as a new checkpoint, policy logic, embodiment, environment, or
command behavior. Read `miniverse agent-help tests`, then test the exact uploaded
revision:

```bash
miniverse test start ID@REVISION_ID --file test_policy.py --json
```

Write checks for the bundle's intended behavior, including observable motion
and command-to-effect where applicable. Inspect the report and fix failures
before declaring the bundle working. Upload success or a `running` session
status alone is not evidence that the policy works.
