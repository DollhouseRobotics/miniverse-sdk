# Miniverse agent guide

You are using `miniverse-sdk` 0.4.5. Miniverse is Dollhouse Robotics'
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
```

Read `onnx` BEFORE exporting a checkpoint: it lists the TensorRT-RTX
compatibility rules (no full-axis sorts, static TopK K, no data-dependent
control flow) that determine whether Miniverse can serve the checkpoint
through optimized inference instead of the ONNX fallback.
