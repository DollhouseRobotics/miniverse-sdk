# Miniverse agent guide

You are using `miniverse-sdk` 0.4.2. Miniverse is Dollhouse Robotics'
cloud-based robotics physics simulation platform.

Start with:

```bash
miniverse version --json
miniverse auth status --json
miniverse bundle validate PATH.mini --json
```
Use `miniverse auth login` once for a persistent agent workspace; the CLI
refreshes that grant automatically. Use `MINIVERSE_API_TOKEN` for CI. Never
print credentials or signed transfer URLs.

Read the relevant topic before acting:

```bash
miniverse agent-help auth
miniverse agent-help bundles
miniverse agent-help environments
miniverse agent-help onnx
miniverse agent-help terrain
miniverse agent-help upload
miniverse agent-help sessions
```

Read `onnx` BEFORE exporting a checkpoint: it lists the TensorRT-RTX
compatibility rules (no full-axis sorts, static TopK K, no data-dependent
control flow) that determine whether Miniverse can serve the checkpoint
through optimized inference instead of the ONNX fallback.
