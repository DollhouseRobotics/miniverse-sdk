# Miniverse agent guide

You are using `miniverse-sdk` 0.1.0. Miniverse is Dollhouse Robotics'
cloud-based robotics physics simulation platform.

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
miniverse agent-help onnx
miniverse agent-help upload
miniverse agent-help sessions
```

Read `onnx` BEFORE exporting a checkpoint: it lists the TensorRT-RTX
compatibility rules (no full-axis sorts, static TopK K, no data-dependent
control flow) that determine whether Miniverse can serve the checkpoint
through optimized inference instead of the ONNX fallback.
