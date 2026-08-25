# Exporting TensorRT-compatible ONNX checkpoints

Miniverse attempts to compile each uploaded checkpoint with TensorRT-RTX and
standard TensorRT to serve optimized inference. A checkpoint that only runs
under the ONNX Runtime fallback still works, but loses the optimized path — so
export with the compiler in mind, then lint before bundling:

```bash
miniverse model validate model.onnx --strict --json
```

Fix findings by re-exporting from the source checkpoint, never by editing
graph bytes by hand.

## Contract requirements

- Server-side import requires opset 13–21 (18 recommended) with static input
  and output shapes. Local CLI validation does not currently enforce the opset
  range, so treat a successful local result as preflight feedback rather than
  import acceptance.
- Embed the Miniverse simulation contract (schema 0.3) in the model metadata,
  including `precision` (`fp32`, `fp16`, or `bf16`). Weights stay fp32 in the
  ONNX; `precision` declares the execution precision the compilers target.
- Export a single policy step.
- Every graph op must be on the runtime allowlist (deterministic tensor math
  only); `miniverse model validate` reports violations as errors.
- Recurrent policies declare `stateBindings` pairing a state input with a
  same-shape state output; the runtime owns allocation, feedback, and reset.
- Temporal features come from declarative `historyBuffers` (capacity +
  `sampleOffsets` over any provider source), not from hand-rolled buffers.
- Actuator-target outputs are physical values: v0.3 has no output
  `scale`/`offset`. Bake PD scaling and decoding into the graph.
- `miniverse model validate` also lints exact per-input operations against each
  simulator profile. Fix `unsupported` rows before uploading. This is a lint,
  not a `requiredCapabilities` declaration or runtime negotiation mechanism;
  runtime failures repeat the selected and supporting backends descriptively.

## TensorRT compatibility rules

**No full-axis sorts.** `torch.sort` and full-vocabulary `torch.topk` lower to
ONNX `TopK` with K equal to the axis length, which TensorRT cannot compile.
Restructure the math instead: prefer fixed-iteration, data-independent
formulations (for example threshold-based selection built from elementwise
compares and reductions) that preserve the checkpoint's output distribution.
Do not simply cap K when doing so changes policy behavior.

**Keep every `TopK` K a graph constant.** Shape-derived or otherwise dynamic
K cannot be verified at build time and fails compilation.

**For generative policies with in-graph sampling, declare `precision: "fp16"`
only when the whole graph tolerates it.** The compilers convert every float
tensor, not just the network weights. An fp16 cumulative distribution resolves
~5e-4 absolute, so a sampler over a large vocabulary cannot draw tokens below
that probability and their mass shifts onto neighbours — a silent behavior
change that ONNX Runtime (which executes the declared fp32 graph) never shows.
Declare `fp32` for sampler-bearing graphs unless the CDF is kept in full
precision by construction.

**No data-dependent control flow.** Unroll loops to a fixed iteration count at
export; avoid `If`/`Loop`/`Scan` nodes.

**Do not change floating-point precision to gain compatibility.** Export the
exact dtypes of the source checkpoint. Never recast weights, activations, or
accumulations to lower or mixed precision merely to satisfy a compiler finding;
if a checkpoint only compiles after altering precision, treat it as
incompatible and report the finding to the user instead of uploading the
modified graph.

**For transformer-based policies, avoid known PyTorch export traps:**

- Disable the fused transformer fast path before export
  (`torch.backends.mha.set_fastpath_enabled(False)`); the fused aten op has no
  ONNX lowering.
- Avoid mutation-based indexed assignment and indexed constant construction;
  they can produce invalid-rank `ScatterElements` nodes. Use functional
  equivalents (`torch.where`, `torch.cat`, arithmetic index transforms).

## Verify, bundle, upload

```bash
miniverse model validate model.onnx --strict --json
miniverse bundle validate bundle.dhsim --strict --json
miniverse bundle upload bundle.dhsim --json
```

`bundle validate --strict` runs the same model lint across every model in the
archive. Warnings mean the checkpoint will import and run on the fallback but
cannot be compiled; treat them as errors unless the fallback is intentional.
