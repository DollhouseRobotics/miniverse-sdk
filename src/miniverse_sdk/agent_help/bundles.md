# Bundle validation

A `.dhsim` archive contains exactly `bundle.json`, `policy.py`, mandatory
`embodiment/mjcf.zip`, and one or more `models/<id>.onnx` files. Miniverse
derives SHA-256 and byte identities from those included files. Do not put
hashes, a `scene`, `seed`, `robot`, `visual`, `primaryModel`, model providers,
or model loading policy in `bundle.json`.

The environment is `builtin/flat-ground-v1`. The Python controller declares
its physics/policy/publication timing, owns randomness, sees all models through
`context.models`, constructs all inputs from `PolicyStep.sim_data`, and returns
physical actuation in canonical MJCF actuator order. Optional `metadata` is a
freeform JSON object and has no execution semantics.

For a policy-specific spawn/reset, define `initial_state()` on the controller
and return `ControllerInitialState` with the canonical root body ID, a
world-frame position, and an XYZW unit quaternion. Do not put reset state in
`bundle.json`; without the hook Miniverse uses the MJCF's compiled `qpos0`.
Miniverse also derives the renderer-only visual GLB from the included MJCF and
`embodiment.appearance.geometry`; do not include or hash visual bytes yourself.
MJCF geoms may be both visible and collision-active. With `geometry: "auto"`,
Miniverse prefers dedicated non-colliding visual geoms when present, otherwise
renders the MJCF's collision-active geoms, and uses the inertial shell only
when the model contains no renderable geoms. Authors can add physics-neutral
custom visuals with `contype="0" conaffinity="0" mass="0"` without changing
the bundle schema or simulation dynamics.

Use `embodiment.dynamicsOverrides` for actuator values. Use
`embodiment.bodyDynamicsOverrides` for exact named-body linear/angular damping
and maximum linear/angular velocity. Those per-body fields run on the two
Isaac PhysX profiles and produce a descriptive runtime error on MuJoCo.

Each ONNX checkpoint must be self-contained and embed all Miniverse policy
metadata in
`com.dollhouserobotics.miniverse.simulation_contract`. Do not create an
adjacent JSON manifest. The embedded contract must include:

- complete model tensor shapes and dtypes;
- model outputs consumed explicitly by controller code;
- `precision: fp32 | fp16 | bf16`.

Precision is the requested Miniverse inference/compiler precision; it does not
assert that every tensor in the ONNX graph has that dtype. Missing and unknown
precision values are rejected. Exporters using
`miniverse.contracts.embed_onnx_contract` should pass
`{"schemaVersion": "0.3", "precision": "fp16", ...}`.

Run `miniverse bundle validate PATH.dhsim --json` before upload. Treat local
validation as feedback; its `model_precisions` result reports the discovered
precision for each model, and the server-side importer is authoritative. Do not alter
a bundle after recording its archive hash. Preserve simulator profile, model,
embodiment, coordinate-frame, and actuator-order identities.

Validation also statically scans each ONNX graph for TensorRT builder limits
and reports `model_findings` (for example `tensorrt_topk_k_limit`: TopK `K`
must be at most 3840, and a full sort lowers to TopK over the entire axis).
Findings do not invalidate a bundle — the ONNX Runtime fallback still executes
it — but Miniverse cannot derive TensorRT or TensorRT-RTX artifacts until they
are fixed, so re-export the checkpoint before uploading. Pass `--strict` to
fail validation on any finding. To lint a checkpoint during export iteration,
before it is zipped into a bundle, run:

```bash
miniverse model validate PATH.onnx --json
```

ONNX is the developer-facing checkpoint submission artifact. Developers do not
build or upload TensorRT, TensorRT-RTX, plan, or runtime-cache files; Miniverse
derives and invalidates those artifacts internally.
