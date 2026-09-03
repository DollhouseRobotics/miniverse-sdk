# Bundle validation

A `.mini` archive contains `bundle.json`, `policy.py`, mandatory
`embodiment/mjcf.zip`, one or more `models/<id>.onnx` files, and optionally
`environment/terrain.glb`. Do not put a
`scene`, `seed`, `robot`, `visual`, `primaryModel`, model providers,
or model loading policy in `bundle.json`.

Omit `environment` for Miniverse flat ground. To use one static heightfield,
put `"environment": {"kind": "heightfield"}` in `bundle.json` and generate
the fixed `environment/terrain.glb` member with `miniverse terrain build`.
Read `miniverse agent-help terrain` before converting height arrays.

The Python controller declares
its physics/policy/publication timing, owns randomness, sees all models through
`context.models`, constructs all inputs from `PolicyStep.sim_data`, and returns
physical actuation in canonical MJCF actuator order. Optional `metadata` is a
freeform JSON object and has no execution semantics.

For a policy-specific spawn/reset, define `initial_state()` on the controller
and return `ControllerInitialState` with the canonical root body ID, a
world-frame position, and an XYZW unit quaternion. Do not put reset state in
`bundle.json`; without the hook Miniverse uses the MJCF's compiled default pose.
Miniverse also derives rendering-only visuals from the included MJCF and
`embodiment.appearance.geometry`; do not include visual bytes yourself.
MJCF geoms may be both visible and collision-active. Authors can add
physics-neutral custom visuals with `contype="0" conaffinity="0" mass="0"`
without changing the bundle schema or simulation dynamics. Author visual `pos`,
`quat`, and `fromto` values in the MJCF source frame.

Author actuator gains, control/force ranges, and Miniverse's namespaced
actuator velocity-limit numeric directly in `embodiment/mjcf.zip`. Use
`embodiment.bodyDynamicsOverrides` only for exact named-body linear/angular
damping and maximum linear/angular velocity. Unsupported backends produce a
descriptive runtime error.

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

Run `miniverse bundle validate PATH.mini --json` before upload. Local validation
checks the archive structure, normative manifest, ONNX graph and embedded
contract, hashes, tensor mappings, fixed batch-one shapes, simulator support,
and statically knowable compatibility. It returns `errors`, `warnings`, and a
per-model result. Errors always produce exit status 2. Optimization warnings
produce exit status 0 unless `--strict` promotes them to errors. Server import
repeats validation over the uploaded bytes and remains authoritative for
embodiment/environment preprocessing and genuine backend execution. Treat an uploaded
bundle as immutable and preserve simulator profile, model, embodiment,
coordinate-frame, and actuator-order identities.

Validation also statically scans each ONNX graph for TensorRT builder limits
and reports them in `warnings` (for example `tensorrt_topk_k_limit`: TopK `K`
must be at most 3840, and a full sort lowers to TopK over the entire axis).
Warning findings do not invalidate a bundle — the ONNX Runtime fallback still
executes it — but Miniverse cannot derive TensorRT or TensorRT-RTX artifacts
until they are fixed, so re-export the checkpoint before uploading. Error
findings always invalidate the bundle. Pass `--strict` to promote optimization
warnings to errors. To validate a checkpoint during export iteration,
before it is zipped into a bundle, run:

```bash
miniverse model validate PATH.onnx --json
```

ONNX is the developer-facing checkpoint submission artifact. Developers do not
build or upload TensorRT, TensorRT-RTX, plan, or runtime-cache files; Miniverse
derives and invalidates those artifacts internally.
