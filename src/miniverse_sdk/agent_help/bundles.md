# Bundle validation

A `.dhsim` archive contains exactly `bundle.json`, `policy.py`, mandatory
`embodiment/mjcf.zip`, and one or more `models/<id>.onnx` files. Do not put a
`scene`, `seed`, `robot`, `visual`, `primaryModel`, model providers,
or model loading policy in `bundle.json`.

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
precision values are rejected.

Run `miniverse bundle validate PATH.dhsim --json` before upload. Treat local
validation as feedback; its `model_precisions` result reports the discovered
precision for each model, and the server-side importer is authoritative. Treat
an uploaded bundle as immutable. Preserve simulator profile, model,
embodiment, coordinate-frame, and actuator-order identities.

Validation also statically scans each ONNX graph for TensorRT builder limits
and reports `model_findings` (for example `tensorrt_topk_k_limit`: TopK `K`
must be at most 3840, and a full sort lowers to TopK over the entire axis).
Error findings, including disallowed operators and dynamic input or output
shapes, are rejected during server import. Warning findings can use the ONNX
Runtime fallback but block the optimized path. Re-export the checkpoint before
uploading, and pass `--strict` to fail local validation on any finding. To lint
a checkpoint during export iteration, before it is zipped into a bundle, run:

```bash
miniverse model validate PATH.onnx --json
```

ONNX is the checkpoint format accepted in a `.dhsim` bundle.
