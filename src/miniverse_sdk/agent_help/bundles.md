# Bundle validation

A `.dhsim` archive contains `bundle.json`, `policy.py`, `scene.glb`, one or more
`models/<id>.onnx` files, and optional `robot/mjcf.zip` and
`visual/robot.glb`. Every executable or binary member is SHA-256-bound by the
manifest.

Each ONNX checkpoint must be self-contained and embed all Miniverse policy
metadata in
`com.dollhouserobotics.miniverse.simulation_contract`. Do not create an
adjacent JSON manifest. The embedded contract must include:

- complete input/observation mappings;
- complete output/action mappings;
- `precision: fp32 | fp16 | bf16`.

Precision is the requested Miniverse inference/compiler precision; it does not
assert that every tensor in the ONNX graph has that dtype. Missing and unknown
precision values are rejected. Exporters using
`miniverse.contracts.embed_onnx_contract` should pass
`{"schemaVersion": "0.2", "precision": "fp16", ...}`.

Run `miniverse bundle validate PATH.dhsim --json` before upload. Treat local
validation as feedback; its `model_precisions` result reports the discovered
precision for each model, and the server-side importer is authoritative. Do not alter
a bundle after recording its archive hash. Preserve simulator profile, model,
scene, embodiment, coordinate-frame, actuator-order, and provenance identities.

ONNX is the developer-facing checkpoint submission artifact. Developers do not
build or upload TensorRT, TensorRT-RTX, plan, or runtime-cache files; Miniverse
derives and invalidates those artifacts internally.
