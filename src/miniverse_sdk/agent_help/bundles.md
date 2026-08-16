# Bundle validation

A `.dhsim` archive contains `bundle.json`, `policy.py`, `scene.glb`, one or more
`models/<id>.onnx` files, and optional `robot/mjcf.zip` and
`visual/robot.glb`. Every executable or binary member is SHA-256-bound by the
manifest.

Run `miniverse bundle validate PATH.dhsim --json` before upload. Treat local
validation as feedback; the server-side importer is authoritative. Do not alter
a bundle after recording its archive hash. Preserve simulator profile, model,
scene, embodiment, coordinate-frame, actuator-order, and provenance identities.
