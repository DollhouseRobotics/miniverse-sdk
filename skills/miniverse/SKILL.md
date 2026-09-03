---
name: miniverse
description: Work with Dollhouse Robotics Miniverse simulation bundles. Use for authoring, validating, inspecting, uploading, or publishing .mini bundles; packaging MJCF embodiments, ONNX policies, Python controllers, commands, or gizmos; and diagnosing Miniverse bundle-import failures.
---

# Miniverse

Miniverse is Dollhouse Robotics' cloud-based robotics physics simulation
platform.

Before performing Miniverse work:

1. Require `uv`; if it is unavailable, stop and explain that the supported
   isolated installer is required.
2. Run `uv tool list`. If `miniverse-sdk` is installed, run
   `uv tool upgrade miniverse-sdk`. Otherwise run
   `uv tool install miniverse-sdk`.
3. Run `miniverse version --json` and require
   `distribution: miniverse-sdk`. Report an executable collision instead of
   using an unrelated `miniverse` command.
4. Run `miniverse agent-help`, then run the relevant topic named by that output.
5. Follow the installed CLI's instructions because they are versioned with its
   supported API.

When authoring or diagnosing controller observations:

- Author the canonical archive layout: `bundle.json`, `policy.py`,
  an MJCF entrypoint and its dependencies under `embodiment/`,
  `models/<id>.onnx`, and an optional environment entrypoint and dependencies
  under `environment/`. Use `miniverse terrain build` for a heightfield and
  read `miniverse agent-help environments` and `miniverse agent-help terrain`.

- Treat `PolicyStep.sim_data` as the simulator-neutral physics snapshot. Read
  contract ordering and static/runtime model values from `sim_data.model`; read
  joint, body, contact, tick, and time state from `sim_data`. Keep policy code
  simulator-neutral and use the public controller interfaces.
- Construct every ONNX input in bundle controller code from `sim_data`,
  commands, deterministic controller state, and declared constants.
- Treat every `context.models` entry as an equal named model handle and let the
  runtime select execution providers. Return `StepResult.actuation` in
  canonical MJCF actuator order and physical units.
- Put `physics_hz`, `policy_hz`, `publish_hz`, and `control_loop` on the Python
  controller class. The controller owns all randomness: hardcode or construct
  its RNG there when needed.
- Put a policy/task-specific spawn pose in an optional controller
  `initial_state()` method that returns `ControllerInitialState` with the exact
  canonical root body ID, world-frame position, XYZW unit quaternion, and any
  named joint overrides. When no hook is present, Miniverse uses the MJCF's
  compiled default pose.
- Treat embodiment visuals as platform-derived output selected by
  `embodiment.appearance.geometry`, with simulation geometry authored in MJCF.
- Author actuator gains and limits in the embodiment MJCF. Put per-body
  linear/angular damping and maximum linear/angular velocity under
  `embodiment.bodyDynamicsOverrides`, using exact MJCF body IDs.
- Use `step.sim_queries.height_samples(...)` for live collision raycasts and
  `step.sim_queries.heightmap_nearest(...)` for authoritative grid sampling.
- Put deterministic selection, normalization, thresholds, and feature math in
  the controller/ONNX graph. Backends only capture canonical state, apply
  actuation, step physics, and perform authoritative collision queries.
- Let `miniverse bundle validate` catch statically knowable mistakes and
  preserve `SIM_DATA_UNAVAILABLE` runtime failures with the operation, selected
  backend, supporting backends, exact request, and a useful remediation.
- Keep optional bundle `metadata` freeform and non-operative.
- For rough terrain, use `heightmapNearest` only with a valid `terrainId`,
  keeping its origin, XY resolution, vertical scale/offset, and out-of-bounds
  rule explicit. Use `heightSamples` for live collision raycasts.

Prefer `miniverse auth login` for a long-lived agent workspace; the installed
CLI retains and refreshes that authorization automatically. Persist the CLI
state directory when the agent itself is ephemeral. Use `MINIVERSE_API_TOKEN`
for CI or intentionally non-interactive authentication. Never print, persist,
or pass that environment token as a command-line argument. Treat upload, import
readiness, and publication as separate claims.
