# Bundle validation

A `.mini` archive contains `bundle.json`, `policy.py`, an MJCF embodiment
entrypoint and its dependencies under `embodiment/`, one or more
`models/<id>.onnx` files, and optionally an environment source and its
dependencies. Declare the embodiment as
`{"kind":"mjcf","path":"embodiment/robot.xml"}`. Do not put a
`scene`, `seed`, `robot`, `visual`, `primaryModel`, model providers,
or model loading policy in `bundle.json`.

Omit `environment` for Miniverse flat ground. Otherwise declare the source kind
and its safe bundle-relative path. Read `miniverse agent-help environments` for
scene ownership and packaging, and `miniverse agent-help terrain` before
converting height arrays.

The Python controller declares
its physics/policy/publication timing, owns randomness, sees all models through
`context.models`, constructs all inputs from `PolicyStep.sim_data`, and returns
physical actuation in canonical MJCF actuator order. Optional `metadata` is a
freeform JSON object and has no execution semantics.

Set `primarySimulator` to `mujoco`, `browser-mujoco`,
`isaac-sim-cpu-physx`, or `isaac-sim-gpu-physx`. The `browser-mujoco` profile
uses the model's `mujoco-cpu` compatibility declaration.

For a permanent rigid attachment, make the attached object a jointless child
body in the embodiment MJCF. To close a second attachment point, add a standard
MJCF `<equality><weld .../></equality>` using either two sites or one or two
bodies. Welds do not add actuator DOFs; simulator support is checked by the
server for the selected profile.

`viewer.camera` and `viewer.worldBend` are independent, optional presentation
preferences; `viewer` may be empty. To change only this bundle's automatic
camera framing, add:

```json
"viewer": {"camera": {"framingScale": 1.17}}
```

`framingScale` accepts `0.5` through `3`. `1` is exactly the stock fit, values
above `1` show more of the scene, and values below `1` show less. The setting
applies to the initial view, **Reset camera**, generated preview, and public
embed. It does not change simulation state, terrain, tracking controls, or
episode resets.

World bending is enabled by default when `worldBend` is omitted. Prefer world
bending when the terrain is flat and the scene is relatively simple. For uneven
terrain or more complex scenes where curvature obscures the authored layout,
disable it independently of camera framing:

```json
"viewer": {"worldBend": {"enabled": false}}
```

Set `enabled` to `true` to request world bending explicitly. This setting is
presentation only and applies consistently on all viewer surfaces; it does not
change simulation state or physics. Omit either preference
unless the bundle deliberately needs it; validation does not inject defaults.

## Gamepad controls

Add a `builtin/gamepad` UI
component for each command to be controlled. Its required `id` and `commandId`
have their usual meanings, while `options` contains only a required `bindings`
array (1–32 entries). The viewer combines all gamepad components into one
selector. A gamepad component can coexist with another component for the same
command, such as a touch `joystick2d`; a hidden command is also allowed when it
should be gamepad-only. There may be at most one gamepad component per command.

Each binding requires `source` (`axis`, `button`, `magnitude`, or `angle`) and
a named `input`. `component` is a zero-based numeric command component and
defaults to `0`. Optional `min` and `max`
default to the command range bounds, or `[-1, 1]` when the command has no
range. `invert` and `toggle` default to `false`; `deadzone` defaults to `0.15`
and must be at least 0 and less than 1.

```json
"commands": [{
  "id": "walking-control", "kind": "joystick2d",
  "default": [0, 0], "range": [-1, 1], "step": 0.05,
  "frame": "base", "sliceLength": 2, "update": "continuous"
}],
"ui": {"components": [{
  "id": "walking-gamepad",
  "renderer": "builtin/gamepad",
  "commandId": "walking-control",
  "options": {"bindings": [
    {"component": 0, "source": "axis", "input": "left-stick-x"},
    {"component": 1, "source": "axis", "input": "left-stick-y", "invert": true}
  ]}
}]}
```

The exact inputs are:

- `axis`: `left-stick-x`, `left-stick-y`, `right-stick-x`, `right-stick-y`;
- `button`: `south`, `east`, `west`, `north`, `left-bumper`, `right-bumper`,
  `left-trigger`, `right-trigger`, `select`, `start`, `left-stick-press`,
  `right-stick-press`, `dpad-up`, `dpad-down`, `dpad-left`, `dpad-right`, `home`;
- `magnitude` and `angle`: `left-stick`, `right-stick`.

Face-button names are positional: south=A/Cross, east=B/Circle,
west=X/Square, and north=Y/Triangle. The triggers are left-trigger=LT/L2 and
right-trigger=RT/R2. Here `walking-control` is ordered `[turn, forward]`:
the horizontal axis drives turn and the inverted vertical axis drives forward.
Axis input is signed.
Buttons and stick magnitude are unipolar and map from zero through `min..max`.
Set `min: 0` when a trigger or magnitude controls nonnegative speed.
Stick angle maps up to the output midpoint and right to three quarters of the
output range; the centered stick holds the current command. Bindings use the
command's declared units and coordinate frame.

Choose a unique, in-bounds command component for each binding and an output
interval with `min < max` within the command range. Use `button` sources for
`boolean` and `momentary` commands; `toggle` applies to `boolean` commands.
Gamepad bindings support `joystick2d`, `scalar`, `boolean`, and `momentary`
commands that continue the current episode. The browser uses the
standard Gamepad API mapping.

## Bundle metadata conventions

- **Name:** Use a simple name for the policy and what it does. Do not include
  information provided by other metadata or easy to inspect, such as the
  embodiment.
- **Description:** Describe the task the policy performs or its purpose. Omit
  low-level details such as epochs and hyperparameters.
- **Attributes:** Add all available policy and provenance information, such as
  the project page, GitHub repository, authors, paper, and license. Manage these
  web-only references with the Miniverse MCP after upload; they do not change
  the immutable bundle or its digest.

For a policy-specific spawn/reset, define `initial_state()` on the controller
and return `ControllerInitialState` with the canonical root body ID, a
world-frame position, and an XYZW unit quaternion. Do not put reset state in
`bundle.json`; without the hook Miniverse uses the MJCF's compiled default pose.

To start a new episode when a command changes, set `"restart": "episode"` on
that command. Its `update` must be `"on-release"` or `"edge"`, not
`"continuous"`. Define `reset_state(commands)` to choose the new episode's
`ControllerInitialState` from the current command values. Miniverse calls it
after `initialize(context)`. Without `reset_state(commands)`, Miniverse uses
`initial_state()`. **Reset simulation** follows the same behavior with the
current command values; **Reset camera** does not reset the policy or episode.

Miniverse also derives rendering-only visuals from the included MJCF and
`embodiment.appearance.geometry`; do not include visual bytes yourself.
MJCF geoms may be both visible and collision-active. Authors can add
physics-neutral custom visuals with `contype="0" conaffinity="0" mass="0"`
without changing the bundle schema or simulation dynamics. Author visual `pos`,
`quat`, and `fromto` values in the MJCF source frame.

Declare spatial tendons in the embodiment MJCF. The viewer renders tendons
whose routes consist entirely of sites as straight segments connecting those
sites in declaration order. Each site is an attachment or routing point fixed
in its owning body's local frame; its displayed position follows that body's
pose. The visual preserves the compiled MJCF `width` and `rgba`.

Tendon visualization supports site-only spatial routes. Routes containing
wrapping geoms or pulleys, and fixed tendons defined by joint coefficients, are
not rendered. This limitation concerns visualization, not the tendon physics
supported by the selected simulation backend. The visual represents the tendon
path, not individual chain links or a deformable pipe mesh.

Author actuator gains, control/force ranges, and Miniverse's namespaced
actuator velocity-limit numeric directly in the embodiment MJCF. Use
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
