# Bundle environments

An environment defines the world around an embodiment. Put the environment
entrypoint and all of its dependencies under `environment/` in the `.mini`
archive, then declare the entrypoint in `bundle.json`.

GLB environments support only canonical data-only heightfields produced by
`miniverse terrain build`:

```json
"environment": {
  "kind": "glb",
  "path": "environment/world.glb"
}
```

For an MJCF environment:

```json
"environment": {
  "kind": "mjcf",
  "path": "environment/world.xml"
}
```

Paths are safe bundle-relative paths. Keep referenced includes and STL meshes
under `environment/`, and use relative references between them. The complete
subtree must be the entrypoint's dependency closure: missing, unused, escaping,
case-colliding, or cyclic members fail validation. Preserve source coordinate
frames, units, transforms, joint axes and limits, masses, inertias, collision
properties, friction, damping, materials, and asset identities.

Separate the embodiment from the environment by control ownership:

- The MJCF entrypoint declared by `embodiment.path`, together with its relative
  dependencies under `embodiment/`, defines the policy-controlled robot: its
  bodies, joints, actuators, robot sensors, collision geometry, and canonical
  actuator ordering.
- `environment` contains the surrounding world: terrain, ground, buildings,
  props, obstacles, and mechanisms that are not controlled as robot actuators.
- Articulation alone does not make an asset part of the embodiment. A swing set
  whose seats move on passive joints belongs in the environment. Preserve its
  bodies, joints, limits, damping, mass, inertia, and collision definitions in
  the environment source.
- A mechanism directly controlled by the policy belongs in the embodiment so
  its actuators participate in the embodiment's explicit action contract and
  canonical actuator order.

The embodiment and environment may contact and move one another while keeping
their identities and state namespaces separate. Environment bodies and passive
joints remain environment state; they are not appended to the robot's actuator
order.

MJCF environments support `mujoco`, `isaac-sim-cpu-physx`, and
`isaac-sim-gpu-physx` as the primary simulator or as compatible simulators.
Each backend compiles the same environment source independently. Environment
body transforms stream under `environment:<body-name>` object IDs. Policy code
remains backend-neutral, and environment joints never enter the embodiment
joint or actuator order.

Supported MJCF content across MuJoCo and Isaac includes fixed and unactuated
passive rigid bodies; hinge, slide, ball, and free joints; plane, sphere,
capsule, ellipsoid, cylinder, box, and STL mesh geoms; native limits, damping,
friction, collision masks, masses, inertias, defaults, and RGBA materials. MJCF
lights and cameras are accepted by MuJoCo, but the browser uses its own camera
and lighting.

Put `<compiler>` only in the entrypoint. Its supported attributes are `angle`,
`meshdir`, `texturedir`, and `autolimits`. `assetdir`, `strippath`, and other
compiler attributes are rejected. Textures, skins, MJCF heightfields, non-STL
meshes, actuators, sensors, tendons, equality or explicit contact sections,
keyframes, custom or global settings, plugins, extensions, flex, deformable,
composite elements, and other geom types are rejected.

A bundle may contain at most 1,024 ZIP members. An MJCF environment may contain
at most 4,096 named bodies, 8,192 joints, and 16,384 geoms. Every body must have
a unique name matching `[A-Za-z0-9][A-Za-z0-9._-]{0,119}`; names must also stay
unique when `-` and `.` are normalized to `_` for Isaac USD identifiers. The
browser displays an MJCF plane using each positive authored X/Y half-size and
substitutes a 20 m half-size for any zero axis; physics collision remains an
infinite plane.

For a heightfield sourced from a numeric grid, generate the GLB with
`miniverse terrain build`, declare that generated file as a `glb` environment,
and read `miniverse agent-help terrain` for its coordinate and query contract.

Run `miniverse bundle validate PATH.mini --json` before upload. It checks the
schema, backend declaration, dependency closure, paths, and supported XML
sections. Upload validation also compiles the environment with MuJoCo and
checks body identities, count limits, and deterministic browser geometry.
