# Bundle environments

An environment defines the world around an embodiment. Put the environment
entrypoint and all of its dependencies under `environment/` in the `.mini`
archive, then declare the entrypoint in `bundle.json`.

For a GLB environment:

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

Paths are safe bundle-relative paths. Keep referenced meshes, textures,
heightfields, includes, and other source files under `environment/`, and use
relative references between them. Preserve source coordinate frames, units,
transforms, joint axes and limits, masses, inertias, collision properties,
friction, damping, materials, and asset identities.

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

Choose `primarySimulator` for the authoritative execution target and list only
the other verified targets in `compatibleSimulators`. Miniverse preprocesses
the declared environment into immutable, content-addressed artifacts for the
viewer and each declared simulation backend. MuJoCo uses the validated MJCF
scene representation, Isaac Sim uses its validated scene representation, and
the browser viewer uses the derived GLB representation. Policy code remains
backend-neutral and addresses stable Miniverse body, joint, object, and query
identities rather than simulator-private handles.

For a heightfield sourced from a numeric grid, generate the GLB with
`miniverse terrain build`, declare that generated file as a `glb` environment,
and read `miniverse agent-help terrain` for its coordinate and query contract.

Run `miniverse bundle validate PATH.mini --json` before upload. Validation
checks the declared entrypoint, dependency closure, paths, source structure,
backend compatibility, identities, and environment/embodiment separation.
