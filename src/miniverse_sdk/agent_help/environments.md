# Bundle environments

An environment defines the world around an embodiment. The current V1
authoring contract accepts one canonical, data-only heightfield GLB. Build it
with `miniverse terrain build`, put it under `environment/` in the `.mini`
archive, and declare that exact path in `bundle.json`.

```json
"environment": {
  "kind": "glb",
  "path": "environment/terrain.glb"
}
```

The path is bundle-relative and may use subdirectories, but the GLB must be the
self-contained heightfield form produced by the SDK. General GLB scenes, MJCF
environments, referenced environment dependencies, props, and passive
mechanisms are planned capabilities; SDK 0.4 does not accept them.

Separate the embodiment from the environment by control ownership:

- The MJCF entrypoint declared by `embodiment.path`, together with its relative
  dependencies under `embodiment/`, defines the policy-controlled robot: its
  bodies, joints, actuators, robot sensors, collision geometry, and canonical
  actuator ordering.
- `environment` contains the surrounding world. In the current contract that
  means the one static heightfield; future scene support will also cover
  buildings, props, obstacles, and passive mechanisms.
- Under the planned general-scene contract, articulation alone will not make an
  asset part of the embodiment. A swing set whose seats move on passive joints
  will belong in the environment, with its physical definitions preserved.
- A mechanism directly controlled by the policy belongs in the embodiment so
  its actuators participate in the embodiment's explicit action contract and
  canonical actuator order.

The embodiment and heightfield contact one another while keeping their
identities separate. The environment does not add actuators or joints to the
robot's action ordering.

Choose `primarySimulator` for the authoritative execution target and list only
the other verified targets in `compatibleSimulators`. Miniverse derives the
browser mesh and backend collision representations from the same immutable
float grid. Policy code remains backend-neutral and queries the stable terrain
ID rather than simulator-private handles.

For a heightfield sourced from a numeric grid, generate the GLB with
`miniverse terrain build`, declare that generated file as a `glb` environment,
and read `miniverse agent-help terrain` for its coordinate and query contract.

Run `miniverse bundle validate PATH.mini --json` before upload. Validation
checks the declared path, canonical heightfield structure, backend
compatibility, identities, and environment/embodiment separation.
