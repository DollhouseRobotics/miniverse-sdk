# Heightfield terrain

Miniverse accepts one static, immutable heightfield per bundle. Start from a
two-dimensional `.npy` array or a JSON array of numeric rows, then run:

```bash
miniverse terrain build heights.npy environment/terrain.glb \
  --id climb-001 \
  --cell-size 0.05 0.05 \
  --origin -2.0 -2.0 0.0 \
  --out-of-bounds clamp \
  --json
```

Rows advance along world `+Y`; columns advance along world `+X`. `--origin`
is the world-space `X Y Z` position of row 0, column 0. Samples are converted
to meters as `origin.z + vertical_offset + vertical_scale * sample`. The
source frame is right-handed, `+Z` up, `+X` forward, and meters. Do not rotate
or transpose a grid to make the preview look right; fix the array conversion
and verify known lattice points instead.

Choose the out-of-bounds rule deliberately:

- `error` fails a policy query outside the grid;
- `clamp` returns the nearest edge sample;
- `constant` requires `--out-of-bounds-value VALUE` in meters.

The command refuses to overwrite an existing GLB unless `--force` is passed.
It emits a data-only GLB: Miniverse derives the render mesh, MuJoCo heightfield,
Isaac collision mesh, and `heightmap_nearest` observations from the same
embedded float grid. The result is capped at 1,048,576 samples.

Add this exact source declaration to `bundle.json`:

```json
"environment": {"kind": "glb", "path": "environment/terrain.glb"}
```

Then pack the GLB at the declared archive path and run
`miniverse bundle validate PATH.mini --json`. Validation reports the terrain
ID, grid SHA-256, dimensions, origin, XY resolution, scale/offset, and
out-of-bounds behavior. In policy code, use the same ID with
`step.sim_queries.heightmap_nearest("climb-001", points)`.
