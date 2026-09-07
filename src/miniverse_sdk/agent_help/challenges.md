# Challenges

## Author challenge definitions

A challenge definition is an immutable `.mini` archive with a v1
`bundle.json`, `kind: "challenge"`, and `challenge.py`. It defines one set and
one or more challenges sharing an environment and simulator profile. It must
not contain a participant embodiment, policy, or ONNX model.

```bash
miniverse challenge bundle validate challenge.mini --json
miniverse challenge bundle upload challenge.mini --json
miniverse challenge bundle status SET@REVISION --json
miniverse challenge bundle revisions SET --json
miniverse challenge bundle publish SET@REVISION \
  --expected-current CURRENT_REVISION --json
```

Upload creates an immutable private revision, transfers directly to R2, and
waits for server validation unless `--no-wait` is passed. Publishing is a
separate optimistic-concurrency operation. Pass `--expected-current null`
explicitly for a set's first publication; otherwise pass the revision ID that
is currently published. Challenge bundle authoring uses the existing
`challenges:submit` authorization.

## Submit participant policies

Challenge sets organize public, versioned evaluation contracts. Each challenge
declares its own simulator, environment, command schema, timeout, goal, and
scoring rule. Inspect the set, then choose one challenge and one policy bundle:

```bash
miniverse challenge search walking --json
miniverse challenge inspect microduck-limit --json
```

A stable submission belongs to one challenge and has immutable revisions.
Upload a `.mini` bundle, then validate and submit its ready revision:

```bash
miniverse challenge validate SET:CHALLENGE --bundle POLICY@BRV --json
miniverse challenge submit SET:CHALLENGE --name "My policy" \
  --bundle POLICY@BRV --json
```

Only ready, owned bundle revisions may be submitted. Validation checks command
IDs, kinds, shapes, units, frames, ranges, update semantics, and simulator
compatibility before any work is queued.

One bundle revision can appear only once in a given challenge revision. If a
client retries the same submission, Miniverse returns the existing submission
instead of queueing another evaluation.

To revise a submission, upload the new bundle revision and use its current
`latestRevisionNumber` as the optimistic concurrency value:

```bash
miniverse challenge update chsub_... --expected-revision 1 \
  --bundle POLICY@NEW-BRV --json
miniverse challenge status chsub_... --json
```

The old active revision remains active while the replacement evaluates. A
failed replacement leaves the old result untouched. Compute placement is an
operator decision. Submissions cannot select fleet or Modal execution.
