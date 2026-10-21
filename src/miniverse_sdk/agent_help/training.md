# Training runs

Install scalar metric support with `pip install 'miniverse-sdk[training]'`.
Training runs belong to an organization. Choose an organization you belong to
and a ready source bundle revision you manage. Miniverse creates a dedicated
private base for the run, owned by that organization. The base and checkpoint
Miniverses appear only through the run and are accessible only to its members.
The source bundle keeps its existing ownership and visibility.

Create a run with a stable idempotency key, then retain the returned run ID:

```bash
miniverse training create --name locomotion --project experiments \
  --organization ORGANIZATION_ID --base BUNDLE_ID@REVISION_ID \
  --idempotency-key TRAINING_JOB_ID --json
```

Log scalar Optics-compatible metrics from JSON Lines. Each line is either
`{"step":1,"metrics":{"train/loss":0.4}}` or
`{"step":1,"name":"train/loss","value":0.4}`.

```bash
miniverse training log-rrd RUN_ID --file metrics.jsonl --json
```

The SDK journals samples and their sequence numbers under the user state
directory (or `MINIVERSE_TRAINING_SPOOL`). Reusing the run ID resumes that
journal. A flush advances its acknowledgement only after Miniverse reports a
durable commit. In Python, call `run.flush()` again after an interruption to
retry the same sealed layer.

Flushed metrics appear on the run page within seconds. Flush as often as you
want them live, for example every training iteration or every 10–30 seconds.
Each flush uploads one small layer, and Miniverse compacts layers in the
background, so frequent flushes do not hit a layer limit. Do not batch metrics
into long intervals to save layers.

Upload one or more named ONNX files at a checkpoint without rebuilding or
uploading the base bundle:

```bash
miniverse training checkpoint RUN_ID --step 10000 --label candidate \
  --model actor=actor.onnx --model critic=critic.onnx \
  --idempotency-key TRAINING_JOB_ID:10000 --json
```

Inspect and update runs, or attach a retained Miniverse recording:

```bash
miniverse training list --json
miniverse training get RUN_ID --json
miniverse training update RUN_ID --status completed --expected-version VERSION --json
miniverse training attach RUN_ID CHECKPOINT_ID RECORDING_ID --json
```

Python training loops use the same durable spool:

```python
from miniverse_sdk.api import Client

runs = Client(origin, token).training()
run = runs.create("locomotion", "bundle-id", "revision-id",
                  organization_id="organization-id",
                  idempotency_key="training-job-id")
run.log({"train/loss": 0.4, "train/reward": 12.0}, step=1)
run.flush()
run.checkpoint(10000, "candidate", {"actor": "actor.onnx", "critic": "critic.onnx"},
               idempotency_key="training-job-id:10000")
run.finish()
```
