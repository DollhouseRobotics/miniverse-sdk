from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from miniverse_sdk.api import ApiError
from miniverse_sdk.training import RrdLayer, Training, TrainingRun, write_metric_layer


class FakeClient:
    def __init__(self, durable_commits: list[bool] | None = None) -> None:
        self.calls = []
        self.uploads = []
        self.durable_commits = list(durable_commits or [True])

    def request(self, path, value=None, method=None):
        self.calls.append((path, value, method))
        if path.endswith("/rrd/layers"):
            return {"layerId": value["id"], "upload": {"url": "https://upload.invalid/rrd", "method": "PUT", "headers": {}}}
        if path.endswith("/finalize") and "/rrd/layers/" in path:
            return {"durableCommit": self.durable_commits.pop(0)}
        if path.endswith("/checkpoints"):
            return {
                "checkpoint": {"id": "tck_1"},
                "uploads": [
                    {"modelId": model["modelId"], "url": f"https://upload.invalid/{model['modelId']}", "method": "PUT", "headers": {}}
                    for model in value["models"]
                ],
            }
        if path.endswith("/checkpoints/tck_1/finalize"):
            return {"checkpoint": {"id": "tck_1", "state": "deriving"}}
        return {"ok": True}

    def upload(self, url, path, required_headers=None, *, content_type=None):
        self.uploads.append((url, Path(path), required_headers, content_type))


def fake_writer(run_id, spool_dir, rows, *, source):
    path = spool_dir / "metrics-test.rrd"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"rrd")
    return RrdLayer(
        id="metrics-test", kind="metrics", path=path,
        sha256=hashlib.sha256(b"rrd").hexdigest(), bytes=3,
        recording_id=run_id, source=source,
        metadata={"metricNames": sorted({row["name"] for row in rows})},
    )


class TrainingIngestionTest(unittest.TestCase):
    def test_create_requires_and_sends_organization_in_retry_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient()
            original_request = client.request

            def request(path, value=None, method=None):
                original_request(path, value, method)
                return {"run": {"id": "trn_test"}}

            client.request = request
            training = Training(client, directory)
            with self.assertRaises(TypeError):
                training.create("PPO", "robot", "revision")
            training.create("PPO", "robot", "revision", organization_id="lab-a")
            training.create("PPO", "robot", "revision", organization_id="lab-b")
            first, second = [value for _, value, _ in client.calls]
            self.assertEqual(first["organizationId"], "lab-a")
            self.assertEqual(second["organizationId"], "lab-b")
            self.assertNotEqual(first["idempotencyKey"], second["idempotencyKey"])

    def test_flush_resumes_sealed_layer_and_advances_ack_only_after_durable_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient([False, True])
            run = TrainingRun(client, "trn_test", directory, writer=fake_writer)
            self.assertEqual(run.log({"train/loss": 0.5, "train/reward": 2}, step=7), 1)
            with self.assertRaises(ApiError):
                run.flush()
            self.assertEqual(run.acked_sample_sequence, 0)

            def must_not_rewrite(*args, **kwargs):
                raise AssertionError("a sealed pending layer must be reused")

            resumed = TrainingRun(client, "trn_test", directory, writer=must_not_rewrite)
            result = resumed.flush()
            self.assertEqual(result["ackedSampleSequence"], 1)
            self.assertEqual(resumed.acked_sample_sequence, 1)
            reservations = [value for path, value, _ in client.calls if path.endswith("/rrd/layers")]
            self.assertEqual([value["id"] for value in reservations], ["metrics-test", "metrics-test"])
            self.assertEqual(reservations[0]["recordingId"], "trn_test")
            self.assertEqual(reservations[0]["rerunVersion"], "0.36.0")

    def test_lost_commit_acknowledgement_reuses_uploaded_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeClient([False, True])
            run = TrainingRun(client, "trn_test", directory, writer=fake_writer)
            run.log("loss", 3, step=11)
            with self.assertRaises(ApiError):
                run.flush()
            original_request = client.request

            def request(path, value=None, method=None):
                if path.endswith("/rrd/layers"):
                    return {"layerId": value["id"], "upload": None}
                return original_request(path, value, method)

            client.request = request
            self.assertEqual(run.flush()["ackedSampleSequence"], 1)
            self.assertEqual(len(client.uploads), 1)

    def test_checkpoint_hashes_and_uploads_multiple_named_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "actor.onnx"
            critic = root / "critic.onnx"
            actor.write_bytes(b"actor")
            critic.write_bytes(b"critic")
            client = FakeClient()
            run = TrainingRun(client, "trn_test", root / "spool", writer=fake_writer)
            result = run.checkpoint(20, "candidate", {"actor": actor, "critic": critic})
            self.assertEqual(result["checkpoint"]["id"], "tck_1")
            reservation = next(value for path, value, _ in client.calls if path.endswith("/checkpoints"))
            self.assertEqual([model["modelId"] for model in reservation["models"]], ["actor", "critic"])
            self.assertEqual(reservation["models"][0]["sha256"], hashlib.sha256(b"actor").hexdigest())
            self.assertEqual(len(client.uploads), 2)
            self.assertTrue(reservation["idempotencyKey"].startswith("sdk:"))

    def test_real_writer_is_optics_compatible_when_training_extra_is_installed(self):
        try:
            import pyarrow  # noqa: F401
            import rerun  # noqa: F401
        except ImportError:
            self.skipTest("training extra is not installed")
        with tempfile.TemporaryDirectory() as directory:
            layer = write_metric_layer("trn_test", Path(directory), [
                {"name": "train/loss", "value": 0.5, "step": 1, "wallTime": 1.0, "sampleSequence": 1},
                {"name": "train/reward", "value": 2.0, "step": 1, "wallTime": 1.0, "sampleSequence": 1},
            ], source="test")
            self.assertTrue(layer.path.is_file())
            self.assertGreater(layer.bytes, 0)
            self.assertEqual(layer.id, f"metrics-{layer.sha256[:24]}")
            self.assertEqual(layer.metadata["metricNames"], ["train/loss", "train/reward"])


if __name__ == "__main__":
    unittest.main()
