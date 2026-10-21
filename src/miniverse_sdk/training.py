"""Resumable ingestion for Miniverse training runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .api import ApiError, Client

APPLICATION_ID = "optics"
RRD_CONTRACT_VERSION = 1
RERUN_VERSION = "0.36.0"


def _quoted(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, separators=(",", ":"), sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _default_spool_dir() -> Path:
    configured = os.environ.get("MINIVERSE_TRAINING_SPOOL")
    if configured:
        return Path(configured).expanduser()
    state = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state / "miniverse" / "training"


@dataclass(frozen=True)
class RrdLayer:
    id: str
    kind: str
    path: Path
    sha256: str
    bytes: int
    recording_id: str
    source: str
    metadata: dict[str, Any]

    def reservation(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "applicationId": APPLICATION_ID,
            "recordingId": self.recording_id,
            "contractVersion": RRD_CONTRACT_VERSION,
            "rerunVersion": RERUN_VERSION,
            "source": self.source,
            "metadata": self.metadata,
        }


def _entity_and_component(key: str) -> tuple[str, str]:
    clean = key.strip("/")
    if not clean:
        raise ValueError("metric names cannot be empty")
    pieces = clean.split("/")
    return f"/metrics/{'/'.join(pieces[:-1]) or 'default'}", pieces[-1]


def write_metric_layer(
    run_id: str,
    spool_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> RrdLayer:
    """Write Optics-compatible scalar metrics to one immutable RRD layer."""
    try:
        import pyarrow as pa
        import rerun as rr
    except ImportError as error:
        raise RuntimeError("training metrics require `pip install 'miniverse-sdk[training]'`") from error

    spool_dir.mkdir(parents=True, exist_ok=True)
    temporary = spool_dir / f".metrics-{uuid.uuid4().hex}.partial.rrd"
    recording = rr.RecordingStream(APPLICATION_ID, recording_id=run_id)
    recording.save(temporary, write_footer=True)
    metric_names: set[str] = set()
    try:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            name = str(row["name"])
            entity, component = _entity_and_component(name)
            grouped.setdefault(entity, []).append({**row, "component": component})
            metric_names.add(name)
        for entity, values in grouped.items():
            by_step: dict[int, dict[str, Any]] = {}
            for row_index, value in enumerate(values):
                step = int(value["step"])
                sequence = int(value["sampleSequence"])
                row = by_step.setdefault(step, {"step": step, "sampleSequence": sequence, "wallTime": 0.0, "components": {}})
                row["sampleSequence"] = max(int(row["sampleSequence"]), sequence)
                row["wallTime"] = max(float(row["wallTime"]), float(value["wallTime"]))
                component = str(value["component"])
                existing = row["components"].get(component)
                order = (sequence, row_index)
                if existing is None or order >= existing[0]:
                    row["components"][component] = (order, float(value["value"]))
            temporal_rows = [by_step[step] for step in sorted(by_step)]
            indexes = [
                rr.TimeColumn("step", sequence=[int(row["step"]) for row in temporal_rows]),
                rr.TimeColumn("sample_sequence", sequence=[int(row["sampleSequence"]) for row in temporal_rows]),
                rr.TimeColumn("wall_time", timestamp=[float(row["wallTime"]) for row in temporal_rows]),
            ]
            names = sorted({str(value["component"]) for value in values})
            columns = {
                name: pa.array(
                    [[row["components"][name][1]] if name in row["components"] else None for row in temporal_rows],
                    type=pa.list_(pa.float64()),
                )
                for name in names
            }
            recording.send_columns(entity, indexes=indexes, columns=rr.AnyValues.columns(**columns), strict=True)
        recording.flush()
        recording.disconnect()
    except Exception:
        recording.disconnect()
        temporary.unlink(missing_ok=True)
        raise

    digest, size = _sha256(temporary)
    layer_id = f"metrics-{digest[:24]}"
    final = spool_dir / f"{layer_id}.rrd"
    if final.exists():
        temporary.unlink()
    else:
        os.replace(temporary, final)
    return RrdLayer(
        id=layer_id,
        kind="metrics",
        path=final,
        sha256=digest,
        bytes=size,
        recording_id=run_id,
        source=source,
        metadata={"metricNames": sorted(metric_names), "sampleCount": len(rows)},
    )


class TrainingRun:
    """A stable run handle with a durable local metric journal."""

    def __init__(
        self,
        client: Client,
        run_id: str,
        spool_dir: str | Path | None = None,
        *,
        writer: Callable[..., RrdLayer] = write_metric_layer,
    ) -> None:
        self.client = client
        self.id = run_id
        self.root = Path(spool_dir) if spool_dir is not None else _default_spool_dir()
        self.directory = self.root / run_id
        self.layers = self.directory / "layers"
        self.journal_path = self.directory / "journal.json"
        self._writer = writer
        if self.journal_path.exists():
            self._journal = json.loads(self.journal_path.read_text(encoding="utf-8"))
            if self._journal.get("runId") != run_id:
                raise ValueError("training journal belongs to a different run")
        else:
            self._journal = {
                "runId": run_id,
                "nextSampleSequence": 1,
                "ackedSampleSequence": 0,
                "samples": [],
                "sealedLayers": [],
                "pendingLayer": None,
            }
            self._save()

    @property
    def sample_sequence(self) -> int:
        return int(self._journal["nextSampleSequence"]) - 1

    @property
    def acked_sample_sequence(self) -> int:
        return int(self._journal["ackedSampleSequence"])

    def _save(self) -> None:
        _atomic_json(self.journal_path, self._journal)

    def log(
        self,
        metrics: Mapping[str, float] | str,
        value: float | None = None,
        *,
        step: int,
        wall_time: float | None = None,
    ) -> int:
        values = {metrics: value} if isinstance(metrics, str) else dict(metrics)
        if not values:
            raise ValueError("at least one metric is required")
        sequence = int(self._journal["nextSampleSequence"])
        timestamp = time.time() if wall_time is None else float(wall_time)
        for name, scalar in values.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric names must be nonempty strings")
            if scalar is None:
                raise ValueError("metric values must be scalar numbers")
            self._journal["samples"].append({
                "name": name,
                "value": float(scalar),
                "step": int(step),
                "wallTime": timestamp,
                "sampleSequence": sequence,
            })
        self._journal["nextSampleSequence"] = sequence + 1
        self._save()
        return sequence

    def _pending_layer(self) -> RrdLayer | None:
        value = self._journal.get("pendingLayer")
        if not isinstance(value, dict):
            return None
        return RrdLayer(
            id=str(value["id"]), kind=str(value["kind"]), path=Path(value["path"]),
            sha256=str(value["sha256"]), bytes=int(value["bytes"]),
            recording_id=self.id, source=str(value["source"]), metadata=dict(value["metadata"]),
        )

    def flush(self) -> dict[str, Any]:
        layer = self._pending_layer()
        if layer is None:
            rows = [row for row in self._journal["samples"] if int(row["sampleSequence"]) > self.acked_sample_sequence]
            if not rows:
                return {"durableCommit": True, "ackedSampleSequence": self.acked_sample_sequence, "empty": True}
            layer = self._writer(self.id, self.layers, rows, source="miniverse-sdk")
            through = max(int(row["sampleSequence"]) for row in rows)
            self._journal["pendingLayer"] = {**asdict(layer), "path": str(layer.path), "throughSampleSequence": through}
            self._save()
        else:
            through = int(self._journal["pendingLayer"]["throughSampleSequence"])

        base = f"/api/v1/training-runs/{_quoted(self.id)}/rrd/layers"
        reserved = self.client.request(base, layer.reservation())
        transfer = reserved.get("upload")
        if "upload" not in reserved:
            raise ApiError(502, "server returned invalid RRD upload instructions", "training_contract_error")
        if transfer is not None:
            if not isinstance(transfer, dict) or transfer.get("method") != "PUT" or not isinstance(transfer.get("url"), str):
                raise ApiError(502, "server returned invalid RRD upload instructions", "training_contract_error")
            headers = transfer.get("headers", {})
            if not isinstance(headers, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
                raise ApiError(502, "server returned invalid RRD upload headers", "training_contract_error")
            self.client.upload(transfer["url"], layer.path, headers, content_type="application/octet-stream")
        layer_id = reserved.get("layerId", layer.id)
        finalized = self.client.request(f"{base}/{_quoted(str(layer_id))}/finalize", {})
        if finalized.get("durableCommit") is not True:
            raise ApiError(502, "RRD layer was not durably committed", "training_commit_pending")

        self._journal["ackedSampleSequence"] = through
        self._journal["samples"] = [row for row in self._journal["samples"] if int(row["sampleSequence"]) > through]
        self._journal["sealedLayers"].append({**asdict(layer), "path": str(layer.path), "throughSampleSequence": through})
        self._journal["pendingLayer"] = None
        self._save()
        return {**finalized, "layerId": layer_id, "ackedSampleSequence": through}

    def checkpoint(
        self,
        step: int,
        label: str,
        models: Mapping[str, str | Path],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not models:
            raise ValueError("at least one checkpoint model is required")
        files = {model_id: Path(path) for model_id, path in models.items()}
        descriptors = []
        for model_id, path in files.items():
            digest, size = _sha256(path)
            descriptors.append({"modelId": model_id, "sha256": digest, "bytes": size})
        retry_identity = hashlib.sha256(json.dumps(
            {"runId": self.id, "step": int(step), "label": label, "models": descriptors},
            separators=(",", ":"), sort_keys=True,
        ).encode()).hexdigest()
        base = f"/api/v1/training-runs/{_quoted(self.id)}/checkpoints"
        prepared = self.client.request(base, {
            "step": int(step), "label": label, "models": descriptors,
            "idempotencyKey": idempotency_key or f"sdk:{retry_identity}",
        })
        uploads = prepared.get("uploads")
        if not isinstance(uploads, list):
            raise ApiError(502, "server returned invalid checkpoint uploads", "training_contract_error")
        for upload in uploads:
            if not isinstance(upload, dict) or upload.get("method") != "PUT" or upload.get("modelId") not in files or not isinstance(upload.get("url"), str):
                raise ApiError(502, "server returned invalid checkpoint upload instructions", "training_contract_error")
            headers = upload.get("headers", {})
            if not isinstance(headers, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in headers.items()):
                raise ApiError(502, "server returned invalid checkpoint upload headers", "training_contract_error")
            self.client.upload(upload["url"], files[upload["modelId"]], headers, content_type="application/octet-stream")
        checkpoint = prepared.get("checkpoint")
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("id"), str):
            raise ApiError(502, "server returned an invalid checkpoint", "training_contract_error")
        return self.client.request(f"{base}/{_quoted(checkpoint['id'])}/finalize", {})

    def attach_recording(self, checkpoint_id: str, recording_id: str) -> dict[str, Any]:
        path = f"/api/v1/training-runs/{_quoted(self.id)}/checkpoints/{_quoted(checkpoint_id)}/recordings"
        return self.client.request(path, {"recordingId": recording_id})

    def finish(self, status: str = "completed") -> dict[str, Any]:
        self.flush()
        return self.update(status=status)

    def get(self, *, cursor: str | None = None) -> dict[str, Any]:
        query = f"?cursor={urllib.parse.quote(cursor)}" if cursor else ""
        return self.client.request(f"/api/v1/training-runs/{_quoted(self.id)}{query}")

    def update(self, **changes: Any) -> dict[str, Any]:
        if "expectedVersion" not in changes:
            changes["expectedVersion"] = self.get()["run"]["version"]
        return self.client.request(f"/api/v1/training-runs/{_quoted(self.id)}", changes, method="PATCH")


class Training:
    """Training-run collection attached to an authenticated API client."""

    def __init__(self, client: Client, spool_dir: str | Path | None = None) -> None:
        self.client = client
        self.spool_dir = spool_dir

    def create(
        self,
        name: str,
        base_bundle_id: str,
        base_revision_id: str,
        *,
        organization_id: str,
        project: str = "",
        config: Mapping[str, Any] | None = None,
        tags: Iterable[str] = (),
        idempotency_key: str | None = None,
    ) -> TrainingRun:
        request = {
            "name": name, "organizationId": organization_id, "project": project, "baseBundleId": base_bundle_id,
            "baseRevisionId": base_revision_id, "config": dict(config or {}),
            "tags": list(tags),
        }
        retry_identity = hashlib.sha256(json.dumps(request, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
        value = self.client.request("/api/v1/training-runs", {
            **request, "idempotencyKey": idempotency_key or f"sdk:{retry_identity}",
        })
        run = value.get("run", value)
        if not isinstance(run, dict) or not isinstance(run.get("id"), str):
            raise ApiError(502, "server returned an invalid training run", "training_contract_error")
        return self.resume(run["id"])

    def resume(self, run_id: str) -> TrainingRun:
        return TrainingRun(self.client, run_id, self.spool_dir)

    def list(self, *, cursor: str | None = None) -> dict[str, Any]:
        query = f"?cursor={urllib.parse.quote(cursor)}" if cursor else ""
        return self.client.request(f"/api/v1/training-runs{query}")
