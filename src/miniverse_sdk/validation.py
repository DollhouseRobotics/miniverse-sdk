"""Authoritative local validation for public Miniverse bundle contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from .onnx_compat import CompatFinding, ModelScan, scan_model
from .onnx_metadata import ONNX_HASH_KEY, compatibility_report

MAX_MODEL_BYTES = 1024 * 1024 * 1024
MAX_TENSOR_ELEMENTS = 16 * 1024 * 1024
ALLOWED_ONNX_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}
SOURCE_TO_CONTRACT_BACKEND = {"mujoco": "mujoco-cpu"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class ModelValidation:
    precision: str | None
    contract: dict[str, Any] | None
    compatibility: dict[str, Any]
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "precision": self.precision,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
            "compatibility": self.compatibility,
        }


def _schema(name: str) -> dict[str, Any]:
    root = resources.files("miniverse_sdk.schemas")
    return json.loads(root.joinpath(name).read_text(encoding="utf-8"))


def schema_issues(value: Any, name: str, code: str) -> tuple[ValidationIssue, ...]:
    validator = Draft202012Validator(_schema(name))
    errors = sorted(validator.iter_errors(value), key=lambda error: tuple(str(part) for part in error.absolute_path))
    result = []
    for error in errors:
        path = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)
        result.append(ValidationIssue(code=code, severity="error", path=path, message=error.message))
    return tuple(result)


def validate_bundle_manifest(manifest: Any) -> tuple[ValidationIssue, ...]:
    return schema_issues(manifest, "simulation-bundle-v1.schema.json", "bundle_schema")


def validate_onnx_contract_schema(contract: Any) -> tuple[ValidationIssue, ...]:
    return schema_issues(contract, "onnx-simulation-contract-0.3.schema.json", "onnx_contract_schema")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _contract_hash(contract: Mapping[str, Any]) -> str:
    value = dict(contract)
    value.pop("contractHash", None)
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _shape(value: Any) -> tuple[int, ...]:
    dimensions: list[int] = []
    for dimension in value.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or int(dimension.dim_value) <= 0:
            raise ValueError(f"tensor {value.name} has a dynamic or unbounded shape")
        dimensions.append(int(dimension.dim_value))
    if not dimensions or math.prod(dimensions) > MAX_TENSOR_ELEMENTS:
        raise ValueError(f"tensor {value.name} exceeds the static shape bound")
    if dimensions[0] != 1:
        raise ValueError(f"tensor {value.name} must use fixed batch size one")
    return tuple(dimensions)


def _dtype(value: Any, onnx: Any) -> str:
    names = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.DOUBLE: "float64",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.BOOL: "bool",
    }
    result = names.get(int(value.type.tensor_type.elem_type))
    if result is None:
        raise ValueError(f"tensor {value.name} has an unsupported dtype")
    return result


def _finding_issue(finding: CompatFinding) -> ValidationIssue:
    return ValidationIssue(
        code=finding.code,
        severity=finding.severity,
        path=finding.node or None,
        message=finding.message,
    )


def _append_unique(target: list[ValidationIssue], issue: ValidationIssue) -> None:
    identity = (issue.code, issue.severity, issue.message, issue.path)
    if all((value.code, value.severity, value.message, value.path) != identity for value in target):
        target.append(issue)


def _cross_validate_contract(
    contract: Mapping[str, Any],
    model_inputs: Mapping[str, tuple[str, tuple[int, ...]]],
    model_outputs: Mapping[str, tuple[str, tuple[int, ...]]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    rates = contract.get("rates")
    if isinstance(rates, Mapping):
        try:
            physics = float(rates.get("physicsHz", 0))
            policy = float(rates.get("policyHz", 0))
            publish = float(rates.get("publishHz", 0))
            if not all(math.isfinite(value) and value > 0 for value in (physics, policy, publish)) or physics > 5000 or policy > 1000 or publish > 120:
                raise ValueError
            if policy > physics or publish > physics:
                issues.append(ValidationIssue("onnx_contract_rates", "error", "policyHz and publishHz cannot exceed physicsHz", "$.rates"))
            if rates.get("controlLoop", "independent-clocks") == "policy-then-decimation":
                decimation = physics / policy
                if not math.isclose(decimation, round(decimation), rel_tol=0.0, abs_tol=1e-12):
                    issues.append(ValidationIssue("onnx_contract_rates", "error", "policy-then-decimation requires an integer physicsHz/policyHz ratio", "$.rates.controlLoop"))
        except (TypeError, ValueError, ZeroDivisionError):
            issues.append(ValidationIssue("onnx_contract_rates", "error", "contract rates are outside the supported bounds", "$.rates"))

    contract_inputs = {str(value.get("name", "")): value for value in contract.get("inputs", ()) if isinstance(value, Mapping)}
    contract_outputs = {str(value.get("name", "")): value for value in contract.get("outputs", ()) if isinstance(value, Mapping)}
    if set(contract_inputs) != set(model_inputs):
        issues.append(ValidationIssue("onnx_input_mapping", "error", "contract inputs do not exactly match ONNX graph inputs", "$.inputs"))
    if set(contract_outputs) != set(model_outputs):
        issues.append(ValidationIssue("onnx_output_mapping", "error", "contract outputs do not exactly match ONNX graph outputs", "$.outputs"))

    state_inputs = {
        str(value.get("inputName", ""))
        for value in contract.get("stateBindings", ())
        if isinstance(value, Mapping)
    }
    for name, (dtype, shape) in model_inputs.items():
        declared = contract_inputs.get(name)
        if not isinstance(declared, Mapping):
            continue
        if declared.get("dtype") != dtype or tuple(declared.get("shape", ())) != shape:
            issues.append(ValidationIssue("onnx_input_mapping", "error", f"input mapping for {name} does not match ONNX dtype and shape", f"$.inputs.{name}"))
        slices = declared.get("slices", ())
        if name in state_inputs:
            if slices:
                issues.append(ValidationIssue("onnx_state_binding", "error", f"state input {name} must not declare observation slices", f"$.inputs.{name}.slices"))
            continue
        cursor = 0
        if not isinstance(slices, list):
            slices = []
        for index, value in enumerate(slices):
            if not isinstance(value, Mapping) or value.get("start") != cursor:
                issues.append(ValidationIssue("onnx_input_slices", "error", f"input {name} slices must be contiguous and ordered", f"$.inputs.{name}.slices[{index}]"))
                break
            cursor += int(value.get("length", 0))
        if cursor != math.prod(shape):
            issues.append(ValidationIssue("onnx_input_slices", "error", f"input {name} slices do not cover its tensor", f"$.inputs.{name}.slices"))

    actuator_outputs = 0
    output_roles: dict[str, str] = {}
    for name, (dtype, shape) in model_outputs.items():
        declared = contract_outputs.get(name)
        if not isinstance(declared, Mapping):
            continue
        if declared.get("dtype") != dtype or tuple(declared.get("shape", ())) != shape:
            issues.append(ValidationIssue("onnx_output_mapping", "error", f"output mapping for {name} does not match ONNX dtype and shape", f"$.outputs.{name}"))
        role = str(declared.get("role", ""))
        output_roles[name] = role
        if role == "actuatorTargets":
            actuator_outputs += 1
            elements = math.prod(shape)
            if len(declared.get("actuators", ())) != elements:
                issues.append(ValidationIssue("onnx_actuator_mapping", "error", f"output {name} actuator ordering does not cover its tensor", f"$.outputs.{name}.actuators"))
            modes = declared.get("controlModes")
            if isinstance(modes, list) and modes and len(modes) != elements:
                issues.append(ValidationIssue("onnx_actuator_mapping", "error", f"output {name} controlModes do not cover its tensor", f"$.outputs.{name}.controlModes"))
            ranges = declared.get("actuatorRanges")
            if isinstance(ranges, list) and ranges and len(ranges) != elements:
                issues.append(ValidationIssue("onnx_actuator_mapping", "error", f"output {name} actuatorRanges do not cover its tensor", f"$.outputs.{name}.actuatorRanges"))
            failsafe = declared.get("failsafe")
            if isinstance(failsafe, list) and len(failsafe) != elements:
                issues.append(ValidationIssue("onnx_actuator_mapping", "error", f"output {name} failsafe does not cover its tensor", f"$.outputs.{name}.failsafe"))
    if actuator_outputs != 1:
        issues.append(ValidationIssue("onnx_actuator_mapping", "error", "contract requires exactly one actuatorTargets output", "$.outputs"))

    seen_state_ids: set[str] = set()
    seen_state_inputs: set[str] = set()
    seen_state_outputs: set[str] = set()
    for index, binding in enumerate(contract.get("stateBindings", ())):
        if not isinstance(binding, Mapping):
            continue
        binding_id = str(binding.get("id", ""))
        input_name = str(binding.get("inputName", ""))
        output_name = str(binding.get("outputName", ""))
        if binding_id in seen_state_ids or input_name in seen_state_inputs or output_name in seen_state_outputs:
            issues.append(ValidationIssue("onnx_state_binding", "error", f"state binding {binding_id} is duplicate", f"$.stateBindings[{index}]"))
        elif input_name not in model_inputs or output_name not in model_outputs or model_inputs[input_name] != model_outputs[output_name] or output_roles.get(output_name) != "state":
            issues.append(ValidationIssue("onnx_state_binding", "error", f"state binding {binding_id} does not pair compatible state tensors", f"$.stateBindings[{index}]"))
        seen_state_ids.add(binding_id)
        seen_state_inputs.add(input_name)
        seen_state_outputs.add(output_name)

    if contract.get("execution") != {"kind": "singlePolicyStep"}:
        issues.append(ValidationIssue("onnx_execution", "error", "policy execution must be exactly one self-contained ONNX invocation", "$.execution"))
    return issues


def validate_model(path: Path) -> ModelValidation:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    contract: dict[str, Any] | None = None
    scanned: ModelScan | None = None
    compatibility: dict[str, Any] = {"declaredSimulators": [], "inputs": {}, "incompatible": []}
    precision: str | None = None
    contract_schema_valid = False

    if not path.is_file():
        return ModelValidation(None, None, compatibility, (ValidationIssue("model_missing", "error", f"model does not exist: {path}"),), ())
    if path.stat().st_size > MAX_MODEL_BYTES:
        return ModelValidation(None, None, compatibility, (ValidationIssue("model_size", "error", "ONNX model exceeds the 1 GiB validation bound"),), ())

    try:
        with path.open("rb") as source:
            scanned = scan_model(source)
        contract = scanned.contract
        precision = scanned.precision
        compatibility = compatibility_report(contract or {})
        for finding in scanned.findings:
            issue = _finding_issue(finding)
            _append_unique(errors if issue.severity == "error" else warnings, issue)
        for value in compatibility["incompatible"]:
            _append_unique(errors, ValidationIssue("simulator_incompatible", "error", f"operation is unavailable for a declared simulator: {value}", "$.backends"))
    except Exception as error:
        _append_unique(errors, ValidationIssue("invalid_model_metadata", "error", str(error)))

    if contract is not None:
        contract_schema_errors = validate_onnx_contract_schema(contract)
        contract_schema_valid = not contract_schema_errors
        for issue in contract_schema_errors:
            _append_unique(errors, issue)

    try:
        import onnx

        model = onnx.load(str(path), load_external_data=False)
        if any(tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data for tensor in model.graph.initializer):
            _append_unique(errors, ValidationIssue("onnx_external_data", "error", "external ONNX tensor data is forbidden"))
        if any(node.domain not in ALLOWED_ONNX_DOMAINS for node in model.graph.node):
            _append_unique(errors, ValidationIssue("onnx_custom_domain", "error", "ONNX contains a custom operator domain"))
        opsets = {value.domain or "ai.onnx": int(value.version) for value in model.opset_import}
        if any((value.domain or "") not in ALLOWED_ONNX_DOMAINS or int(value.version) < 13 or int(value.version) > 21 for value in model.opset_import):
            _append_unique(errors, ValidationIssue("onnx_opset", "error", "ONNX opset is outside the supported range 13-21"))
        try:
            onnx.checker.check_model(model, full_check=True)
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        except Exception as error:
            _append_unique(errors, ValidationIssue("onnx_graph", "error", f"ONNX graph validation failed: {error}"))
            inferred = model

        initializer_names = {value.name for value in inferred.graph.initializer}
        model_inputs: dict[str, tuple[str, tuple[int, ...]]] = {}
        model_outputs: dict[str, tuple[str, tuple[int, ...]]] = {}
        for value in inferred.graph.input:
            if value.name in initializer_names:
                continue
            try:
                model_inputs[value.name] = (_dtype(value, onnx), _shape(value))
            except ValueError as error:
                _append_unique(errors, ValidationIssue("onnx_tensor", "error", str(error), value.name))
        for value in inferred.graph.output:
            try:
                model_outputs[value.name] = (_dtype(value, onnx), _shape(value))
            except ValueError as error:
                _append_unique(errors, ValidationIssue("onnx_tensor", "error", str(error), value.name))

        metadata_entries = [(value.key, value.value) for value in inferred.metadata_props]
        metadata = dict(metadata_entries)
        if len(metadata) != len(metadata_entries):
            _append_unique(errors, ValidationIssue("onnx_metadata_duplicate", "error", "ONNX metadata contains duplicate keys"))
        if contract is not None:
            clone = type(model)()
            clone.CopyFrom(model)
            del clone.metadata_props[:]
            graph_hash = hashlib.sha256(clone.SerializeToString(deterministic=True)).hexdigest()
            expected_contract_hash = _contract_hash(contract)
            if contract.get("contractHash") != expected_contract_hash:
                _append_unique(errors, ValidationIssue("onnx_contract_hash", "error", "ONNX contractHash does not match canonical metadata", "$.contractHash"))
            if metadata.get(ONNX_HASH_KEY) != expected_contract_hash:
                _append_unique(errors, ValidationIssue("onnx_contract_hash", "error", "ONNX metadata contract hash disagrees with the canonical contract"))
            if contract.get("modelSha256") != graph_hash:
                _append_unique(errors, ValidationIssue("onnx_model_hash", "error", "ONNX contract modelSha256 disagrees with the executable graph", "$.modelSha256"))
            if contract.get("opset") != opsets.get("ai.onnx"):
                _append_unique(errors, ValidationIssue("onnx_opset", "error", "contract opset does not match the ONNX graph", "$.opset"))
            if contract_schema_valid:
                for issue in _cross_validate_contract(contract, model_inputs, model_outputs):
                    _append_unique(errors, issue)
    except Exception as error:
        _append_unique(errors, ValidationIssue("invalid_onnx", "error", f"could not validate ONNX graph: {error}"))

    return ModelValidation(precision, contract, compatibility, tuple(errors), tuple(warnings))


def validate_bundle_model_backends(manifest: Mapping[str, Any], models: Mapping[str, ModelValidation]) -> tuple[ValidationIssue, ...]:
    requested = [manifest.get("primarySimulator"), *manifest.get("compatibleSimulators", ())]
    expected = {SOURCE_TO_CONTRACT_BACKEND.get(str(value), str(value)) for value in requested}
    issues: list[ValidationIssue] = []
    for model_id, validation in models.items():
        contract = validation.contract or {}
        declared = {str(value.get("id", "")) for value in contract.get("backends", ()) if isinstance(value, Mapping)}
        missing = sorted(expected - declared)
        if missing:
            issues.append(ValidationIssue("bundle_model_backend", "error", f"model {model_id} does not declare bundle simulator support for: {', '.join(missing)}", f"$.models.{model_id}.backends"))
    return tuple(issues)
