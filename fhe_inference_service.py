import json
import logging
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openfhe import BINARY, DeserializeCiphertext, Serialize, SerializeToFile

from fhe_encrypt_service import ENCRYPTED_DIR
from fhe_key_load import KeyLoadError, load_inference_context

logger = logging.getLogger("fhe_vault")

RESULTS_DIR = Path(os.environ.get("FHE_ENCRYPTED_RESULTS_DIR", "/data/fhe-encrypted-results"))


def _preview_floats(values: list[float], *, limit: int = 5) -> str:
    if not values:
        return "[]"
    if len(values) <= limit:
        return str([round(v, 6) for v in values])
    head = [round(v, 6) for v in values[:limit]]
    return f"{head} … (+{len(values) - limit} more)"


def _preview_names(names: list[str], *, limit: int = 5) -> str:
    if not names:
        return "[]"
    if len(names) <= limit:
        return str(names)
    return f"{names[:limit]} … (+{len(names) - limit} more)"


@dataclass
class LogisticModelParams:
    intercept: float
    weights: list[float]
    feature_names: list[str]
    classes: list[str]
    threshold: float


@dataclass
class InferencePlan:
    slots: int
    params_count: int
    rows_per_ciphertext: int
    total_rows: int
    ciphertext_count: int
    columns: list[str]


@dataclass
class InferenceResult:
    result_id: str
    output_dir: Path
    result_files: list[str]
    manifest_file: str
    manifest: dict


def parse_logistic_model(model_json: dict[str, Any]) -> LogisticModelParams:
    if not isinstance(model_json, dict):
        raise ValueError("model_json must be an object")

    features = model_json.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Logistic model must include a non-empty features array")

    feature_names: list[str] = []
    weights: list[float] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            raise ValueError(f"Feature at index {index - 1} must be an object")
        name = feature.get("name")
        weight = feature.get("weight")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Feature at index {index - 1} is missing a name")
        if not isinstance(weight, (int, float)):
            raise ValueError(f"Feature '{name}' is missing a numeric weight")
        feature_names.append(name)
        weights.append(float(weight))

    classes = model_json.get("classes")
    class_names = (
        [value for value in classes if isinstance(value, str)]
        if isinstance(classes, list)
        else []
    )

    intercept = model_json.get("intercept")
    threshold = model_json.get("threshold")

    params = LogisticModelParams(
        intercept=float(intercept) if isinstance(intercept, (int, float)) else 0.0,
        weights=weights,
        feature_names=feature_names,
        classes=class_names,
        threshold=float(threshold) if isinstance(threshold, (int, float)) else 0.5,
    )
    logger.info(
        "[inference] parsed logistic model: features=%s intercept=%s threshold=%s "
        "classes=%s weights=%s feature_names=%s",
        len(params.weights),
        params.intercept,
        params.threshold,
        params.classes,
        _preview_floats(params.weights),
        _preview_names(params.feature_names),
    )
    return params


def align_weights_to_columns(
    *,
    columns: list[str],
    model: LogisticModelParams,
) -> tuple[list[float], list[dict[str, Any]]]:
    weight_by_name = dict(zip(model.feature_names, model.weights, strict=False))
    aligned_weights: list[float] = []
    mapping: list[dict[str, Any]] = []

    for column in columns:
        if column not in weight_by_name:
            raise ValueError(
                f"Model is missing a weight for dataset column '{column}'. "
                f"Model features={model.feature_names}"
            )
        weight = weight_by_name[column]
        aligned_weights.append(weight)
        mapping.append({"column": column, "feature": column, "weight": weight})

    if len(model.feature_names) != len(columns):
        unused = [name for name in model.feature_names if name not in columns]
        if unused:
            raise ValueError(
                f"Model has features not present in encrypted dataset columns: {unused}"
            )

    logger.info(
        "[inference] aligned weights to dataset columns: column_count=%s "
        "weights=%s mapping_sample=%s",
        len(columns),
        _preview_floats(aligned_weights),
        mapping[:3],
    )
    return aligned_weights, mapping


def build_inference_plan(
    *,
    slots: int,
    params_count: int,
    total_rows: int,
    ciphertext_count: int,
    columns: list[str],
) -> InferencePlan:
    if params_count <= 0:
        raise ValueError("params_count must be greater than 0")
    if slots <= 0:
        raise ValueError("slots must be greater than 0")

    rows_per_ciphertext = slots // params_count
    if rows_per_ciphertext <= 0:
        raise ValueError(
            f"params_count ({params_count}) exceeds available slots ({slots})"
        )

    plan = InferencePlan(
        slots=slots,
        params_count=params_count,
        rows_per_ciphertext=rows_per_ciphertext,
        total_rows=total_rows,
        ciphertext_count=ciphertext_count,
        columns=columns,
    )
    logger.info(
        "[inference] built inference plan: slots=%s params_count=%s "
        "rows_per_ciphertext=%s total_rows=%s ciphertext_count=%s columns=%s",
        plan.slots,
        plan.params_count,
        plan.rows_per_ciphertext,
        plan.total_rows,
        plan.ciphertext_count,
        _preview_names(plan.columns),
    )
    return plan


def _pack_repeated_row_values(
    row_values: list[float],
    *,
    params_count: int,
    slots: int,
) -> list[float]:
    if len(row_values) != params_count:
        raise ValueError(
            f"Expected {params_count} row values, got {len(row_values)}"
        )

    rows_per_ciphertext = slots // params_count
    packed: list[float] = []
    for _ in range(rows_per_ciphertext):
        packed.extend(row_values)

    if len(packed) > slots:
        raise ValueError(
            f"Packed values ({len(packed)}) exceed available slots ({slots})"
        )

    packed.extend([0.0] * (slots - len(packed)))
    logger.debug(
        "[inference] packed repeated row values: params_count=%s rows_per_ciphertext=%s "
        "nonzero_slots=%s preview=%s",
        params_count,
        rows_per_ciphertext,
        len(row_values) * rows_per_ciphertext,
        _preview_floats(packed[: params_count * min(3, rows_per_ciphertext)]),
    )
    return packed


def _pack_first_slot_values(
    values_per_row: list[float],
    *,
    params_count: int,
    slots: int,
) -> list[float]:
    rows_per_ciphertext = slots // params_count
    if len(values_per_row) != rows_per_ciphertext:
        raise ValueError(
            f"Expected {rows_per_ciphertext} per-row values, got {len(values_per_row)}"
        )

    packed = [0.0] * slots
    for row_index, value in enumerate(values_per_row):
        packed[row_index * params_count] = value
    return packed


def _pack_intercept_plaintext(intercept: float, plan: InferencePlan):
    values_per_row = [intercept] * plan.rows_per_ciphertext
    return _pack_first_slot_values(
        values_per_row,
        params_count=plan.params_count,
        slots=plan.slots,
    )


def _read_ciphertext(path: Path):
    logger.info(
        "[inference] deserializing input ciphertext: path=%s size_bytes=%s",
        path,
        path.stat().st_size if path.exists() else "missing",
    )
    ciphertext, ok = DeserializeCiphertext(str(path), BINARY)
    if not ok:
        raise ValueError(f"Failed to deserialize ciphertext from {path}")
    logger.info("[inference] input ciphertext deserialized OK: path=%s", path)
    return ciphertext


def _write_ciphertext(path: Path, ciphertext) -> None:
    if SerializeToFile(str(path), ciphertext, BINARY):
        logger.info(
            "[inference] wrote result ciphertext via SerializeToFile: path=%s size_bytes=%s",
            path,
            path.stat().st_size if path.exists() else "unknown",
        )
        return
    path.write_bytes(Serialize(ciphertext, BINARY))
    logger.info(
        "[inference] wrote result ciphertext via Serialize: path=%s size_bytes=%s",
        path,
        path.stat().st_size if path.exists() else "unknown",
    )


def compute_batched_linear_scores(
    cc,
    ciphertext,
    *,
    weights: list[float],
    intercept: float,
    plan: InferencePlan,
):
    """
    Mirror encryption packing: each row uses params_count consecutive slots.
    Linear score for row k is stored at slot k * params_count.
    """
    logger.info(
        "[inference] compute_batched_linear_scores: params_count=%s intercept=%s "
        "weights=%s batch_size=%s",
        plan.params_count,
        intercept,
        _preview_floats(weights),
        plan.params_count,
    )

    packed_weights = _pack_repeated_row_values(
        weights,
        params_count=plan.params_count,
        slots=plan.slots,
    )
    logger.info("[inference] step EvalInnerProduct: creating plaintext weights vector")
    pt_weights = cc.MakeCKKSPackedPlaintext(packed_weights)
    logger.info(
        "[inference] step EvalInnerProduct: running ct * pt_weights "
        "(batch_size=%s)",
        plan.params_count,
    )
    weighted = cc.EvalInnerProduct(ciphertext, pt_weights, plan.params_count)
    logger.info("[inference] step EvalInnerProduct: complete")

    packed_intercept = _pack_intercept_plaintext(intercept, plan)
    logger.info(
        "[inference] step EvalAdd intercept: intercept=%s rows_per_ciphertext=%s",
        intercept,
        plan.rows_per_ciphertext,
    )
    pt_intercept = cc.MakeCKKSPackedPlaintext(packed_intercept)
    result = cc.EvalAdd(weighted, pt_intercept)
    logger.info("[inference] compute_batched_linear_scores: complete")
    return result


def run_inference(
    *,
    dataset_id: int,
    encrypt_id: str,
    fhe_key_id: int,
    fhe_key_storage_path: str,
    dataset_model_id: int,
    dataset_model_name: str,
    dataset_model_type: str,
    inference_model_id: int,
    inference_model_name: str,
    inference_model_type: str,
    columns: list[str],
    ciphertext_files: list[str],
    slots: int,
    params_count: int,
    rows_per_ciphertext: int,
    total_rows: int,
    ciphertext_count: int,
    model_json: dict[str, Any],
) -> InferenceResult:
    logger.info(
        "[inference] run_inference start: dataset_id=%s encrypt_id=%s "
        "inference_model_id=%s inference_model_type=%s fhe_key_id=%s "
        "fhe_key_storage_path=%s slots=%s params_count=%s rows_per_ciphertext=%s "
        "total_rows=%s ciphertext_count=%s ciphertext_files=%s",
        dataset_id,
        encrypt_id,
        inference_model_id,
        inference_model_type,
        fhe_key_id,
        fhe_key_storage_path,
        slots,
        params_count,
        rows_per_ciphertext,
        total_rows,
        ciphertext_count,
        ciphertext_files,
    )

    if inference_model_type != "logistic":
        raise ValueError(
            f"Only logistic models are supported for inference, got '{inference_model_type}'"
        )

    logger.info("[inference] step 1/8: parse logistic model JSON")
    model = parse_logistic_model(model_json)
    if len(model.weights) != params_count:
        raise ValueError(
            f"Model params_count ({len(model.weights)}) does not match encrypted dataset "
            f"params_count ({params_count})"
        )

    logger.info("[inference] step 2/8: build inference plan")
    plan = build_inference_plan(
        slots=slots,
        params_count=params_count,
        total_rows=total_rows,
        ciphertext_count=ciphertext_count,
        columns=columns,
    )
    logger.info("[inference] step 3/8: align model weights to encrypted dataset columns")
    aligned_weights, weight_mapping = align_weights_to_columns(
        columns=plan.columns,
        model=model,
    )

    logger.info(
        "[inference] step 4/8: load inference crypto context + eval keys "
        "(storage_path=%s)",
        fhe_key_storage_path,
    )
    try:
        cc, _public_key = load_inference_context(fhe_key_storage_path)
    except KeyLoadError as exc:
        logger.error(
            "[inference] failed to load inference context: storage_path=%s error=%s",
            fhe_key_storage_path,
            exc,
        )
        raise ValueError(str(exc)) from exc
    try:
        ring_dim = cc.GetRingDimension()
    except Exception:
        ring_dim = "unknown"
    logger.info(
        "[inference] inference context loaded: storage_path=%s ring_dimension=%s",
        fhe_key_storage_path,
        ring_dim,
    )

    dataset_dir = ENCRYPTED_DIR / encrypt_id
    logger.info(
        "[inference] step 5/8: resolve encrypted dataset directory: path=%s exists=%s",
        dataset_dir,
        dataset_dir.exists(),
    )
    if not dataset_dir.exists():
        raise ValueError(
            f"Encrypted dataset files not found at {dataset_dir}. "
            f"encrypt_id={encrypt_id}"
        )

    result_id = secrets.token_hex(16)
    output_dir = RESULTS_DIR / result_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[inference] step 6/8: created result output directory: result_id=%s path=%s",
        result_id,
        output_dir,
    )

    result_files: list[str] = []
    logger.info(
        "[inference] step 7/8: processing %s ciphertext chunk(s)",
        len(ciphertext_files),
    )
    for chunk_index, ciphertext_name in enumerate(ciphertext_files):
        input_path = dataset_dir / ciphertext_name
        logger.info(
            "[inference] chunk %s/%s: input=%s",
            chunk_index + 1,
            len(ciphertext_files),
            input_path,
        )
        if not input_path.exists():
            raise ValueError(f"Missing ciphertext file: {input_path}")

        ciphertext = _read_ciphertext(input_path)
        logger.info(
            "[inference] chunk %s/%s: running batched linear score "
            "(rows_in_chunk=%s, score_slot_stride=%s)",
            chunk_index + 1,
            len(ciphertext_files),
            min(plan.rows_per_ciphertext, plan.total_rows - chunk_index * plan.rows_per_ciphertext),
            plan.params_count,
        )
        result_ct = compute_batched_linear_scores(
            cc,
            ciphertext,
            weights=aligned_weights,
            intercept=model.intercept,
            plan=plan,
        )

        result_name = f"result_{chunk_index:04d}.bin"
        result_path = output_dir / result_name
        _write_ciphertext(result_path, result_ct)
        result_files.append(result_name)
        logger.info(
            "[inference] chunk %s/%s: wrote result ciphertext: %s",
            chunk_index + 1,
            len(ciphertext_files),
            result_path,
        )

    manifest = {
        "result_id": result_id,
        "result_path": str(output_dir),
        "encrypted_dataset_id": dataset_id,
        "encrypt_id": encrypt_id,
        "dataset_model_id": dataset_model_id,
        "dataset_model_name": dataset_model_name,
        "dataset_model_type": dataset_model_type,
        "model_id": inference_model_id,
        "model_name": inference_model_name,
        "model_type": inference_model_type,
        "supabase_fhe_key_id": fhe_key_id,
        "fhe_key_storage_path": fhe_key_storage_path,
        "operation": "batched_linear_score",
        "slots": plan.slots,
        "params_count": plan.params_count,
        "rows_per_ciphertext": plan.rows_per_ciphertext,
        "total_rows": plan.total_rows,
        "ciphertext_count": plan.ciphertext_count,
        "result_count": len(result_files),
        "row_result_slot_map": {
            "description": "Linear score for row k in chunk is at slot k * params_count",
            "formula": "score_k = intercept + sum(weight_j * feature_k_j)",
        },
        "columns": plan.columns,
        "model_feature_names": model.feature_names,
        "weight_mapping": weight_mapping,
        "intercept": model.intercept,
        "threshold": model.threshold,
        "classes": model.classes,
        "input_ciphertext_files": ciphertext_files,
        "result_files": result_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        "[inference] step 8/8: wrote manifest: path=%s result_files=%s "
        "weight_mapping_count=%s",
        manifest_path,
        result_files,
        len(weight_mapping),
    )
    logger.info(
        "[inference] run_inference complete: result_id=%s output_dir=%s "
        "encrypted_dataset_id=%s model_id=%s total_rows=%s result_count=%s",
        result_id,
        output_dir,
        dataset_id,
        inference_model_id,
        plan.total_rows,
        len(result_files),
    )

    return InferenceResult(
        result_id=result_id,
        output_dir=output_dir,
        result_files=result_files,
        manifest_file=str(manifest_path),
        manifest=manifest,
    )
