import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openfhe import BINARY, DeserializeCiphertext, Serialize, SerializeToFile

from fhe_encrypt_service import ENCRYPTED_DIR
from fhe_inference_service import InferenceResult, RESULTS_DIR
from fhe_key_load import KeyLoadError, load_inference_context
from fhe_tree_eval import approx_comp, enc_tree_evaluator

logger = logging.getLogger("fhe_vault")


@dataclass
class TreeServerBundle:
    tree_depth: int
    num_paths: int
    num_nodes: int
    scale: float
    threshvals_packed: list[float]
    leaf_labels_by_path_index: dict[int, str]


@dataclass
class TreeInferencePlan:
    slots: int
    slots_per_sample: int
    rows_per_ciphertext: int
    total_rows: int
    ciphertext_count: int
    tree_depth: int
    num_paths: int
    columns: list[str]


def parse_tree_server_bundle(model_json: dict[str, Any]) -> TreeServerBundle:
    bundle = model_json.get("fhe_server_bundle")
    if not isinstance(bundle, dict):
        raise ValueError(
            "Tree model is missing fhe_server_bundle. Publish with /fhe-tree-publish first."
        )

    threshvals_packed = bundle.get("threshvals_packed")
    if not isinstance(threshvals_packed, list) or not threshvals_packed:
        raise ValueError("Tree server bundle is missing threshvals_packed")

    labels_raw = bundle.get("leaf_labels_by_path_index") or {}
    if not isinstance(labels_raw, dict) or not labels_raw:
        raise ValueError("Tree server bundle is missing leaf_labels_by_path_index")

    return TreeServerBundle(
        tree_depth=int(bundle["tree_depth"]),
        num_paths=int(bundle["num_paths"]),
        num_nodes=int(bundle["num_nodes"]),
        scale=float(bundle["scale"]),
        threshvals_packed=[float(value) for value in threshvals_packed],
        leaf_labels_by_path_index={
            int(key): str(value) for key, value in labels_raw.items()
        },
    )


def build_tree_inference_plan(
    *,
    slots: int,
    slots_per_sample: int,
    rows_per_ciphertext: int,
    total_rows: int,
    ciphertext_count: int,
    tree_depth: int,
    num_paths: int,
    columns: list[str],
) -> TreeInferencePlan:
    if slots_per_sample <= 0:
        raise ValueError("slots_per_sample must be greater than 0")
    if rows_per_ciphertext <= 0:
        raise ValueError("rows_per_ciphertext must be greater than 0")
    if rows_per_ciphertext * slots_per_sample > slots:
        raise ValueError(
            f"Tree chunk requires {rows_per_ciphertext * slots_per_sample} slots, "
            f"key only has {slots}"
        )

    return TreeInferencePlan(
        slots=slots,
        slots_per_sample=slots_per_sample,
        rows_per_ciphertext=rows_per_ciphertext,
        total_rows=total_rows,
        ciphertext_count=ciphertext_count,
        tree_depth=tree_depth,
        num_paths=num_paths,
        columns=columns,
    )


def _read_ciphertext(path: Path):
    ciphertext, ok = DeserializeCiphertext(str(path), BINARY)
    if not ok:
        raise ValueError(f"Failed to deserialize ciphertext from {path}")
    return ciphertext


def _write_ciphertext(path: Path, ciphertext) -> None:
    if SerializeToFile(str(path), ciphertext, BINARY):
        return
    path.write_bytes(Serialize(ciphertext, BINARY))


def server_evaluate_tree(
    cc,
    public_key,
    cipherinput,
    *,
    num_samples: int,
    server_bundle: TreeServerBundle,
    num_slots: int,
):
    """
    Subtract scaled thresholds as a plaintext operand, then ApproxComp + EncTreeEvaluator.
    Mirrors fhe-decision-tree/CKKS-tree-inference.py server_evaluate().
    """
    scaled_thresh = [value * server_bundle.scale for value in server_bundle.threshvals_packed]
    packed_thresh = scaled_thresh * num_samples
    thresh_plaintext = cc.MakeCKKSPackedPlaintext(packed_thresh)
    subcipher = cc.EvalSub(cipherinput, thresh_plaintext)
    comparisons = approx_comp(subcipher, cc)
    return enc_tree_evaluator(
        comparisons,
        cc,
        num_samples,
        num_slots,
        server_bundle.tree_depth,
        public_key,
    )


def run_tree_inference(
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
        "[tree-inference] start: dataset_id=%s encrypt_id=%s model_id=%s "
        "slots=%s slots_per_sample=%s rows_per_ciphertext=%s total_rows=%s",
        dataset_id,
        encrypt_id,
        inference_model_id,
        slots,
        params_count,
        rows_per_ciphertext,
        total_rows,
    )

    server_bundle = parse_tree_server_bundle(model_json)
    if params_count != server_bundle.num_paths:
        raise ValueError(
            f"Encrypted dataset slots_per_sample ({params_count}) does not match "
            f"tree num_paths ({server_bundle.num_paths})"
        )

    plan = build_tree_inference_plan(
        slots=slots,
        slots_per_sample=params_count,
        rows_per_ciphertext=rows_per_ciphertext,
        total_rows=total_rows,
        ciphertext_count=ciphertext_count,
        tree_depth=server_bundle.tree_depth,
        num_paths=server_bundle.num_paths,
        columns=columns,
    )

    try:
        cc, public_key = load_inference_context(fhe_key_storage_path)
    except KeyLoadError as exc:
        raise ValueError(str(exc)) from exc

    dataset_dir = ENCRYPTED_DIR / encrypt_id
    if not dataset_dir.exists():
        raise ValueError(f"Encrypted dataset files not found at {dataset_dir}")

    result_id = secrets.token_hex(16)
    output_dir = RESULTS_DIR / result_id
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files: list[str] = []
    for chunk_index, ciphertext_name in enumerate(ciphertext_files):
        input_path = dataset_dir / ciphertext_name
        if not input_path.exists():
            raise ValueError(f"Missing ciphertext file: {input_path}")

        rows_remaining = total_rows - chunk_index * plan.rows_per_ciphertext
        num_samples = min(plan.rows_per_ciphertext, rows_remaining)
        if num_samples <= 0:
            break

        logger.info(
            "[tree-inference] chunk %s/%s: input=%s num_samples=%s",
            chunk_index + 1,
            len(ciphertext_files),
            input_path,
            num_samples,
        )

        ciphertext = _read_ciphertext(input_path)
        result_ct = server_evaluate_tree(
            cc,
            public_key,
            ciphertext,
            num_samples=num_samples,
            server_bundle=server_bundle,
            num_slots=plan.slots,
        )

        result_name = f"result_{chunk_index:04d}.bin"
        result_path = output_dir / result_name
        _write_ciphertext(result_path, result_ct)
        result_files.append(result_name)

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
        "operation": "tree_eval",
        "slots": plan.slots,
        "params_count": plan.slots_per_sample,
        "slots_per_sample": plan.slots_per_sample,
        "rows_per_ciphertext": plan.rows_per_ciphertext,
        "total_rows": plan.total_rows,
        "ciphertext_count": plan.ciphertext_count,
        "result_count": len(result_files),
        "tree_depth": plan.tree_depth,
        "num_paths": plan.num_paths,
        "scale": server_bundle.scale,
        "columns": plan.columns,
        "leaf_labels_by_path_index": {
            str(index): label
            for index, label in server_bundle.leaf_labels_by_path_index.items()
        },
        "row_result_slot_map": {
            "description": (
                "Per-sample path scores occupy num_paths consecutive slots; "
                "predicted leaf is argmin of the score block"
            ),
        },
        "input_ciphertext_files": ciphertext_files,
        "result_files": result_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(
        "[tree-inference] complete: result_id=%s result_files=%s total_rows=%s",
        result_id,
        result_files,
        plan.total_rows,
    )

    return InferenceResult(
        result_id=result_id,
        output_dir=output_dir,
        result_files=result_files,
        manifest_file=str(manifest_path),
        manifest=manifest,
    )
