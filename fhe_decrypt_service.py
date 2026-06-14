import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openfhe import BINARY, DeserializeCiphertext

from fhe_inference_service import RESULTS_DIR
from fhe_key_load import KeyLoadError, load_decryption_context

logger = logging.getLogger("fhe_vault")

_RESULT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class DecryptResultsOutput:
    result_id: str
    manifest: dict[str, Any]
    decrypted_values: list[float]
    # Decision-tree only: per-sample predicted leaf label (argmin of path scores
    # mapped through the model's leaf label LUT). None for logistic models.
    predicted_labels: list[str] | None = None


def _validate_result_id(result_id: str) -> str:
    if not _RESULT_ID_RE.fullmatch(result_id):
        raise ValueError(f"Invalid result_id: {result_id}")
    return result_id


def _load_result_manifest(result_id: str) -> dict[str, Any]:
    manifest_path = RESULTS_DIR / result_id / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Result manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid manifest JSON at {manifest_path}")
    return manifest


def _read_ciphertext(path: Path):
    ciphertext, ok = DeserializeCiphertext(str(path), BINARY)
    if not ok:
        raise ValueError(f"Failed to deserialize ciphertext from {path}")
    return ciphertext


def _extract_decrypted_values(
    packed_values: list[float],
    *,
    params_count: int,
    rows_in_chunk: int,
) -> list[float]:
    scores: list[float] = []
    for local_row in range(rows_in_chunk):
        slot_index = local_row * params_count
        if slot_index >= len(packed_values):
            break
        scores.append(float(packed_values[slot_index]))
    return scores


def decrypt_inference_results(*, result_id: str) -> DecryptResultsOutput:
    result_id = _validate_result_id(result_id)
    manifest = _load_result_manifest(result_id)
    if manifest.get("operation") == "tree_eval":
        from fhe_tree_decrypt_service import decrypt_tree_inference_results

        return decrypt_tree_inference_results(result_id=result_id, manifest=manifest)

    logger.info("[decrypt] decrypt_inference_results start: result_id=%s", result_id)
    logger.info("[decrypt] loaded manifest: path=%s", RESULTS_DIR / result_id / "manifest.json")

    fhe_key_storage_path = manifest.get("fhe_key_storage_path")
    if not isinstance(fhe_key_storage_path, str) or not fhe_key_storage_path:
        raise ValueError("Result manifest is missing fhe_key_storage_path")

    slots = int(manifest["slots"])
    params_count = int(manifest["params_count"])
    rows_per_ciphertext = int(manifest["rows_per_ciphertext"])
    total_rows = int(manifest["total_rows"])
    result_files = manifest.get("result_files") or []
    if not isinstance(result_files, list) or not result_files:
        raise ValueError("Result manifest is missing result_files")

    logger.info(
        "[decrypt] manifest summary: encrypted_dataset_id=%s slots=%s params_count=%s "
        "rows_per_ciphertext=%s total_rows=%s result_files=%s fhe_key_storage_path=%s",
        manifest.get("encrypted_dataset_id"),
        slots,
        params_count,
        rows_per_ciphertext,
        total_rows,
        result_files,
        fhe_key_storage_path,
    )

    try:
        cc, secret_key = load_decryption_context(fhe_key_storage_path)
    except KeyLoadError as exc:
        raise ValueError(str(exc)) from exc

    result_dir = RESULTS_DIR / result_id
    decrypted_values: list[float] = []

    for chunk_index, result_name in enumerate(result_files):
        result_path = result_dir / str(result_name)
        if not result_path.exists():
            raise ValueError(f"Missing result ciphertext file: {result_path}")

        rows_remaining = total_rows - len(decrypted_values)
        rows_in_chunk = min(rows_per_ciphertext, rows_remaining)
        if rows_in_chunk <= 0:
            break

        logger.info(
            "[decrypt] chunk %s/%s: file=%s rows_in_chunk=%s row_offset=%s",
            chunk_index + 1,
            len(result_files),
            result_path,
            rows_in_chunk,
            len(decrypted_values),
        )

        ciphertext = _read_ciphertext(result_path)
        plaintext = cc.Decrypt(ciphertext, secret_key)
        plaintext.SetLength(slots)
        packed_values = plaintext.GetRealPackedValue()

        chunk_scores = _extract_decrypted_values(
            packed_values,
            params_count=params_count,
            rows_in_chunk=rows_in_chunk,
        )

        logger.info(
            "[decrypt] chunk %s/%s: extracted %s score(s) preview=%s",
            chunk_index + 1,
            len(result_files),
            len(chunk_scores),
            [round(score, 6) for score in chunk_scores[:3]],
        )
        decrypted_values.extend(chunk_scores)

    if len(decrypted_values) != total_rows:
        raise ValueError(
            f"Expected {total_rows} decrypted scores, extracted {len(decrypted_values)}"
        )

    logger.info(
        "[decrypt] decrypt_inference_results complete: result_id=%s total_rows=%s",
        result_id,
        len(decrypted_values),
    )

    return DecryptResultsOutput(
        result_id=result_id,
        manifest=manifest,
        decrypted_values=decrypted_values,
    )
