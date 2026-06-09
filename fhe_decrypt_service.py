import json
import logging
import math
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
class DecryptedRow:
    row_index: int
    linear_score: float
    probability: float | None = None
    predicted_class: str | None = None


@dataclass
class DecryptResultsOutput:
    result_id: str
    manifest: dict[str, Any]
    rows: list[DecryptedRow]


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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _predict_class(
    *,
    probability: float,
    threshold: float,
    classes: list[str],
) -> str | None:
    if not classes:
        return None
    class_index = 0 if probability >= threshold else 1
    return classes[class_index] if class_index < len(classes) else classes[0]


def _extract_linear_scores(
    packed_values: list[float],
    *,
    params_count: int,
    row_offset: int,
    rows_in_chunk: int,
) -> list[DecryptedRow]:
    rows: list[DecryptedRow] = []
    for local_row in range(rows_in_chunk):
        slot_index = local_row * params_count
        if slot_index >= len(packed_values):
            break
        rows.append(
            DecryptedRow(
                row_index=row_offset + local_row,
                linear_score=float(packed_values[slot_index]),
            )
        )
    return rows


def decrypt_inference_results(*, result_id: str) -> DecryptResultsOutput:
    result_id = _validate_result_id(result_id)
    logger.info("[decrypt] decrypt_inference_results start: result_id=%s", result_id)

    manifest = _load_result_manifest(result_id)
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

    threshold = manifest.get("threshold")
    threshold_value = float(threshold) if isinstance(threshold, (int, float)) else None
    classes_raw = manifest.get("classes") or []
    classes = (
        [value for value in classes_raw if isinstance(value, str)]
        if isinstance(classes_raw, list)
        else []
    )

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
    decrypted_rows: list[DecryptedRow] = []

    for chunk_index, result_name in enumerate(result_files):
        result_path = result_dir / str(result_name)
        if not result_path.exists():
            raise ValueError(f"Missing result ciphertext file: {result_path}")

        rows_remaining = total_rows - len(decrypted_rows)
        rows_in_chunk = min(rows_per_ciphertext, rows_remaining)
        if rows_in_chunk <= 0:
            break

        logger.info(
            "[decrypt] chunk %s/%s: file=%s rows_in_chunk=%s row_offset=%s",
            chunk_index + 1,
            len(result_files),
            result_path,
            rows_in_chunk,
            len(decrypted_rows),
        )

        ciphertext = _read_ciphertext(result_path)
        plaintext = cc.Decrypt(ciphertext, secret_key)
        plaintext.SetLength(slots)
        packed_values = plaintext.GetRealPackedValue()

        chunk_rows = _extract_linear_scores(
            packed_values,
            params_count=params_count,
            row_offset=len(decrypted_rows),
            rows_in_chunk=rows_in_chunk,
        )

        if threshold_value is not None and classes:
            for row in chunk_rows:
                row.probability = _sigmoid(row.linear_score)
                row.predicted_class = _predict_class(
                    probability=row.probability,
                    threshold=threshold_value,
                    classes=classes,
                )

        logger.info(
            "[decrypt] chunk %s/%s: extracted %s row score(s) preview=%s",
            chunk_index + 1,
            len(result_files),
            len(chunk_rows),
            [
                {
                    "row_index": row.row_index,
                    "linear_score": round(row.linear_score, 6),
                    "probability": round(row.probability, 6) if row.probability is not None else None,
                    "predicted_class": row.predicted_class,
                }
                for row in chunk_rows[:3]
            ],
        )
        decrypted_rows.extend(chunk_rows)

    if len(decrypted_rows) != total_rows:
        raise ValueError(
            f"Expected {total_rows} decrypted rows, extracted {len(decrypted_rows)}"
        )

    logger.info(
        "[decrypt] decrypt_inference_results complete: result_id=%s total_rows=%s",
        result_id,
        len(decrypted_rows),
    )

    return DecryptResultsOutput(
        result_id=result_id,
        manifest=manifest,
        rows=decrypted_rows,
    )
