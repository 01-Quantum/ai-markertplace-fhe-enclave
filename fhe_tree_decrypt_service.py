import logging
from pathlib import Path

from openfhe import BINARY, DeserializeCiphertext

from fhe_decrypt_service import DecryptResultsOutput, _load_result_manifest, _validate_result_id
from fhe_inference_service import RESULTS_DIR
from fhe_key_load import KeyLoadError, load_decryption_context

logger = logging.getLogger("fhe_vault")


def _read_ciphertext(path: Path):
    ciphertext, ok = DeserializeCiphertext(str(path), BINARY)
    if not ok:
        raise ValueError(f"Failed to deserialize ciphertext from {path}")
    return ciphertext


def _path_index_from_scores(scores: list[float]) -> int:
    return min(range(len(scores)), key=lambda index: scores[index])


def decrypt_tree_inference_results(
    *,
    result_id: str,
    manifest: dict | None = None,
) -> DecryptResultsOutput:
    result_id = _validate_result_id(result_id)
    logger.info("[tree-decrypt] start: result_id=%s", result_id)

    if manifest is None:
        manifest = _load_result_manifest(result_id)
    fhe_key_storage_path = manifest.get("fhe_key_storage_path")
    if not isinstance(fhe_key_storage_path, str) or not fhe_key_storage_path:
        raise ValueError("Result manifest is missing fhe_key_storage_path")

    num_paths = int(manifest["num_paths"])
    rows_per_ciphertext = int(manifest["rows_per_ciphertext"])
    total_rows = int(manifest["total_rows"])
    result_files = manifest.get("result_files") or []
    if not isinstance(result_files, list) or not result_files:
        raise ValueError("Result manifest is missing result_files")

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

        ciphertext = _read_ciphertext(result_path)
        plaintext = cc.Decrypt(ciphertext, secret_key)
        plaintext.SetLength(rows_in_chunk * num_paths)
        flat = plaintext.GetRealPackedValue()[: rows_in_chunk * num_paths]

        for sample_index in range(rows_in_chunk):
            start = sample_index * num_paths
            chunk_scores = flat[start : start + num_paths]
            decrypted_values.append(float(_path_index_from_scores(chunk_scores)))

        logger.info(
            "[tree-decrypt] chunk %s/%s: extracted %s path index(es) preview=%s",
            chunk_index + 1,
            len(result_files),
            rows_in_chunk,
            [int(value) for value in decrypted_values[:3]],
        )

    if len(decrypted_values) != total_rows:
        raise ValueError(
            f"Expected {total_rows} decrypted path indices, extracted {len(decrypted_values)}"
        )

    logger.info(
        "[tree-decrypt] complete: result_id=%s total_rows=%s",
        result_id,
        len(decrypted_values),
    )

    return DecryptResultsOutput(
        result_id=result_id,
        manifest=manifest,
        decrypted_values=decrypted_values,
    )
