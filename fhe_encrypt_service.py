import csv
import io
import json
import logging
import math
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openfhe import BINARY, Serialize, SerializeToFile

from fhe_key_load import load_encryption_context
from fhe_mem import log_memory

logger = logging.getLogger("fhe_vault")

ENCRYPTED_DIR = Path(os.environ.get("FHE_ENCRYPTED_DIR", "/data/fhe-encrypted"))
STRIP_COLUMNS = frozenset({"actual", "expected"})
_ENCRYPT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class CsvPreprocessResult:
    columns: list[str]
    data_rows: list[list[str]]
    total_rows: int
    removed_columns: list[str]


@dataclass
class EncryptPlan:
    slots: int
    params_count: int
    rows_per_ciphertext: int
    total_rows: int
    ciphertext_count: int
    removed_columns: list[str]
    columns: list[str]
    client_metadata: dict[str, Any] | None = None


@dataclass
class ManifestContext:
    supabase_fhe_key_id: int
    fhe_key_storage_path: str
    model_id: str
    model_name: str
    model_type: str


@dataclass
class EncryptResult:
    encrypt_id: str
    output_dir: Path
    ciphertext_files: list[str]
    manifest_file: str
    manifest: dict


def _parse_float(value: str) -> float:
    value = value.strip()
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric CSV value '{value}' — remove label columns or use numeric features only"
        ) from exc


def _strip_label_columns(
    header: list[str], data_rows: list[list[str]]
) -> tuple[list[str], list[list[str]], list[str]]:
    drop_indices = {
        i
        for i, name in enumerate(header)
        if name.strip().lower() in STRIP_COLUMNS
    }
    if not drop_indices:
        return header, data_rows, []

    removed_columns = [header[i] for i in sorted(drop_indices)]
    new_header = [name for i, name in enumerate(header) if i not in drop_indices]
    new_rows = [
        [cell for i, cell in enumerate(row) if i not in drop_indices]
        for row in data_rows
    ]
    return new_header, new_rows, removed_columns


def preprocess_csv(content: bytes) -> CsvPreprocessResult:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV file must be UTF-8 encoded") from exc

    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return CsvPreprocessResult(
            columns=[],
            data_rows=[],
            total_rows=0,
            removed_columns=[],
        )

    header = rows[0]
    data_rows = rows[1:]
    header, data_rows, removed_columns = _strip_label_columns(header, data_rows)

    return CsvPreprocessResult(
        columns=header,
        data_rows=data_rows,
        total_rows=len(data_rows),
        removed_columns=removed_columns,
    )


def _validate_params_count(params_count: int, csv: CsvPreprocessResult) -> int:
    column_count = len(csv.columns)
    if column_count == 0:
        raise ValueError("CSV has no feature columns after removing label columns")

    if params_count != column_count:
        removed = ", ".join(csv.removed_columns) if csv.removed_columns else "none"
        raise ValueError(
            f"params_count ({params_count}) must exactly match CSV column count "
            f"({column_count}) after removing [{removed}]; columns={csv.columns}"
        )

    for row_index, row in enumerate(csv.data_rows, start=1):
        if len(row) != column_count:
            raise ValueError(
                f"CSV row {row_index} has {len(row)} values, expected {column_count} "
                f"to match header {csv.columns}"
            )

    return column_count


def _validate_client_metadata(client_metadata: dict[str, Any]) -> None:
    for field in (
        "node_features",
        "feature_order",
        "num_paths",
        "scale",
        "packing",
        "crypto",
    ):
        if field not in client_metadata:
            raise ValueError(f"client_metadata is missing required field '{field}'")


def _validate_tree_csv(client_metadata: dict[str, Any], csv: CsvPreprocessResult) -> None:
    if not csv.columns:
        raise ValueError("CSV has no feature columns after removing label columns")

    required_features = client_metadata["feature_order"]
    missing = [feature for feature in required_features if feature not in csv.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required tree feature columns {missing}; "
            f"expected feature_order={required_features}, got columns={csv.columns}"
        )

    column_index = {name: index for index, name in enumerate(csv.columns)}
    for row_index, row in enumerate(csv.data_rows, start=1):
        for feature in required_features:
            column_index_value = column_index[feature]
            if column_index_value >= len(row):
                raise ValueError(
                    f"CSV row {row_index} is missing value for feature '{feature}'"
                )


def _normalize_tree_value(
    feature: str,
    raw: float,
    normalization: dict[str, Any],
) -> float:
    method = normalization.get("method", "none")
    if method == "none":
        return float(raw)
    params = normalization.get("features", {}).get(feature, {})
    if method == "minmax":
        lo = float(params["min"])
        hi = float(params["max"])
        span = (hi - lo) or 1.0
        return 2.0 * (float(raw) - lo) / span - 1.0
    if method == "standard":
        mean = float(params["mean"])
        std = float(params.get("std", 1.0) or 1.0)
        return (float(raw) - mean) / std
    raise ValueError(f"Unknown normalization method: {method}")


def _csv_row_to_feature_map(
    columns: list[str],
    row: list[str],
) -> dict[str, float]:
    return {column: _parse_float(row[index]) for index, column in enumerate(columns)}


def build_tree_block(
    row: dict[str, float],
    client_metadata: dict[str, Any],
) -> list[float]:
    """Heap-layout input block for one sample, per the client contract."""
    node_features = client_metadata["node_features"]
    num_paths = int(client_metadata["num_paths"])
    normalization = client_metadata.get("normalization", {"method": "none"})

    block = [0.0] * num_paths
    for index, feature in enumerate(node_features):
        if feature is not None:
            block[index] = _normalize_tree_value(feature, row[feature], normalization)
    return block


def plan_tree_encryption(
    slots: int,
    client_metadata: dict[str, Any],
    csv: CsvPreprocessResult,
) -> EncryptPlan:
    _validate_client_metadata(client_metadata)
    _validate_tree_csv(client_metadata, csv)

    slots_per_sample = int(client_metadata["packing"]["slots_per_sample"])
    max_samples = int(client_metadata["crypto"]["max_samples_per_ciphertext"])
    if slots_per_sample <= 0:
        raise ValueError("client_metadata packing.slots_per_sample must be greater than 0")
    if max_samples <= 0:
        raise ValueError(
            "client_metadata crypto.max_samples_per_ciphertext must be greater than 0"
        )

    packed_slots = slots_per_sample * max_samples
    if packed_slots > slots:
        raise ValueError(
            f"Tree batch requires {packed_slots} slots "
            f"({max_samples} samples x {slots_per_sample} slots_per_sample) "
            f"but FHE key only has {slots} slots"
        )

    ciphertext_count = (
        math.ceil(csv.total_rows / max_samples) if csv.total_rows > 0 else 0
    )
    logger.info(
        "[encrypt] tree plan: slots_per_sample=%s max_samples=%s rows=%s ciphertexts=%s "
        "feature_order=%s",
        slots_per_sample,
        max_samples,
        csv.total_rows,
        ciphertext_count,
        client_metadata["feature_order"],
    )

    return EncryptPlan(
        slots=slots,
        params_count=slots_per_sample,
        rows_per_ciphertext=max_samples,
        total_rows=csv.total_rows,
        ciphertext_count=ciphertext_count,
        removed_columns=csv.removed_columns,
        columns=csv.columns,
        client_metadata=client_metadata,
    )


def plan_encryption(
    slots: int,
    params_count: int,
    csv: CsvPreprocessResult,
    *,
    client_metadata: dict[str, Any] | None = None,
) -> EncryptPlan:
    if client_metadata is not None:
        return plan_tree_encryption(slots, client_metadata, csv)

    if params_count <= 0:
        raise ValueError("params_count must be greater than 0")
    if slots <= 0:
        raise ValueError("slots must be greater than 0")

    _validate_params_count(params_count, csv)

    rows_per_ciphertext = slots // params_count
    if rows_per_ciphertext <= 0:
        raise ValueError(
            f"params_count ({params_count}) exceeds available slots ({slots})"
        )

    ciphertext_count = (
        math.ceil(csv.total_rows / rows_per_ciphertext) if csv.total_rows > 0 else 0
    )

    return EncryptPlan(
        slots=slots,
        params_count=params_count,
        rows_per_ciphertext=rows_per_ciphertext,
        total_rows=csv.total_rows,
        ciphertext_count=ciphertext_count,
        removed_columns=csv.removed_columns,
        columns=csv.columns,
    )


def _pack_linear_chunk(
    rows: list[list[str]],
    params_count: int,
    slots: int,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if len(row) != params_count:
            raise ValueError(
                f"Row has {len(row)} values, expected {params_count} feature columns"
            )
        for cell in row:
            values.append(_parse_float(cell))

    if len(values) > slots:
        raise ValueError(
            f"Packed values ({len(values)}) exceed available slots ({slots})"
        )

    values.extend([0.0] * (slots - len(values)))
    return values


def _pack_tree_chunk(
    rows: list[list[str]],
    columns: list[str],
    client_metadata: dict[str, Any],
    slots: int,
) -> list[float]:
    scale = float(client_metadata["scale"])
    num_paths = int(client_metadata["num_paths"])
    packed: list[float] = []

    for row in rows:
        feature_map = _csv_row_to_feature_map(columns, row)
        block = build_tree_block(feature_map, client_metadata)
        packed.extend(value * scale for value in block)

    expected = len(rows) * num_paths
    if len(packed) != expected:
        raise ValueError(
            f"Tree packing produced {len(packed)} values, expected {expected}"
        )
    if len(packed) > slots:
        raise ValueError(
            f"Packed tree values ({len(packed)}) exceed available slots ({slots})"
        )

    packed.extend([0.0] * (slots - len(packed)))
    return packed


def _write_ciphertext(path: Path, ciphertext) -> None:
    if SerializeToFile(str(path), ciphertext, BINARY):
        return
    path.write_bytes(Serialize(ciphertext, BINARY))


def encrypt_csv(
    *,
    fhe_key_storage_path: str,
    manifest_ctx: ManifestContext,
    csv: CsvPreprocessResult,
    plan: EncryptPlan,
) -> EncryptResult:
    is_tree = plan.client_metadata is not None
    encrypt_id = secrets.token_hex(16)
    output_dir = ENCRYPTED_DIR / encrypt_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ciphertext_files: list[str] = []
    with log_memory(f"encrypt({'tree' if is_tree else 'linear'}, rows={csv.total_rows})"):
        cc, public_key = load_encryption_context(fhe_key_storage_path)

        for chunk_index, start in enumerate(
            range(0, csv.total_rows, plan.rows_per_ciphertext)
        ):
            chunk_rows = csv.data_rows[start : start + plan.rows_per_ciphertext]
            if is_tree:
                values = _pack_tree_chunk(
                    chunk_rows,
                    plan.columns,
                    plan.client_metadata,
                    plan.slots,
                )
            else:
                values = _pack_linear_chunk(chunk_rows, plan.params_count, plan.slots)

            plaintext = cc.MakeCKKSPackedPlaintext(values)
            ciphertext = cc.Encrypt(public_key, plaintext)

            ct_path = output_dir / f"ciphertext_{chunk_index:04d}.bin"
            _write_ciphertext(ct_path, ciphertext)
            ciphertext_files.append(str(ct_path))

    manifest: dict[str, Any] = {
        "encrypt_path": str(output_dir),
        "model_id": manifest_ctx.model_id,
        "model_name": manifest_ctx.model_name,
        "model_type": manifest_ctx.model_type,
        "supabase_fhe_key_id": manifest_ctx.supabase_fhe_key_id,
        "fhe_key_storage_path": manifest_ctx.fhe_key_storage_path,
        "slots": plan.slots,
        "params_count": plan.params_count,
        "rows_per_ciphertext": plan.rows_per_ciphertext,
        "total_rows": plan.total_rows,
        "ciphertext_count": plan.ciphertext_count,
        "removed_columns": plan.removed_columns,
        "columns": plan.columns,
        "ciphertext_files": [Path(path).name for path in ciphertext_files],
    }
    if is_tree:
        manifest["operation"] = "tree_encrypt"
        manifest["slots_per_sample"] = plan.params_count
        manifest["feature_order"] = list(plan.client_metadata["feature_order"])
        manifest["node_features"] = list(plan.client_metadata["node_features"])
        manifest["scale"] = float(plan.client_metadata["scale"])
        manifest["tree_depth"] = int(plan.client_metadata["tree_depth"])
        manifest["num_paths"] = int(plan.client_metadata["num_paths"])

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return EncryptResult(
        encrypt_id=encrypt_id,
        output_dir=output_dir,
        ciphertext_files=ciphertext_files,
        manifest_file=str(manifest_path),
        manifest=manifest,
    )


def _encrypted_dataset_dir(encrypt_id: str) -> Path:
    if not _ENCRYPT_ID_RE.fullmatch(encrypt_id):
        raise ValueError(f"Invalid encrypt_id: {encrypt_id}")

    base = ENCRYPTED_DIR.resolve()
    target = (ENCRYPTED_DIR / encrypt_id).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"Refusing to delete path outside encrypted dir: {target}")
    return target


def delete_encrypted_dataset_files(encrypt_id: str) -> bool:
    """Remove ciphertext folder for encrypt_id. Returns True if a directory was removed."""
    target = _encrypted_dataset_dir(encrypt_id)
    if not target.exists():
        return False
    if not target.is_dir():
        raise ValueError(f"Encrypted dataset path is not a directory: {target}")
    shutil.rmtree(target)
    return True
