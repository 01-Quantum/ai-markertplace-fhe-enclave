import csv
import io
import json
import math
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from openfhe import BINARY, Serialize, SerializeToFile

from fhe_key_load import load_encryption_context

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


def plan_encryption(slots: int, params_count: int, csv: CsvPreprocessResult) -> EncryptPlan:
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


def _pack_chunk(
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
    cc, public_key = load_encryption_context(fhe_key_storage_path)

    encrypt_id = secrets.token_hex(16)
    output_dir = ENCRYPTED_DIR / encrypt_id
    output_dir.mkdir(parents=True, exist_ok=True)

    ciphertext_files: list[str] = []
    for chunk_index, start in enumerate(
        range(0, csv.total_rows, plan.rows_per_ciphertext)
    ):
        chunk_rows = csv.data_rows[start : start + plan.rows_per_ciphertext]
        values = _pack_chunk(chunk_rows, plan.params_count, plan.slots)
        plaintext = cc.MakeCKKSPackedPlaintext(values)
        ciphertext = cc.Encrypt(public_key, plaintext)

        ct_path = output_dir / f"ciphertext_{chunk_index:04d}.bin"
        _write_ciphertext(ct_path, ciphertext)
        ciphertext_files.append(str(ct_path))

    manifest = {
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
