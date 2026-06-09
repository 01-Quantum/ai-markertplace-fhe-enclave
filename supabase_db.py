import os
from dataclasses import dataclass
from typing import Any, Dict
from uuid import UUID

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_TABLE = os.environ.get("SUPABASE_FHE_KEYS_TABLE", "fhe_keys")
SUPABASE_MODELS_TABLE = os.environ.get("SUPABASE_MODELS_TABLE", "models")
SUPABASE_ENCRYPTED_DATASETS_TABLE = os.environ.get(
    "SUPABASE_ENCRYPTED_DATASETS_TABLE", "fhe_encrypted_datasets"
)


class SupabaseError(Exception):
    pass


class SupabaseNotFoundError(SupabaseError):
    pass


def _user_headers(access_token: str) -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _get_single_row(
    *,
    table: str,
    access_token: str,
    filters: Dict[str, str],
    select: str,
    not_found_message: str,
) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise SupabaseError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

    filter_query = "&".join(f"{key}=eq.{value}" for key, value in filters.items())
    url = f"{SUPABASE_URL}/rest/v1/{table}?{filter_query}&select={select}&limit=1"
    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=_user_headers(access_token))

    if response.status_code >= 400:
        raise SupabaseError(
            f"Supabase query failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    if not data:
        raise SupabaseNotFoundError(not_found_message)
    return data[0]


@dataclass
class FheKeyRecord:
    id: int
    slots: int
    storage_path: str
    key_name: str


def _fhe_key_lookup_filter(fhe_key_id: str) -> Dict[str, str]:
    if fhe_key_id.isdigit():
        return {"id": fhe_key_id}
    return {"public_key_storage_path": fhe_key_id}


def resolve_fhe_key(*, fhe_key_id: str, access_token: str) -> FheKeyRecord:
    row = _get_single_row(
        table=SUPABASE_TABLE,
        access_token=access_token,
        filters=_fhe_key_lookup_filter(fhe_key_id),
        select="id,slots,public_key_storage_path,is_active,key_name",
        not_found_message=f"FHE key not found: {fhe_key_id}",
    )

    slots = row.get("slots")
    storage_path = row.get("public_key_storage_path")
    if slots is None:
        raise SupabaseError(f"FHE key {fhe_key_id} has no slots value")
    if not storage_path:
        raise SupabaseError(
            f"FHE key {fhe_key_id} has no public_key_storage_path (local key id missing)"
        )

    return FheKeyRecord(
        id=int(row["id"]),
        slots=int(slots),
        storage_path=str(storage_path),
        key_name=str(row.get("key_name", "")),
    )


@dataclass
class ModelRecord:
    id: str
    params_count: int
    name: str
    model_type: str


def resolve_model(*, model_id: str, access_token: str) -> ModelRecord:
    row = _get_single_row(
        table=SUPABASE_MODELS_TABLE,
        access_token=access_token,
        filters={"id": model_id},
        select="id,params_count,model_name,model_type",
        not_found_message=f"Model not found: {model_id}",
    )
    params_count = row.get("params_count")
    if params_count is None:
        raise SupabaseError(f"Model {model_id} has no params_count value")

    name = row.get("model_name") or ""
    model_type = row.get("model_type") or ""

    return ModelRecord(
        id=str(row["id"]),
        params_count=int(params_count),
        name=str(name),
        model_type=str(model_type),
    )


def insert_fhe_key_record(
    *,
    key_name: str,
    scheme: str,
    multiplicative_depth: int,
    slots: int,
    key_id: str,
    user_id: UUID,
    access_token: str,
) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise SupabaseError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

    row: Dict[str, Any] = {
        "key_name": key_name,
        "scheme": scheme,
        "multiplicative_depth": multiplicative_depth,
        "slots": slots,
        "public_key_storage_path": key_id,
        "user_id": str(user_id),
        "is_active": True,
    }

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=_user_headers(access_token), json=row)

    if response.status_code >= 400:
        raise SupabaseError(
            f"Supabase insert failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return row


def insert_fhe_encrypted_dataset(
    *,
    user_id: UUID,
    encrypt_id: str,
    encrypt_path: str,
    source_file_name: str | None,
    source_file_size: int,
    model_id: int,
    model_name: str,
    model_type: str,
    fhe_key_id: int,
    fhe_key_storage_path: str,
    slots: int,
    params_count: int,
    rows_per_ciphertext: int,
    total_rows: int,
    ciphertext_count: int,
    removed_columns: list[str],
    columns: list[str],
    ciphertext_files: list[str],
    manifest_json: Dict[str, Any],
    access_token: str,
    status: str = "encrypted",
) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise SupabaseError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

    row: Dict[str, Any] = {
        "user_id": str(user_id),
        "encrypt_id": encrypt_id,
        "encrypt_path": encrypt_path,
        "source_file_name": source_file_name,
        "source_file_size": source_file_size,
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "fhe_key_id": fhe_key_id,
        "fhe_key_storage_path": fhe_key_storage_path,
        "slots": slots,
        "params_count": params_count,
        "rows_per_ciphertext": rows_per_ciphertext,
        "total_rows": total_rows,
        "ciphertext_count": ciphertext_count,
        "removed_columns": removed_columns,
        "columns": columns,
        "ciphertext_files": ciphertext_files,
        "manifest_json": manifest_json,
        "status": status,
    }

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_ENCRYPTED_DATASETS_TABLE}"
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, headers=_user_headers(access_token), json=row)

    if response.status_code >= 400:
        raise SupabaseError(
            f"Supabase insert failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return row


@dataclass
class EncryptedDatasetRecord:
    id: int
    encrypt_id: str
    encrypt_path: str
    user_id: str


def resolve_fhe_encrypted_dataset(
    *, dataset_id: int, access_token: str
) -> EncryptedDatasetRecord:
    row = _get_single_row(
        table=SUPABASE_ENCRYPTED_DATASETS_TABLE,
        access_token=access_token,
        filters={"id": str(dataset_id)},
        select="id,encrypt_id,encrypt_path,user_id",
        not_found_message=f"Encrypted dataset not found: {dataset_id}",
    )
    return EncryptedDatasetRecord(
        id=int(row["id"]),
        encrypt_id=str(row["encrypt_id"]),
        encrypt_path=str(row.get("encrypt_path", "")),
        user_id=str(row["user_id"]),
    )


def delete_fhe_encrypted_dataset(*, dataset_id: int, access_token: str) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise SupabaseError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_ENCRYPTED_DATASETS_TABLE}?id=eq.{dataset_id}"
    with httpx.Client(timeout=30.0) as client:
        response = client.delete(url, headers=_user_headers(access_token))

    if response.status_code >= 400:
        raise SupabaseError(
            f"Supabase delete failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    raise SupabaseNotFoundError(f"Encrypted dataset not found: {dataset_id}")
