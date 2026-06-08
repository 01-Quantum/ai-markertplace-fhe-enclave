import os
from dataclasses import dataclass
from typing import Any, Dict
from uuid import UUID

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_TABLE = os.environ.get("SUPABASE_FHE_KEYS_TABLE", "fhe_keys")
SUPABASE_MODELS_TABLE = os.environ.get("SUPABASE_MODELS_TABLE", "models")


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
