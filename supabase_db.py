import os
from typing import Any, Dict
from uuid import UUID

import httpx

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_TABLE = os.environ.get("SUPABASE_FHE_KEYS_TABLE", "fhe_keys")


class SupabaseError(Exception):
    pass


def _user_headers(access_token: str) -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def insert_fhe_key_record(
    *,
    key_name: str,
    scheme: str,
    multiplicative_dep: int,
    key_id: str,
    user_id: UUID,
    access_token: str,
) -> Dict[str, Any]:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise SupabaseError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

    row: Dict[str, Any] = {
        "key_name": key_name,
        "scheme": scheme,
        "multiplicative_dep": multiplicative_dep,
        "public_key_storag": key_id,
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
