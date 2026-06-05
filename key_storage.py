import os
import secrets
from pathlib import Path
from typing import Tuple

from fhe_key_gen import fhe_key_gen

KEYS_DIR = Path(os.environ.get("FHE_KEYS_DIR", "/data/keys"))

FILE_NAMES = {
    "cryptocontext": "cryptocontext.bin",
    "secretkey": "secretkey.bin",
    "publickey": "publickey.bin",
    "evalmult": "evalmult.bin",
    "evalauto": "evalauto.bin",
}


def new_key_id() -> str:
    return secrets.token_hex(16)


def key_dir(key_id: str) -> Path:
    return KEYS_DIR / key_id


def write_key_files(
    key_id: str,
    cc_bin: bytes,
    sk_bin: bytes,
    pk_bin: bytes,
    mk: bytes,
    ak: bytes,
) -> None:
    directory = key_dir(key_id)
    directory.mkdir(parents=True, exist_ok=True)

    files = {
        FILE_NAMES["cryptocontext"]: cc_bin,
        FILE_NAMES["secretkey"]: sk_bin,
        FILE_NAMES["publickey"]: pk_bin,
        FILE_NAMES["evalmult"]: mk,
        FILE_NAMES["evalauto"]: ak,
    }
    for name, data in files.items():
        (directory / name).write_bytes(data)


def generate_and_store(
    mult_depth: int,
    eval_at_index_keys: list[int] | None = None,
) -> str:
    key_id = new_key_id()
    cc_bin, sk_bin, pk_bin, mk, ak = fhe_key_gen(
        key_id, mult_depth=mult_depth, eval_at_index_keys=eval_at_index_keys
    )
    write_key_files(key_id, cc_bin, sk_bin, pk_bin, mk, ak)
    return key_id
