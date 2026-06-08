from pathlib import Path

from openfhe import BINARY, DeserializeCryptoContext, DeserializePublicKey

from key_storage import FILE_NAMES, key_dir


class KeyLoadError(Exception):
    pass


def load_encryption_context(fhe_key_id: str):
    directory = key_dir(fhe_key_id)
    if not directory.exists():
        raise KeyLoadError(
            f"FHE key files not found locally at {directory}. "
            f"Ensure the key was generated on this server (storage path: {fhe_key_id})."
        )

    cc_path = directory / FILE_NAMES["cryptocontext"]
    pk_path = directory / FILE_NAMES["publickey"]
    for path in (cc_path, pk_path):
        if not path.exists():
            raise KeyLoadError(f"Missing key file: {path}")

    cc, cc_ok = DeserializeCryptoContext(str(cc_path), BINARY)
    if not cc_ok:
        raise KeyLoadError(f"Failed to deserialize crypto context from {cc_path}")

    public_key, pk_ok = DeserializePublicKey(str(pk_path), BINARY)
    if not pk_ok:
        raise KeyLoadError(f"Failed to deserialize public key from {pk_path}")

    return cc, public_key
