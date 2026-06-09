import logging
from pathlib import Path

from openfhe import BINARY, DeserializeCryptoContext, DeserializePublicKey

from key_storage import FILE_NAMES, key_dir

logger = logging.getLogger("fhe_vault")


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


def load_inference_context(fhe_key_id: str):
    """Load crypto context, public key, and evaluation keys for homomorphic inference."""
    directory = key_dir(fhe_key_id)
    logger.info(
        "[inference] load_inference_context: key_id=%s directory=%s exists=%s",
        fhe_key_id,
        directory,
        directory.exists(),
    )
    if not directory.exists():
        raise KeyLoadError(
            f"FHE key files not found locally at {directory}. "
            f"Ensure the key was generated on this server (storage path: {fhe_key_id})."
        )

    cc_path = directory / FILE_NAMES["cryptocontext"]
    pk_path = directory / FILE_NAMES["publickey"]
    eval_mult_path = directory / FILE_NAMES["evalmult"]
    eval_auto_path = directory / FILE_NAMES["evalauto"]
    for path in (cc_path, pk_path, eval_mult_path, eval_auto_path):
        if not path.exists():
            raise KeyLoadError(f"Missing key file: {path}")
        logger.info(
            "[inference] key file present: name=%s path=%s size_bytes=%s",
            path.name,
            path,
            path.stat().st_size,
        )

    logger.info("[inference] deserializing crypto context: %s", cc_path)
    cc, cc_ok = DeserializeCryptoContext(str(cc_path), BINARY)
    if not cc_ok:
        raise KeyLoadError(f"Failed to deserialize crypto context from {cc_path}")

    logger.info("[inference] deserializing public key: %s", pk_path)
    public_key, pk_ok = DeserializePublicKey(str(pk_path), BINARY)
    if not pk_ok:
        raise KeyLoadError(f"Failed to deserialize public key from {pk_path}")

    logger.info("[inference] deserializing eval mult keys: %s", eval_mult_path)
    if not cc.DeserializeEvalMultKey(str(eval_mult_path), BINARY):
        raise KeyLoadError(f"Failed to deserialize eval mult keys from {eval_mult_path}")

    logger.info("[inference] deserializing eval automorphism keys: %s", eval_auto_path)
    if not cc.DeserializeEvalAutomorphismKey(str(eval_auto_path), BINARY):
        raise KeyLoadError(
            f"Failed to deserialize eval automorphism keys from {eval_auto_path}"
        )

    logger.info("[inference] load_inference_context complete: key_id=%s", fhe_key_id)
    return cc, public_key
