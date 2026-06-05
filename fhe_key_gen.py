import os
from typing import Dict, List, Tuple

from openfhe import *

_cc_cache: Dict[str, Tuple[bytes, bytes, bytes, bytes, bytes]] = {}
CACHE_MAX_SIZE = 1


def _cache_put(key_id: str, entry: Tuple[bytes, bytes, bytes, bytes, bytes]) -> None:
    _cc_cache.clear()
    _cc_cache[key_id] = entry


def fhe_key_gen(
    key_id: str,
    mult_depth: int = 7,
    eval_at_index_keys: List[int] | None = None,
) -> Tuple[bytes, bytes, bytes, bytes, bytes]:
    if eval_at_index_keys is None:
        eval_at_index_keys = []

    #ClearEvalMultKeys()
    security_level = SecurityLevel.HEStd_NotSet

    parameters = CCParamsCKKSRNS()
    parameters.SetSecurityLevel(security_level)
    parameters.SetRingDim(1 << 16)
    parameters.SetKeySwitchTechnique(HYBRID)

    scaling_mod_size = 30
    first_mod_size = 60

    parameters.SetScalingModSize(scaling_mod_size)
    parameters.SetFirstModSize(first_mod_size)
    parameters.SetMultiplicativeDepth(mult_depth)

    cc = GenCryptoContext(parameters)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    key_pair = cc.KeyGen()
    cc.EvalMultKeyGen(key_pair.secretKey)
    cc.EvalSumKeyGen(key_pair.secretKey)

    if len(eval_at_index_keys) > 0:
        print(f"Generating eval at index keys: {eval_at_index_keys}")
        cc.EvalAtIndexKeyGen(key_pair.secretKey, eval_at_index_keys)
    if mult_depth > 10:
        cc.EvalAtIndexKeyGen(key_pair.secretKey, [-1, -2, -3, 3])

    cc_bin = Serialize(cc, BINARY)
    sk_bin = Serialize(key_pair.secretKey, BINARY)
    pk_bin = Serialize(key_pair.publicKey, BINARY)

    mk = SerializeEvalMultKeyString(BINARY, "")
    ak = SerializeEvalAutomorphismKeyString(BINARY, "")

    #ClearEvalMultKeys()
    cc.ClearEvalAutomorphismKeys()
    #ReleaseAllContexts()

    entry = (cc_bin, sk_bin, pk_bin, mk, ak)
    _cache_put(key_id, entry)
    return entry
