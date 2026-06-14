import os
from typing import Dict, List, Tuple

from openfhe import *

from fhe_mem import log_memory

RING_BASE = 15
RING_DIM = 2**RING_BASE
NUM_SLOTS = 2 ** (RING_BASE - 1)

# Single "big" key that works for both logistic and decision-tree models.
# Trees need a deeper modulus chain (ApproxComp deg-13 + EncTreeEvaluator),
# higher precision, and rotation keys for the tree evaluator's rotations.
SCALING_MOD_SIZE = 59
FIRST_MOD_SIZE = 60
TREE_MIN_MULT_DEPTH = 9
# Rotation keys are generated for every tree depth up to this max (one key
# serves all published trees of depth <= TREE_MAX_DEPTH).
# Each extra depth roughly doubles the rotation-key count and the peak RAM
# needed during key generation/serialization. Depth 4 (the bundled trees)
# needs ~13 keys; depth 6 needs ~54 and can OOM small containers. Raise this
# (and the container memory) only if you publish deeper trees.
TREE_MAX_DEPTH = int(os.environ.get("FHE_TREE_MAX_DEPTH", "4"))

_cc_cache: Dict[str, Tuple[bytes, bytes, bytes, bytes, bytes]] = {}
CACHE_MAX_SIZE = 1


def _cache_put(key_id: str, entry: Tuple[bytes, bytes, bytes, bytes, bytes]) -> None:
    _cc_cache.clear()
    _cc_cache[key_id] = entry


def effective_mult_depth(requested: int) -> int:
    """Bump requested depth so the key always supports tree evaluation."""
    return max(requested, TREE_MIN_MULT_DEPTH)


def tree_rotation_indices(max_depth: int) -> List[int]:
    """
    Exact rotation indices used by fhe_tree_eval.enc_tree_evaluator for every
    tree depth in 1..max_depth. Matches the EvalRotate calls there so a single
    key covers all trees up to max_depth (far smaller than the reference's full
    [+/-1 .. +/-(num_paths-1)] range).
    """
    rotations: set[int] = set()
    for tree_depth in range(1, max_depth + 1):
        num_nodes = 2**tree_depth - 1
        for i in range(num_nodes):
            bl = (i + 1).bit_length() - 1
            firstrotval = -int((i + 1) % (2**bl) * 2 ** (tree_depth - bl) - i)
            if firstrotval != 0:
                rotations.add(firstrotval)
            for j in range(tree_depth - 1 - bl):
                rotations.add(int(-(2**j)))
            rotations.add(int(-(2 ** (tree_depth - bl - 1))))
    return sorted(rotations)


def fhe_key_gen(
    key_id: str,
    mult_depth: int = 7,
    eval_at_index_keys: List[int] | None = None,
) -> Tuple[bytes, bytes, bytes, bytes, bytes]:
    if eval_at_index_keys is None:
        eval_at_index_keys = []

    mult_depth = effective_mult_depth(mult_depth)

    with log_memory(f"keygen(mult_depth={mult_depth})"):
        return _fhe_key_gen(key_id, mult_depth, eval_at_index_keys)


def _fhe_key_gen(
    key_id: str,
    mult_depth: int,
    eval_at_index_keys: List[int],
) -> Tuple[bytes, bytes, bytes, bytes, bytes]:
    #ClearEvalMultKeys()
    security_level = SecurityLevel.HEStd_NotSet

    parameters = CCParamsCKKSRNS()
    parameters.SetSecurityLevel(security_level)
    parameters.SetRingDim(RING_DIM)
    parameters.SetKeySwitchTechnique(HYBRID)

    scaling_mod_size = SCALING_MOD_SIZE
    first_mod_size = FIRST_MOD_SIZE

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

    rotation_indices = sorted(
        set(eval_at_index_keys) | set(tree_rotation_indices(TREE_MAX_DEPTH))
    )
    print(
        f"Generating {len(rotation_indices)} rotation keys "
        f"(tree_max_depth={TREE_MAX_DEPTH}, mult_depth={mult_depth})"
    )
    cc.EvalAtIndexKeyGen(key_pair.secretKey, rotation_indices)

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
