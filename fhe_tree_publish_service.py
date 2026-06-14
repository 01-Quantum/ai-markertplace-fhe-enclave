import logging
from typing import Any

logger = logging.getLogger("fhe_vault")

RING_BASE = 15
RING_DIM = 2**RING_BASE
NUM_SLOTS = 2 ** (RING_BASE - 1)
DEFAULT_MULT_DEPTH = 9
DEFAULT_SCALING_MOD_SIZE = 59
DEFAULT_SECURITY_LEVEL = "HEStd_128_classic"
DEFAULT_SCALE = 0.5


def _validate_tree_json(tree: dict[str, Any]) -> None:
    nodes = tree.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Decision tree model_json must include a non-empty nodes array")

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"Node at index {index} must be an object")
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, int):
            raise ValueError(f"Node at index {index} is missing integer id")
        if node_type not in {"decision", "leaf"}:
            raise ValueError(f"Node id={node_id} has invalid type '{node_type}'")
        if node_type == "decision":
            for field in ("feature", "threshold", "leftBranchId", "rightBranchId"):
                if field not in node:
                    raise ValueError(f"Decision node id={node_id} is missing '{field}'")
        elif "label" not in node:
            raise ValueError(f"Leaf node id={node_id} is missing 'label'")


def infer_tree_depth(nodes_by_id: dict[int, dict[str, Any]], root_id: int) -> int:
    def depth(node_id: int) -> int:
        node = nodes_by_id[node_id]
        if node["type"] == "leaf":
            return 0
        return 1 + max(
            depth(int(node["leftBranchId"])),
            depth(int(node["rightBranchId"])),
        )

    return depth(root_id)


def id_to_heap_index(
    root_id: int,
    node_id: int,
    nodes_by_id: dict[int, dict[str, Any]],
) -> int:
    if node_id == root_id:
        return 0

    def search(parent_id: int, heap_index: int) -> int | None:
        node = nodes_by_id[parent_id]
        if node["type"] == "leaf":
            return None
        left_id = int(node["leftBranchId"])
        right_id = int(node["rightBranchId"])
        left_index = 2 * heap_index + 1
        right_index = 2 * heap_index + 2
        if node_id == left_id:
            return left_index
        if node_id == right_id:
            return right_index
        found = search(left_id, left_index)
        if found is not None:
            return found
        return search(right_id, right_index)

    index = search(root_id, 0)
    if index is None:
        raise ValueError(f"Node id={node_id} not found under root id={root_id}")
    return index


def path_index_to_label(
    root_id: int,
    nodes_by_id: dict[int, dict[str, Any]],
    tree_depth: int,
) -> dict[int, str]:
    labels: dict[int, str] = {}

    for leaf_index in range(2**tree_depth):
        bits = format(leaf_index, f"0{tree_depth}b")
        node = nodes_by_id[root_id]
        for bit in bits:
            if node["type"] == "leaf":
                break
            next_id = (
                int(node["rightBranchId"]) if bit == "1" else int(node["leftBranchId"])
            )
            node = nodes_by_id[next_id]
        label = str(node["label"])
        labels[leaf_index] = label.split("=", 1)[-1].strip()
    return labels


def materialize_tree_arrays(
    nodes_by_id: dict[int, dict[str, Any]],
    root_id: int,
    tree_depth: int,
) -> tuple[list[str | None], list[float], dict[int, str]]:
    num_nodes = 2**tree_depth - 1
    features: list[str | None] = [None] * num_nodes
    threshvals = [0.0] * num_nodes

    for node_id, node in nodes_by_id.items():
        heap_index = id_to_heap_index(root_id, int(node_id), nodes_by_id)
        if heap_index >= num_nodes or node["type"] != "decision":
            continue
        features[heap_index] = str(node["feature"])
        threshvals[heap_index] = float(node["threshold"])

    path_labels = path_index_to_label(root_id, nodes_by_id, tree_depth)
    return features, threshvals, path_labels


def build_tree_arrays(
    tree: dict[str, Any],
    *,
    tree_depth: int | None = None,
) -> dict[str, Any]:
    nodes = tree["nodes"]
    nodes_by_id = {int(node["id"]): node for node in nodes}
    root_id = min(nodes_by_id)
    if tree_depth is None:
        tree_depth = infer_tree_depth(nodes_by_id, root_id)

    num_nodes = 2**tree_depth - 1
    num_paths = 2**tree_depth
    features, threshvals, path_labels = materialize_tree_arrays(
        nodes_by_id,
        root_id,
        tree_depth,
    )

    return {
        "tree_depth": tree_depth,
        "num_nodes": num_nodes,
        "num_paths": num_paths,
        "features": features,
        "threshvals": threshvals,
        "threshvals_packed": threshvals + [0.0],
        "leaf_labels_by_path_index": path_labels,
    }


def build_server_bundle(tree_info: dict[str, Any], scale: float) -> dict[str, Any]:
    return {
        "format_version": 1,
        "tree_depth": tree_info["tree_depth"],
        "num_paths": tree_info["num_paths"],
        "num_nodes": tree_info["num_nodes"],
        "scale": scale,
        "threshvals_packed": tree_info["threshvals_packed"],
        "leaf_labels_by_path_index": {
            str(index): label
            for index, label in tree_info["leaf_labels_by_path_index"].items()
        },
    }


def build_client_contract(
    tree_info: dict[str, Any],
    *,
    scale: float,
    ring_base: int = RING_BASE,
    mult_depth: int = DEFAULT_MULT_DEPTH,
    scaling_mod_size: int = DEFAULT_SCALING_MOD_SIZE,
    security_level: str = DEFAULT_SECURITY_LEVEL,
) -> dict[str, Any]:
    num_paths = int(tree_info["num_paths"])
    num_nodes = int(tree_info["num_nodes"])
    node_features = tree_info["features"]
    used_features = sorted({feature for feature in node_features if feature is not None})
    rotation_keys = [index for index in range(1, num_paths)] + [
        -index for index in range(1, num_paths)
    ]

    return {
        "format_version": 1,
        "tree_depth": tree_info["tree_depth"],
        "num_paths": num_paths,
        "num_nodes": num_nodes,
        "scale": scale,
        "crypto": {
            "scheme": "CKKS",
            "library": "OpenFHE",
            "ring_base": ring_base,
            "ring_dim": RING_DIM,
            "num_slots": NUM_SLOTS,
            "max_samples_per_ciphertext": 2 ** (ring_base - 1 - int(tree_info["tree_depth"])),
            "multiplicative_depth": mult_depth,
            "scaling_mod_size": scaling_mod_size,
            "security_level": security_level,
            "rotation_keys": rotation_keys,
        },
        "packing": {
            "slots_per_sample": num_paths,
            "node_slots": num_nodes,
            "description": (
                "For each sample, build a block of `slots_per_sample` reals. "
                "For slot i in [0, node_slots): block[i] = normalized(row[node_features[i]]) "
                "if node_features[i] is not null, else 0. Slots [node_slots, slots_per_sample) "
                "are 0 (padding). Multiply every block value by `scale`. Concatenate sample "
                "blocks back-to-back (sample j at offset j*slots_per_sample), up to "
                "crypto.max_samples_per_ciphertext samples, then CKKS-encrypt the whole vector."
            ),
        },
        "node_features": node_features,
        "feature_order": used_features,
        "normalization": {
            "method": "none",
            "note": (
                "MODEL OWNER: replace with the exact preprocessing used at training "
                "so clients normalize identically (comparisons are wrong otherwise). "
                "For 'minmax' provide per-feature {min,max} mapped to [-1,1]; for "
                "'standard' provide {mean,std}. 'none' means rows are already normalized."
            ),
            "features": {feature: {} for feature in used_features},
        },
    }


def publish_decision_tree(
    *,
    model_json: dict[str, Any],
    tree_depth: int | None = None,
    scale: float = DEFAULT_SCALE,
) -> dict[str, Any]:
    _validate_tree_json(model_json)
    tree_info = build_tree_arrays(model_json, tree_depth=tree_depth)
    server_bundle = build_server_bundle(tree_info, scale)
    client_metadata = build_client_contract(tree_info, scale=scale)
    params_count = int(tree_info["num_paths"])

    logger.info(
        "[tree-publish] compiled tree: tree_depth=%s num_nodes=%s num_paths=%s "
        "feature_order=%s params_count=%s",
        tree_info["tree_depth"],
        tree_info["num_nodes"],
        tree_info["num_paths"],
        client_metadata["feature_order"],
        params_count,
    )

    return {
        "params_count": params_count,
        "tree_depth": tree_info["tree_depth"],
        "num_nodes": tree_info["num_nodes"],
        "num_paths": tree_info["num_paths"],
        "client_metadata": client_metadata,
        "published_model_json": {
            **model_json,
            "fhe_server_bundle": server_bundle,
        },
    }
