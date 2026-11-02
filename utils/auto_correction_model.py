"""Score new entries with the Smart Cities auto-correction classifier.

The Kubeflow pipeline (kubeflow/smart_cities/auto_correction_pipeline.py) publishes each
trained model as plain JSON numbers (models/<model_type>.json): the scaler and coefficients
of the logistic regression, or the split arrays of every tree in the random forest. This
module rebuilds the pipeline's features for an entry and evaluates those numbers. Nothing is
unpickled, so a tampered model file can give wrong answers but cannot run code.

`featurize` must match `build_features` in the pipeline; tests/test_auto_correction_inference.py
runs both on the same rows.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List

import numpy as np

PORTABLE_FORMAT = "humaine-portable-classifier/1"
FIELD_TYPES = ("name", "surname", "father_name", "address", "vat_number",
               "id_number", "power_supply", "licence_plate", "mobile_phone")
NUMERIC_FIELDS = {"vat_number", "id_number", "power_supply", "licence_plate", "mobile_phone"}
FEATURE_NAMES = [
    *(f"field_{f}" for f in FIELD_TYPES),
    "is_numeric_field", "len_diff", "levenshtein", "levenshtein_norm",
    "char_multiset_equal", "digits_equal",
]
MAX_VALUE_LENGTH = 200  # keeps the O(n*m) edit distance cheap
# Limits for model files, so a planted file cannot exhaust memory or CPU. The pipeline's
# 200-tree, depth-8 forest is about 300 KB with about 20k nodes.
MAX_MODEL_BYTES = 20 * 1024 * 1024
MAX_TREES = 1000
MAX_TOTAL_NODES = 2_000_000
MAX_TREE_DEPTH = 64


def _clean(s: Any) -> str:
    s = unicodedata.normalize("NFD", str(s).upper())
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def featurize(field_type: str, supplied_value: Any, correct_value: Any) -> Dict[str, float]:
    """The pipeline's features for one (supplied value, correct value, field type) entry."""
    s, g = _clean(supplied_value), _clean(correct_value)
    lev = _levenshtein(s, g)
    row = {f"field_{f}": int(field_type == f) for f in FIELD_TYPES}
    row.update({
        "is_numeric_field": int(field_type in NUMERIC_FIELDS),
        "len_diff": len(s) - len(g),
        "levenshtein": lev,
        "levenshtein_norm": lev / max(len(s), len(g), 1),
        "char_multiset_equal": int(Counter(s.replace(" ", "")) == Counter(g.replace(" ", ""))),
        "digits_equal": int(re.sub(r"\D", "", s) == re.sub(r"\D", "", g)),
    })
    return row


def _numbers(value: Any, length: int, name: str, integer: bool = False) -> List:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a list of {length} numbers")
    kind = int if integer else (int, float)
    if not all(isinstance(v, kind) and not isinstance(v, bool) for v in value):
        raise ValueError(f"{name} must contain only {'integers' if integer else 'numbers'}")
    if not integer and not all(math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain only finite numbers")
    return value


def validate_portable_model(model: Any) -> Dict[str, Any]:
    """Check a portable model's structure, so evaluating it cannot fail or loop. Raises ValueError."""
    if not isinstance(model, dict) or model.get("format") != PORTABLE_FORMAT:
        raise ValueError(f"not a {PORTABLE_FORMAT} model")
    if model.get("feature_names") != FEATURE_NAMES:
        raise ValueError("the model was trained on different features than this assistant computes")
    n = len(FEATURE_NAMES)
    threshold = model.get("threshold", 0.5)
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    if model.get("model_type") == "logistic_regression":
        _numbers(model.get("scaler_mean"), n, "scaler_mean")
        scale = _numbers(model.get("scaler_scale"), n, "scaler_scale")
        if any(v == 0 for v in scale):
            raise ValueError("scaler_scale must not contain zeros")
        _numbers(model.get("coef"), n, "coef")
        _numbers([model.get("intercept")], 1, "intercept")
    elif model.get("model_type") == "random_forest":
        trees = model.get("trees")
        if not isinstance(trees, list) or not 0 < len(trees) <= MAX_TREES:
            raise ValueError(f"trees must be a list of 1 to {MAX_TREES} trees")
        if not all(isinstance(t, dict) and isinstance(t.get("left"), list) for t in trees):
            raise ValueError("each tree must have left/right/feature/threshold/p lists")
        if sum(len(t["left"]) for t in trees) > MAX_TOTAL_NODES:
            raise ValueError(f"the forest may have at most {MAX_TOTAL_NODES} nodes in total")
        for tree in trees:
            size = len(tree["left"])
            if size == 0:
                raise ValueError("each tree needs at least one node")
            left = _numbers(tree["left"], size, "left", integer=True)
            right = _numbers(tree.get("right"), size, "right", integer=True)
            feature = _numbers(tree.get("feature"), size, "feature", integer=True)
            _numbers(tree.get("threshold"), size, "threshold")
            p = _numbers(tree.get("p"), size, "p")
            depth = [0] * size
            for i in range(size):
                if left[i] == -1:
                    if right[i] != -1 or not 0 <= p[i] <= 1:
                        raise ValueError("leaf nodes need right == -1 and 0 <= p <= 1")
                    continue
                # Children always come after their parent, so traversal cannot cycle
                if not (i < left[i] < size and i < right[i] < size and 0 <= feature[i] < n):
                    raise ValueError("split nodes need later child indices and a valid feature index")
                # Longest path from the root, which bounds the steps one prediction takes in this tree
                for child in (left[i], right[i]):
                    depth[child] = max(depth[child], depth[i] + 1)
                    if depth[child] > MAX_TREE_DEPTH:
                        raise ValueError(f"trees may be at most {MAX_TREE_DEPTH} levels deep")
    else:
        raise ValueError("model_type must be logistic_regression or random_forest")
    return model


def predict_probability(model: Dict[str, Any], features: Dict[str, float]) -> float:
    """Probability of "auto-correctable" for one featurized entry, as scikit-learn computes it."""
    x = [float(features[name]) for name in model["feature_names"]]
    if model["model_type"] == "logistic_regression":
        z = model["intercept"] + sum(
            c * (v - m) / s for c, v, m, s in zip(model["coef"], x, model["scaler_mean"], model["scaler_scale"])
        )
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

    # Trees compare float32 inputs, as scikit-learn does
    x32 = [float(np.float32(v)) for v in x]
    total = 0.0
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            go_left = x32[tree["feature"][node]] <= tree["threshold"][node]
            node = tree["left"][node] if go_left else tree["right"][node]
        total += tree["p"][node]
    return total / len(model["trees"])
