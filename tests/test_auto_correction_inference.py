"""Tests for predict_auto_correction and the portable Smart Cities models it evaluates."""

import asyncio
import copy
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from agents.code import predict_auto_correction
from utils.auto_correction_model import (
    FEATURE_NAMES,
    MAX_TREE_DEPTH,
    MAX_TREES,
    featurize,
    predict_probability,
    validate_portable_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kubeflow" / "smart_cities"))
import auto_correction_pipeline as pipeline  # noqa: E402

PIPELINE = "smart-cities-auto-correction-classifier"
LEVENSHTEIN = FEATURE_NAMES.index("levenshtein")

ROWS = pd.DataFrame({
    "supplied_value": ["ΠΑΠΑΔΟΠΟΥΛΟΣ", "Γιώργος", "12345", "ABC-1234", "Οδός Ερμού 5", "6912345678", "ΑΒ123456"],
    "ground_truth": ["ΠΑΠΑΔΟΠΟΥΛΟΥ", "ΓΕΩΡΓΙΟΣ", "12354", "ABC 1234", "Ερμού 5 Οδός", "6912345679", "ΑΒ654321"],
    "field_type": ["surname", "name", "vat_number", "licence_plate", "address", "mobile_phone", "id_number"],
    "is_obvious": [1, 0, 1, 1, 1, 0, 0],
})


def _artifact(path):
    return SimpleNamespace(path=path, metadata={})


def test_featurize_matches_the_pipeline_step():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "rows.csv")
        ROWS.to_csv(src, index=False)
        out = _artifact(os.path.join(tmp, "features.csv"))
        pipeline.build_features.python_func(
            train_data=_artifact(src), test_data=_artifact(src),
            train_features=out, test_features=_artifact(os.path.join(tmp, "unused.csv")),
        )
        expected = pd.read_csv(out.path)

    mine = pd.DataFrame([featurize(ft, s, g) for s, g, ft in
                         zip(ROWS.supplied_value, ROWS.ground_truth, ROWS.field_type)])
    assert list(mine.columns) == FEATURE_NAMES
    pd.testing.assert_frame_equal(mine, expected[FEATURE_NAMES], check_dtype=False)


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_portable_model_matches_scikit_learn(model_type):
    pytest.importorskip("sklearn")
    import joblib

    rows = pd.concat([ROWS] * 6, ignore_index=True)
    feats = pd.DataFrame([featurize(ft, s, g) for s, g, ft in
                          zip(rows.supplied_value, rows.ground_truth, rows.field_type)])
    feats["field_type"], feats["label"] = rows.field_type, rows.is_obvious
    with tempfile.TemporaryDirectory() as tmp:
        train = _artifact(os.path.join(tmp, "train.csv"))
        feats.to_csv(train.path, index=False)
        model, portable = _artifact(os.path.join(tmp, "model")), _artifact(os.path.join(tmp, "model.json"))
        pipeline.train_model.python_func(
            train_features=train, model_type=model_type, lr_c=1.0, rf_n_estimators=15,
            rf_max_depth=4, random_state=0, model=model, portable_model=portable,
        )
        clf = joblib.load(model.path)["model"]  # written by train_model a line above, not untrusted
        exported = validate_portable_model(json.load(open(portable.path)))

    expected = clf.predict_proba(feats[FEATURE_NAMES])[:, 1]
    mine = [predict_probability(exported, r) for r in feats[FEATURE_NAMES].to_dict("records")]
    assert max(abs(a - b) for a, b in zip(expected, mine)) < 1e-12


def _lr_model(intercept=0.0):
    n = len(FEATURE_NAMES)
    return {
        "format": "humaine-portable-classifier/1", "model_type": "logistic_regression",
        "model_name": "Logistic Regression", "feature_names": FEATURE_NAMES, "threshold": 0.5,
        "scaler_mean": [0.0] * n, "scaler_scale": [1.0] * n, "coef": [0.0] * n, "intercept": intercept,
    }


def _rf_model():
    """One tree: levenshtein <= 1.5 is auto-correctable (p 0.9), otherwise needs an operator (p 0.1)."""
    return {
        "format": "humaine-portable-classifier/1", "model_type": "random_forest",
        "model_name": "Random Forest", "feature_names": FEATURE_NAMES, "threshold": 0.5,
        "trees": [{"left": [1, -1, -1], "right": [2, -1, -1], "feature": [LEVENSHTEIN, -2, -2],
                   "threshold": [1.5, -2.0, -2.0], "p": [0.5, 0.9, 0.1]}],
    }


def test_predict_probability_for_hand_built_models():
    close, far = featurize("surname", "SMITH", "SMYTH"), featurize("surname", "SMITH", "JONES")
    assert predict_probability(_rf_model(), close) == 0.9
    assert predict_probability(_rf_model(), far) == 0.1
    assert predict_probability(_lr_model(), close) == 0.5
    assert predict_probability(_lr_model(intercept=-800.0), close) == pytest.approx(0.0)


def _tampered(change):
    model = copy.deepcopy(_rf_model())
    change(model)
    return model


def _deep_tree(depth):
    """A chain of splits `depth` levels deep: node 2k splits into leaf 2k+1 and node 2k+2."""
    size = 2 * depth + 1
    left = [i + 1 if i % 2 == 0 and i < size - 1 else -1 for i in range(size)]
    right = [i + 2 if i % 2 == 0 and i < size - 1 else -1 for i in range(size)]
    feature = [LEVENSHTEIN if r != -1 else -2 for r in right]
    model = _rf_model()
    model["trees"] = [{"left": left, "right": right, "feature": feature,
                       "threshold": [0.5] * size, "p": [0.5] * size}]
    return model


def test_depth_limit_accepts_the_deepest_allowed_tree():
    model = validate_portable_model(_deep_tree(MAX_TREE_DEPTH))
    assert predict_probability(model, featurize("surname", "SMITH", "SMITH")) == 0.5


@pytest.mark.parametrize("model", [
    None,
    {"format": "pickle"},
    _tampered(lambda m: m.update(feature_names=FEATURE_NAMES[:-1])),
    _tampered(lambda m: m["trees"][0].update(left=[0, -1, -1])),        # node points back at itself
    _tampered(lambda m: m["trees"][0].update(feature=[99, -2, -2])),   # feature index out of range
    _tampered(lambda m: m["trees"][0].update(p=[0.5, 7.0, 0.1])),      # probability above 1
    _tampered(lambda m: m["trees"][0].update(threshold=[True, -2.0, -2.0])),
    _tampered(lambda m: m.update(trees=[])),
    {**_lr_model(), "scaler_scale": [0.0] * len(FEATURE_NAMES)},
    {**_lr_model(), "intercept": float("inf")},
    _deep_tree(MAX_TREE_DEPTH + 1),
    _tampered(lambda m: m.update(trees=m["trees"] * (MAX_TREES + 1))),
])
def test_validate_rejects_malformed_models(model):
    with pytest.raises(ValueError):
        validate_portable_model(model)


NEW_RUN, OLD_RUN, LEGACY_RUN = "run-new", "run-old", "run-legacy"
FAILED_RUN, PLANTED_RUN, OTHER_PIPELINE_RUN = "run-failed", "run-planted", "run-other-pipeline"
PIPELINE_ID = "pipeline-auto-correction"
# Kubeflow's runs in the user's namespace, newest first. PLANTED_RUN is only a MinIO folder;
# OTHER_PIPELINE_RUN is a real run of another pipeline whose ID was reused as a folder name.
KUBEFLOW_RUNS = [
    (OTHER_PIPELINE_RUN, "SUCCEEDED", "pipeline-other"),
    (FAILED_RUN, "FAILED", PIPELINE_ID),
    (NEW_RUN, "SUCCEEDED", PIPELINE_ID),
    (OLD_RUN, "SUCCEEDED", PIPELINE_ID),
    (LEGACY_RUN, "SUCCEEDED", PIPELINE_ID),
]


def _objects(rf=None):
    def at(run, name):
        return f"kubeflow/{PIPELINE}/{run}/models/{name}"
    return {
        at(PLANTED_RUN, "random_forest.json"): json.dumps(_rf_model()),
        at(OTHER_PIPELINE_RUN, "random_forest.json"): json.dumps(_rf_model()),
        at(FAILED_RUN, "random_forest.json"): json.dumps(_rf_model()),
        at(OLD_RUN, "random_forest.json"): json.dumps(_rf_model()),
        at(NEW_RUN, "random_forest.json"): json.dumps(rf if rf is not None else _rf_model()),
        at(NEW_RUN, "logistic_regression.json"): json.dumps(_lr_model(intercept=2.0)),
        at(NEW_RUN, "random_forest.joblib"): "pickle bytes",
        at(LEGACY_RUN, "random_forest.joblib"): "pickle bytes",
    }


def _minio(objects):
    client = MagicMock()
    client.bucket_exists.return_value = True

    def list_objects(bucket, prefix="", recursive=False):
        return [SimpleNamespace(object_name=name) for name in sorted(objects) if name.startswith(prefix)]

    def get_object(bucket, name):
        response = io.BytesIO(objects[name].encode())
        response.release_conn = lambda: None
        return response

    client.list_objects.side_effect = list_objects
    client.get_object.side_effect = get_object
    return client


def _kfp(runs=KUBEFLOW_RUNS):
    client = MagicMock()
    client.list_runs.return_value = SimpleNamespace(
        runs=[SimpleNamespace(run_id=run_id, state=state,
                              pipeline_version_reference=SimpleNamespace(pipeline_id=pipeline_id))
              for run_id, state, pipeline_id in runs]
    )
    return client


def _predict(minio, kfp=None, **kwargs):
    with patch("agents.code.get_user_minio_client", new=AsyncMock(return_value=minio)), patch(
        "agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=kfp or _kfp())
    ), patch("agents.code.resolve_kubeflow_namespace", return_value=("smart-cities", None)), patch(
        "agents.code.find_pipeline_id", return_value=PIPELINE_ID
    ), patch(
        "agents.code._kubeflow_run_id_for_name", new=AsyncMock(return_value=None)
    ):
        return asyncio.run(predict_auto_correction.__wrapped__(**kwargs))


ENTRIES = [
    {"field_type": "surname", "supplied_value": "Smith", "correct_value": "Smyth"},
    {"field_type": "surname", "supplied_value": "Smith", "correct_value": "Jones"},
]


def test_predicts_with_the_newest_succeeded_run_and_both_models():
    minio, kfp = _minio(_objects()), _kfp()
    result = _predict(minio, kfp, entries=ENTRIES)

    # Not another pipeline's run, not the failed run, not a folder no Kubeflow run produced
    assert result["run_name"] == NEW_RUN
    assert kfp.list_runs.call_args.kwargs["namespace"] == "smart-cities"
    assert sorted(result["models"]) == ["Logistic Regression", "Random Forest"]
    close, far = result["predictions"]
    assert close["predictions"]["Random Forest"] == {"probability_auto_correctable": 0.9, "decision": "auto-correctable"}
    assert far["predictions"]["Random Forest"]["decision"] == "needs operator"
    assert far["predictions"]["Logistic Regression"]["decision"] == "auto-correctable"
    assert close["models_agree"] is True and far["models_agree"] is False
    assert far["features"]["levenshtein"] == 5 and "field_surname" not in far["features"]
    # Only the chosen run's JSON models are read, never the pickles
    read = {c.args[1] for c in minio.get_object.call_args_list}
    assert read and all(name.endswith(".json") and f"/{NEW_RUN}/" in name for name in read)


def test_no_default_run_when_kubeflow_knows_none_with_models():
    runs = [(OTHER_PIPELINE_RUN, "SUCCEEDED", "pipeline-other"), (FAILED_RUN, "FAILED", PIPELINE_ID),
            (LEGACY_RUN, "SUCCEEDED", PIPELINE_ID)]
    result = _predict(_minio(_objects()), _kfp(runs=runs), entries=ENTRIES)
    assert "pass run_name" in result["error"]


def test_model_name_filter_and_explicit_run():
    result = _predict(_minio(_objects()), entries=ENTRIES[:1], run_name=OLD_RUN, model_name="random-forest")
    assert result["run_name"] == OLD_RUN
    assert result["models"] == ["Random Forest"]


def test_run_with_only_pickles_explains_how_to_get_models():
    result = _predict(_minio(_objects()), entries=ENTRIES, run_name=LEGACY_RUN)
    assert "older pipeline version" in result["error"]


def test_tampered_model_is_skipped_not_evaluated():
    broken = _rf_model()
    broken["trees"][0]["left"] = [0, -1, -1]
    result = _predict(_minio(_objects(rf=broken)), entries=ENTRIES)
    assert result["models"] == ["Logistic Regression"]
    assert result["skipped_models"][0]["path"].endswith("random_forest.json")


def test_oversized_model_file_is_refused_by_bytes_read():
    limit = 2000
    objects = _objects()
    rf_path = f"kubeflow/{PIPELINE}/{NEW_RUN}/models/random_forest.json"
    objects[rf_path] += " " * limit  # still valid JSON, but over the limit
    with patch("agents.code.MAX_MODEL_BYTES", limit):
        result = _predict(_minio(objects), entries=ENTRIES)
    assert result["models"] == ["Logistic Regression"]
    assert "limit" in result["skipped_models"][0]["reason"]


def test_pipeline_missing_from_namespace_needs_explicit_run():
    with patch("agents.code.find_pipeline_id", return_value=None):
        minio, kfp = _minio(_objects()), _kfp()
        with patch("agents.code.get_user_minio_client", new=AsyncMock(return_value=minio)), patch(
            "agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=kfp)
        ), patch("agents.code.resolve_kubeflow_namespace", return_value=("smart-cities", None)):
            result = asyncio.run(predict_auto_correction.__wrapped__(entries=ENTRIES))
    assert "was not found" in result["error"]


@pytest.mark.parametrize("entries, message", [
    ([], "non-empty list"),
    ([{"field_type": "email", "supplied_value": "a", "correct_value": "b"}], "field_type must be one of"),
    ([{"field_type": "name", "supplied_value": "a"}], "both supplied_value and correct_value"),
    ([{"field_type": "name", "supplied_value": "a" * 201, "correct_value": "b"}], "200 characters"),
    ([ENTRIES[0]] * 51, "At most 50"),
])
def test_rejects_bad_entries_before_touching_minio(entries, message):
    minio = _minio(_objects())
    result = _predict(minio, entries=entries)
    assert message in result["error"]
    minio.list_objects.assert_not_called()
