"""Tests for resolving pipeline result folders by Kubeflow run ID or run display name."""

import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agents.code import compare_pipeline_runs, get_model_metrics, get_pipeline_visualization

PIPELINE = "smart-cities-auto-correction-classifier"
RUN_ID = "8dfd7810-c2a9-4037-a5bd-7c5fe5fb70ed"
RUN_DISPLAY_NAME = "smart-cities-auto-correction-retest-2026-11-02"
BASE = f"kubeflow/{PIPELINE}/{RUN_ID}"

OBJECTS = {
    f"{BASE}/plots/random_forest_confusion_matrix.html": b"<table>rf</table>",
    f"{BASE}/plots/logistic_regression_confusion_matrix.html": b"<table>lr</table>",
    f"{BASE}/metrics/random_forest_metrics.json": json.dumps(
        {"model_name": "Random Forest", "model_type": "random_forest", "accuracy": 0.997}
    ).encode(),
    f"{BASE}/metrics/logistic_regression_metrics.json": json.dumps(
        {"model_name": "Logistic Regression", "model_type": "logistic_regression", "accuracy": 0.937}
    ).encode(),
}


def _minio(objects=None):
    objects = OBJECTS if objects is None else objects
    client = MagicMock()
    client.bucket_exists.return_value = True

    def list_objects(bucket, prefix="", recursive=False):
        if not recursive:
            # Top-level folders under the prefix
            names = {name[len(prefix):].split("/")[0] for name in objects if name.startswith(prefix)}
            return [SimpleNamespace(object_name=f"{prefix}{n}/", is_dir=True) for n in sorted(names)]
        return [SimpleNamespace(object_name=name) for name in sorted(objects) if name.startswith(prefix)]

    def get_object(bucket, name):
        if name not in objects:
            raise KeyError(name)
        response = io.BytesIO(objects[name])
        response.release_conn = lambda: None
        return response

    client.list_objects.side_effect = list_objects
    client.get_object.side_effect = get_object
    return client


def _kfp(run_id=RUN_ID):
    client = MagicMock()
    client.list_runs.return_value = SimpleNamespace(runs=[SimpleNamespace(run_id=run_id)] if run_id else None)
    return client


def _call(tool, minio, kfp, **kwargs):
    with patch("agents.code.get_user_minio_client", new=AsyncMock(return_value=minio)), patch(
        "agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=kfp)
    ), patch("agents.code.resolve_kubeflow_namespace", return_value=("smart-cities", None)):
        return asyncio.run(tool.__wrapped__(**kwargs))


def test_visualization_by_run_id_skips_kubeflow_lookup():
    kfp = _kfp()
    result = _call(
        get_pipeline_visualization, _minio(), kfp, bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_name=RUN_ID, visualization_type="confusion_matrix", model_name="random_forest",
    )
    assert result["html_content"] == "<table>rf</table>"
    kfp.list_runs.assert_not_called()


def test_visualization_by_display_name_resolves_run_id():
    kfp = _kfp()
    result = _call(
        get_pipeline_visualization, _minio(), kfp, bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_name=RUN_DISPLAY_NAME, visualization_type="confusion_matrix", model_name="random_forest",
    )
    assert result["run_name"] == RUN_ID
    assert result["path"] == f"{BASE}/plots/random_forest_confusion_matrix.html"
    name_filter = json.loads(kfp.list_runs.call_args.kwargs["filter"])
    assert name_filter["predicates"][0]["stringValue"] == RUN_DISPLAY_NAME
    assert kfp.list_runs.call_args.kwargs["namespace"] == "smart-cities"


def test_visualization_accepts_model_name_with_spaces():
    result = _call(
        get_pipeline_visualization, _minio(), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_name=RUN_ID, visualization_type="confusion_matrix", model_name="Logistic Regression",
    )
    assert result["html_content"] == "<table>lr</table>"


def test_unknown_run_lists_available_run_folders():
    result = _call(
        get_pipeline_visualization, _minio(), _kfp(run_id=None), bucket_name="smart-cities-data",
        pipeline_name=PIPELINE, run_name="no-such-run", visualization_type="roc_curve",
    )
    assert "run_name must be the Kubeflow run_id" in result["error"]
    assert RUN_ID in result["error"]


def test_metrics_by_display_name_resolves_run_id():
    result = _call(
        get_model_metrics, _minio(), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_name=RUN_DISPLAY_NAME,
    )
    assert result["metrics_count"] == 2
    assert {m["accuracy"] for m in result["metrics"]} == {0.997, 0.937}


def test_compare_by_display_name_resolves_run_id():
    result = _call(
        compare_pipeline_runs, _minio(), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_names=[RUN_DISPLAY_NAME],
    )
    assert "error" not in result, result
    assert RUN_ID in json.dumps(result)


SUMMARY = json.dumps({"hyperparameters": {"rf_max_depth": 8, "lr_c": 1.0}}).encode()
PARAMETERS = json.dumps({"parameters": {"rf_max_depth": 12, "train_data": "s3://b/train.csv"}}).encode()


def test_metrics_fall_back_to_run_summary_hyperparameters():
    objects = {**OBJECTS, f"{BASE}/metadata/run_summary.json": SUMMARY}
    result = _call(
        get_model_metrics, _minio(objects), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_name=RUN_ID,
    )
    assert all(m["parameters"] == {"rf_max_depth": 8, "lr_c": 1.0} for m in result["metrics"])


def test_run_parameters_json_wins_over_run_summary():
    objects = {
        **OBJECTS,
        f"{BASE}/metadata/run_summary.json": SUMMARY,
        f"{BASE}/metadata/run_parameters.json": PARAMETERS,
    }
    result = _call(
        compare_pipeline_runs, _minio(objects), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_names=[RUN_ID],
    )
    assert '"rf_max_depth": 12' in json.dumps(result)
    assert '"rf_max_depth": 8' not in json.dumps(result)


def test_metrics_without_metadata_have_no_parameters():
    result = _call(
        get_model_metrics, _minio(), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_name=RUN_ID,
    )
    assert all("parameters" not in m for m in result["metrics"])


def test_metrics_model_filter_accepts_display_name_or_type():
    for name in ("Random Forest", "random_forest", "random-forest"):
        result = _call(
            get_model_metrics, _minio(), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
            run_name=RUN_ID, model_name=name,
        )
        assert [m["accuracy"] for m in result["metrics"]] == [0.997], name


def test_compare_ranks_only_numeric_metrics():
    other = "088434dd-ca9e-4d5b-802f-6655c2b1c811"
    rf = {"model_name": "Random Forest", "model_type": "random_forest", "outcomes": {"x": 1}}
    objects = {
        f"{BASE}/metrics/random_forest_metrics.json": json.dumps({**rf, "accuracy": 0.99}).encode(),
        f"kubeflow/{PIPELINE}/{other}/metrics/random_forest_metrics.json": json.dumps({**rf, "accuracy": 0.97}).encode(),
    }
    result = _call(
        compare_pipeline_runs, _minio(objects), _kfp(), bucket_name="smart-cities-data", pipeline_name=PIPELINE,
        run_names=[RUN_ID, other],
    )
    best = result["comparison"]["Random Forest"]["best_runs"]
    assert best["accuracy"] == {"run_name": RUN_ID, "value": 0.99}
    assert "model_type" not in best and "outcomes" not in best
