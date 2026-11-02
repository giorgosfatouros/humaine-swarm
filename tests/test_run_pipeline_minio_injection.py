"""Tests for MinIO session credential injection in run_pipeline."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.code import (
    INJECTED_PARAM_PLACEHOLDER,
    _pipeline_input_names,
    _platform_minio_endpoint,
    get_run_details,
    run_pipeline,
)

SESSION_CREDS = {
    "access_key": "STS-ACCESS",
    "secret_key": "STS-SECRET",
    "session_token": "STS-TOKEN",
}


def _spec(*names):
    return {
        "root": {
            "inputDefinitions": {
                "parameters": {name: {"parameterType": "STRING"} for name in names}
            }
        }
    }


def _client(spec):
    client = MagicMock()
    client.get_pipeline_version.return_value = SimpleNamespace(pipeline_spec=spec)
    client.list_pipeline_versions.return_value = SimpleNamespace(
        pipeline_versions=[SimpleNamespace(pipeline_version_id="ver-latest", pipeline_spec=spec)]
    )
    client.run_pipeline.return_value = SimpleNamespace(
        run_id="run-1",
        display_name="job",
        state="PENDING",
        created_at="2026-11-01",
    )
    return client


def _run(client, creds=SESSION_CREDS, **kwargs):
    kwargs.setdefault("experiment_id", "exp-1")
    kwargs.setdefault("job_name", "job")
    kwargs.setdefault("pipeline_id", "pipe-1")
    kwargs.setdefault("version_id", "ver-1")
    get_creds = AsyncMock(return_value=creds)
    with patch(
        "agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=client)
    ), patch(
        "agents.code.UserSessionManager.get_or_refresh_minio_credentials", new=get_creds
    ):
        result = asyncio.run(run_pipeline.__wrapped__(**kwargs))
    return result, get_creds


MINIO_SPEC = _spec(
    "minio_endpoint", "minio_access_key", "minio_secret_key", "minio_session_token", "max_depth"
)


def test_injects_session_credentials_when_spec_declares_them():
    client = _client(MINIO_SPEC)

    result, _ = _run(client, params={"max_depth": 5})

    sent = client.run_pipeline.call_args.kwargs["params"]
    assert sent == {
        "max_depth": 5,
        "minio_endpoint": _platform_minio_endpoint(),
        "minio_access_key": "STS-ACCESS",
        "minio_secret_key": "STS-SECRET",
        "minio_session_token": "STS-TOKEN",
    }
    assert result["run_id"] == "run-1"


def test_foreign_minio_endpoint_is_rejected_when_injecting():
    client = _client(MINIO_SPEC)

    result, get_creds = _run(client, params={"minio_endpoint": "https://attacker.example"})

    assert "platform MinIO" in result["error"]
    client.run_pipeline.assert_not_called()
    get_creds.assert_not_awaited()


def test_platform_endpoint_in_any_spelling_is_accepted():
    client = _client(MINIO_SPEC)
    host = _platform_minio_endpoint().split("://")[-1]

    result, _ = _run(client, params={"minio_endpoint": f"https://{host.upper()}/"})

    assert "error" not in result
    assert client.run_pipeline.call_args.kwargs["params"]["minio_endpoint"] == _platform_minio_endpoint()


def test_nothing_injected_when_spec_does_not_declare_them():
    client = _client(_spec("max_depth"))

    result, get_creds = _run(client, params={"max_depth": 5})

    assert client.run_pipeline.call_args.kwargs["params"] == {"max_depth": 5}
    assert result["params"] == {"max_depth": 5}
    get_creds.assert_not_awaited()


def test_caller_values_are_kept_and_empty_ones_filled():
    client = _client(MINIO_SPEC)

    result, _ = _run(
        client,
        params={"minio_access_key": "MY-KEY", "minio_secret_key": "MY-SECRET", "minio_session_token": ""},
    )

    sent = client.run_pipeline.call_args.kwargs["params"]
    assert sent["minio_access_key"] == "MY-KEY"
    assert sent["minio_secret_key"] == "MY-SECRET"
    assert sent["minio_session_token"] == "STS-TOKEN"
    assert result["params"]["minio_access_key"] == "MY-KEY"
    assert result["params"]["minio_session_token"] == INJECTED_PARAM_PLACEHOLDER


def test_all_caller_values_supplied_skips_spec_lookup():
    client = _client(MINIO_SPEC)
    params = {"minio_access_key": "A", "minio_secret_key": "S", "minio_session_token": "T"}

    _, get_creds = _run(client, params=params)

    assert client.run_pipeline.call_args.kwargs["params"] == params
    client.get_pipeline_version.assert_not_called()
    get_creds.assert_not_awaited()


def test_injected_secrets_are_redacted_from_return_and_logs(caplog):
    client = _client(MINIO_SPEC)

    with caplog.at_level("DEBUG"):
        result, _ = _run(client, params={"max_depth": 5})

    for name in ("minio_access_key", "minio_secret_key", "minio_session_token"):
        assert result["params"][name] == INJECTED_PARAM_PLACEHOLDER
    dumped = json.dumps(result) + caplog.text
    for secret in SESSION_CREDS.values():
        assert secret not in dumped


def test_error_when_session_has_no_credentials():
    client = _client(MINIO_SPEC)

    result, _ = _run(client, creds=None, params={})

    assert "error" in result
    assert "MinIO credentials" in result["error"]
    client.run_pipeline.assert_not_called()


def test_latest_version_used_when_version_id_missing():
    client = _client(MINIO_SPEC)

    result, _ = _run(client, version_id=None, params={})

    client.list_pipeline_versions.assert_called_once_with(
        pipeline_id="pipe-1", sort_by="created_at desc", page_size=1
    )
    sent = client.run_pipeline.call_args.kwargs
    assert sent["version_id"] == "ver-latest"
    assert sent["params"]["minio_session_token"] == "STS-TOKEN"
    assert result["pipeline_version_id"] == "ver-latest"


def test_explicit_version_id_is_kept():
    client = _client(MINIO_SPEC)

    _run(client, version_id="ver-1", params={})

    client.list_pipeline_versions.assert_not_called()
    assert client.run_pipeline.call_args.kwargs["version_id"] == "ver-1"


def test_error_when_pipeline_has_no_versions():
    client = _client(MINIO_SPEC)
    client.list_pipeline_versions.return_value = SimpleNamespace(pipeline_versions=[])

    result, _ = _run(client, version_id=None, params={})

    assert "no versions" in result["error"]
    client.run_pipeline.assert_not_called()


def test_pipeline_input_names_uses_latest_version_without_version_id():
    client = _client(MINIO_SPEC)

    assert "minio_session_token" in _pipeline_input_names(client, "pipe-1", None)
    client.list_pipeline_versions.assert_called_once()
    client.get_pipeline_version.assert_not_called()


def test_package_path_runs_are_not_touched():
    client = _client(MINIO_SPEC)

    _, get_creds = _run(
        client, pipeline_id=None, version_id=None, pipeline_package_path="/tmp/p.yaml", params={}
    )

    assert client.run_pipeline.call_args.kwargs["params"] == {}
    client.list_pipeline_versions.assert_not_called()
    get_creds.assert_not_awaited()


@pytest.mark.parametrize(
    "spec",
    [
        json.dumps(MINIO_SPEC),
        "root:\n  inputDefinitions:\n    parameters:\n      minio_session_token:\n        parameterType: STRING\n",
        {"pipeline_spec": MINIO_SPEC, "platform_spec": {}},
    ],
)
def test_pipeline_input_names_parses_spec_formats(spec):
    assert "minio_session_token" in _pipeline_input_names(_client(spec), "pipe-1", "ver-1")


def test_pipeline_input_names_returns_empty_set_on_error():
    client = MagicMock()
    client.get_pipeline_version.side_effect = RuntimeError("boom")
    assert _pipeline_input_names(client, "pipe-1", "ver-1") == set()
    assert _pipeline_input_names(_client("not: [valid"), "pipe-1", "ver-1") == set()


def _run_with_tasks(parameters, tasks):
    return SimpleNamespace(
        run_id="run-1",
        display_name="job",
        state="FAILED",
        created_at="2026-11-01",
        finished_at=None,
        runtime_config=SimpleNamespace(parameters=parameters),
        error=SimpleNamespace(code=9, message="run failed"),
        run_details=SimpleNamespace(task_details=tasks),
    )


def test_get_run_details_redacts_credentials_and_lists_failed_tasks():
    tasks = [
        SimpleNamespace(display_name="load-data", state="SUCCEEDED", error=None, pod_name="p1"),
        SimpleNamespace(
            display_name="build-features",
            state="FAILED",
            error=SimpleNamespace(message="ImagePullBackOff"),
            pod_name="p2",
        ),
    ]
    run = _run_with_tasks(
        {"minio_access_key": "STS-ACCESS", "minio_session_token": "STS-TOKEN", "minio_secret_key": "", "max_depth": 5},
        tasks,
    )
    client = MagicMock()
    client.get_run.return_value = run

    with patch("agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=client)):
        result = asyncio.run(get_run_details.__wrapped__(run_id="run-1"))

    params = result["runtime_config"]["parameters"]
    assert params == {
        "minio_access_key": "<redacted>",
        "minio_session_token": "<redacted>",
        "minio_secret_key": "",
        "max_depth": 5,
    }
    assert "STS-" not in json.dumps(result)
    assert result["failed_tasks"] == [
        {"task": "build-features", "state": "FAILED", "error": "ImagePullBackOff", "pod_name": "p2"}
    ]


def test_get_run_details_without_task_details():
    run = _run_with_tasks({}, None)
    client = MagicMock()
    client.get_run.return_value = run

    with patch("agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=client)):
        result = asyncio.run(get_run_details.__wrapped__(run_id="run-1"))

    assert result["failed_tasks"] == []
