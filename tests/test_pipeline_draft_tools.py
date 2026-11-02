"""Tests for the pipeline builder tools (create_pipeline_draft / upload_pipeline_draft) and attachments."""

import asyncio
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.code import MAX_PIPELINE_DRAFTS, create_pipeline_draft, upload_pipeline_draft

REFERENCE_YAML = Path(__file__).resolve().parents[1] / "kubeflow" / "smart_cities" / "auto_correction_pipeline.yaml"

SOURCE = '''from kfp import dsl


@dsl.component(base_image="python:3.11-slim")
def hello(name: str) -> str:
    return "hello " + name


@dsl.pipeline(name="hello-pipeline", description="Says hello")
def hello_pipeline(name: str = "world"):
    hello(name=name)
'''


class FakeSession:
    """Stand-in for the pipeline-related UserSessionManager state."""

    def __init__(self, attachments=None, drafts=None):
        self.attachments = attachments or {}
        self.drafts = drafts or {}

    def patch_into(self, stack: ExitStack) -> None:
        base = "agents.code.UserSessionManager."
        stack.enter_context(patch(base + "get_pipeline_attachments", side_effect=lambda: self.attachments))
        stack.enter_context(patch(base + "get_pipeline_drafts", side_effect=lambda: self.drafts))
        stack.enter_context(patch(base + "set_pipeline_drafts", side_effect=lambda d: setattr(self, "drafts", d)))


def _create(session, **kwargs):
    message = MagicMock()
    message.return_value.send = AsyncMock()
    with ExitStack() as stack:
        session.patch_into(stack)
        stack.enter_context(patch("agents.code.cl.Message", message))
        stack.enter_context(patch("agents.code.cl.File", MagicMock()))
        result = asyncio.run(create_pipeline_draft.__wrapped__(**kwargs))
    return result, message


def _draft(name="hello-pipeline"):
    return {"pipeline_name": name, "description": "Says hello", "yaml": "pipelineInfo: {}\n", "summary": {}}


def _upload(session, client, choice="register", namespace_result=("alice", None), **kwargs):
    ask = MagicMock()
    ask.return_value.send = AsyncMock(return_value={"payload": {"choice": choice}} if choice else None)
    with ExitStack() as stack:
        session.patch_into(stack)
        resolve = stack.enter_context(
            patch("agents.code.resolve_kubeflow_namespace", return_value=namespace_result)
        )
        stack.enter_context(patch("agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=client)))
        stack.enter_context(patch("agents.code.cl.AskActionMessage", ask))
        message = stack.enter_context(patch("agents.code.cl.Message"))
        message.return_value.send = AsyncMock()
        stack.enter_context(patch("agents.code.kubeflow_dashboard_root", return_value="https://kf.example"))
        result = asyncio.run(upload_pipeline_draft.__wrapped__(**kwargs))
    return result, ask, resolve, message


def _client(existing=None):
    client = MagicMock()
    client.list_pipelines.return_value = SimpleNamespace(
        pipelines=[SimpleNamespace(pipeline_id=existing)] if existing else []
    )
    client.upload_pipeline.return_value = SimpleNamespace(pipeline_id="pipe-new")
    client.upload_pipeline_version.return_value = SimpleNamespace(pipeline_version_id="ver-2", display_name="v2")
    client.list_pipeline_versions.return_value = SimpleNamespace(
        pipeline_versions=[SimpleNamespace(pipeline_version_id="ver-1", display_name="hello-pipeline")]
    )
    return client


# -- create_pipeline_draft ----------------------------------------------------

def test_draft_from_source_compiles_and_stores():
    session = FakeSession()
    result, message = _create(session, source_code=SOURCE)

    assert result["status"] == "draft"
    assert result["pipeline_name"] == "hello-pipeline"
    assert set(result["summary"]["inputs"]) == {"name"}
    draft = session.drafts[result["draft_id"]]
    assert draft["origin"] == "generated"
    assert "hello-pipeline" in draft["yaml"]
    message.return_value.send.assert_awaited_once()


def test_draft_requires_exactly_one_source():
    for kwargs in ({}, {"source_code": SOURCE, "attachment_name": "a.py"}):
        result, _ = _create(FakeSession(), **kwargs)
        assert "exactly one" in result["error"]


def test_draft_returns_validation_error_without_storing():
    session = FakeSession()
    result, message = _create(session, source_code="import os\n" + SOURCE)
    assert "rejected" in result["error"]
    assert session.drafts == {}
    message.return_value.send.assert_not_called()


def test_draft_from_yaml_attachment(tmp_path):
    attached = tmp_path / "pipeline.yaml"
    attached.write_text(REFERENCE_YAML.read_text())
    session = FakeSession(attachments={"pipeline.yaml": str(attached)})

    result, _ = _create(session, attachment_name="pipeline.yaml")

    assert result["pipeline_name"] == "smart-cities-auto-correction-classifier"
    assert session.drafts[result["draft_id"]]["origin"] == "attachment"
    assert session.drafts[result["draft_id"]]["source_code"] is None


def test_draft_unknown_attachment_lists_available():
    session = FakeSession(attachments={"real.py": "/tmp/real.py"})
    result, _ = _create(session, attachment_name="missing.py")
    assert result["attached_files"] == ["real.py"]


def test_drafts_are_capped():
    session = FakeSession(drafts={f"old{i}": _draft() for i in range(MAX_PIPELINE_DRAFTS)})
    result, _ = _create(session, source_code=SOURCE)
    assert len(session.drafts) == MAX_PIPELINE_DRAFTS
    assert "old0" not in session.drafts
    assert result["draft_id"] in session.drafts


# -- upload_pipeline_draft ----------------------------------------------------

def test_upload_new_pipeline_into_private_namespace():
    client = _client()
    result, ask, _, message = _upload(FakeSession(drafts={"d1": _draft()}), client, draft_id="d1")

    assert result["status"] == "new_pipeline"
    posted = message.call_args.kwargs["content"]
    assert "`pipe-new`" in posted and "`ver-1`" in posted
    assert result["pipeline_id"] == "pipe-new"
    assert result["version_id"] == "ver-1"
    assert result["namespace"] == "alice"
    assert result["kubeflow_ui_url"].endswith("?ns=alice#/pipelines/details/pipe-new")
    kwargs = client.upload_pipeline.call_args.kwargs
    assert kwargs["namespace"] == "alice"
    assert kwargs["pipeline_name"] == "hello-pipeline"
    assert client.list_pipelines.call_args.kwargs["namespace"] == "alice"
    ask.return_value.send.assert_awaited_once()


def test_upload_existing_pipeline_adds_version():
    client = _client(existing="pipe-old")
    result, _, _, _ = _upload(FakeSession(drafts={"d1": _draft()}), client, draft_id="d1")

    assert result["status"] == "new_version"
    assert result["pipeline_id"] == "pipe-old"
    assert result["version_id"] == "ver-2"
    client.upload_pipeline.assert_not_called()
    assert client.upload_pipeline_version.call_args.kwargs["pipeline_id"] == "pipe-old"


@pytest.mark.parametrize("choice", ["cancel", None])
def test_upload_without_confirmation_registers_nothing(choice):
    client = _client()
    result, _, _, message = _upload(FakeSession(drafts={"d1": _draft()}), client, choice=choice, draft_id="d1")
    assert result["status"] == "cancelled"
    message.assert_not_called()
    client.upload_pipeline.assert_not_called()
    client.upload_pipeline_version.assert_not_called()


def test_upload_rejects_inaccessible_namespace():
    client = _client()
    result, ask, resolve, _ = _upload(
        FakeSession(drafts={"d1": _draft()}),
        client,
        namespace_result=(None, "You do not have access to Kubeflow namespace 'other'."),
        draft_id="d1",
        namespace="other",
    )
    assert "do not have access" in result["error"]
    resolve.assert_called_once_with("other")
    ask.assert_not_called()
    client.upload_pipeline.assert_not_called()


def test_upload_unknown_draft():
    result, ask, _, _ = _upload(FakeSession(drafts={"d1": _draft()}), _client(), draft_id="nope")
    assert result["draft_ids"] == ["d1"]
    ask.assert_not_called()


# -- chat attachments ---------------------------------------------------------

def test_register_pipeline_attachments(tmp_path):
    from app import register_pipeline_attachments

    small = tmp_path / "pipe.py"
    small.write_text("x = 1\n")
    big = tmp_path / "big.yaml"
    big.write_text("#" * 300_000)
    image = tmp_path / "plot.png"
    image.write_bytes(b"png")
    message = SimpleNamespace(elements=[
        SimpleNamespace(name="pipe.py", path=str(small)),
        SimpleNamespace(name="big.yaml", path=str(big)),
        SimpleNamespace(name="plot.png", path=str(image)),
    ])
    stored = {}
    with patch("app.UserSessionManager.get_pipeline_attachments", return_value={}), \
            patch("app.UserSessionManager.set_pipeline_attachments", side_effect=stored.update):
        note = register_pipeline_attachments(message)

    assert stored == {"pipe.py": str(small)}
    assert "[Attached pipeline files: pipe.py]" in note
    assert "big.yaml" in note and "Ignored" in note
    assert "plot.png" not in note


def test_register_pipeline_attachments_without_files():
    from app import register_pipeline_attachments

    with patch("app.UserSessionManager.get_pipeline_attachments", return_value={}), \
            patch("app.UserSessionManager.set_pipeline_attachments") as setter:
        assert register_pipeline_attachments(SimpleNamespace(elements=None)) == ""
    setter.assert_not_called()
