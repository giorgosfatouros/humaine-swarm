"""Tests for pipeline source validation, isolated compilation and spec inspection."""

from pathlib import Path
from unittest.mock import patch

import pytest

from utils.pipeline_builder import (
    compile_pipeline_source,
    inspect_pipeline_spec,
    load_pipeline_yaml,
    pipeline_root_parameters,
    validate_pipeline_source,
)

REFERENCE_PIPELINE = Path(__file__).resolve().parents[1] / "kubeflow" / "smart_cities" / "auto_correction_pipeline.py"

HEADER = '''from kfp import dsl
from kfp.dsl import Input, Output, Dataset


@dsl.component(base_image="python:3.11-slim", packages_to_install=["pandas==2.2.3"])
def make_data(rows: int, data: Output[Dataset]):
    import pandas as pd
    pd.DataFrame({"x": range(rows)}).to_csv(data.path, index=False)


@dsl.component(base_image="python:3.11-slim")
def count_rows(data: Input[Dataset]) -> int:
    return sum(1 for _ in open(data.path)) - 1
'''

GOOD_PIPELINE = HEADER + '''

@dsl.pipeline(name="tiny-pipeline", description="Two steps")
def tiny(rows: int = 5, label: str = "demo"):
    made = make_data(rows=rows)
    made.set_caching_options(False)
    with dsl.If(rows > 1):
        count_rows(data=made.outputs["data"])
'''


def _pipeline(body: str) -> str:
    return HEADER + "\n\n@dsl.pipeline\ndef p(rows: int = 5):\n" + body


def test_reference_pipeline_is_accepted():
    assert validate_pipeline_source(REFERENCE_PIPELINE.read_text()) == []


def test_good_pipeline_is_accepted():
    assert validate_pipeline_source(GOOD_PIPELINE) == []


# Hostile sources the validator must reject. They are only parsed, never executed.
@pytest.mark.parametrize(
    "source, expected",
    [
        ("import os\n" + GOOD_PIPELINE, "import of 'os'"),
        ("import requests\n" + GOOD_PIPELINE, "import of 'requests'"),
        ("from subprocess import run\n" + GOOD_PIPELINE, "import from 'subprocess'"),
        (GOOD_PIPELINE + "\nprint('hi')\n", "not allowed at module level"),
        (GOOD_PIPELINE + "\nX = open('/etc/passwd').read()\n", "literal constants"),
        (GOOD_PIPELINE + "\ndef helper():\n    return 1\n", "exactly one decorator"),
        (_pipeline("    open('/etc/passwd')\n"), "calling 'open'"),
        (_pipeline("    eval('1')\n"), "calling 'eval'"),
        (_pipeline("    t = make_data(rows=rows)\n    t.__class__\n"), "dunder"),
        (_pipeline("    make_data(rows=rows).set_display_name('x')\n"), "set_display_name"),
        (_pipeline("    dsl.types.os.system('id')\n"), "dsl.types.os.system"),
        (_pipeline("    import os\n"), "Import is not allowed"),
        (_pipeline("    while True:\n        pass\n"), "While is not allowed"),
        (HEADER, "no @dsl.pipeline"),
        (GOOD_PIPELINE + "\n\n@dsl.pipeline\ndef other():\n    make_data(rows=1)\n", "exactly one @dsl.pipeline"),
        (HEADER.replace('base_image="python:3.11-slim")', "base_image=__import__('os').getcwd())"), "literals"),
        ("from kfp import dsl\nimport json\n@json.pipeline\ndef p():\n    pass\n", "must be kfp's"),
        ("def broken(:\n", "syntax error"),
    ],
)
def test_unsafe_sources_are_rejected(source, expected):
    problems = validate_pipeline_source(source)
    assert problems, "expected the source to be rejected"
    assert any(expected in problem for problem in problems), problems


def test_oversized_source_is_rejected():
    assert "larger than" in validate_pipeline_source("#" * 200_000)[0]


def test_compile_produces_inspectable_spec():
    yaml_text, error = compile_pipeline_source(GOOD_PIPELINE)
    assert error is None
    spec, load_error = load_pipeline_yaml(yaml_text)
    assert load_error is None

    summary = inspect_pipeline_spec(spec)
    assert summary["name"] == "tiny-pipeline"
    assert set(summary["inputs"]) == {"rows", "label"}
    assert summary["inputs"]["label"]["default"] == "demo"
    assert "make-data" in summary["steps"]
    assert summary["images"] == ["python:3.11-slim"]
    assert summary["packages"] == ["pandas==2.2.3"]


def test_compile_never_runs_rejected_source():
    with patch("utils.pipeline_builder.subprocess.run") as run:
        yaml_text, error = compile_pipeline_source("import os\n" + GOOD_PIPELINE)
    assert yaml_text is None
    assert "rejected" in error
    run.assert_not_called()


def test_compile_error_points_at_user_code():
    yaml_text, error = compile_pipeline_source(_pipeline("    make_data(nope=1)\n"))
    assert yaml_text is None
    assert 'File "pipeline.py"' in error
    assert "nope" in error
    assert "site-packages" not in error


def test_compile_subprocess_gets_minimal_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    with patch("utils.pipeline_builder.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stderr = "boom"
        compile_pipeline_source(GOOD_PIPELINE)
    env = run.call_args.kwargs["env"]
    assert set(env) == {"PATH", "HOME", "LANG"}
    assert "-I" in run.call_args.args[0]


def test_compile_timeout_returns_error():
    import subprocess

    with patch("utils.pipeline_builder.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 1)):
        yaml_text, error = compile_pipeline_source(GOOD_PIPELINE, timeout=1)
    assert yaml_text is None
    assert "timed out" in error


def test_load_pipeline_yaml_rejects_non_pipelines():
    assert "Not valid YAML" in load_pipeline_yaml("a: [b")[1]
    assert "missing" in load_pipeline_yaml("pipelineInfo: {name: x}")[1]
    assert "Argo" in load_pipeline_yaml("apiVersion: argoproj.io/v1alpha1\nkind: Workflow\n")[1]


def test_load_pipeline_yaml_accepts_compiled_reference():
    yaml_text = REFERENCE_PIPELINE.with_suffix(".yaml").read_text()
    spec, error = load_pipeline_yaml(yaml_text)
    assert error is None
    assert spec["pipelineInfo"]["name"] == "smart-cities-auto-correction-classifier"


def test_pipeline_root_parameters_unwraps_platform_spec():
    inner = {"root": {"inputDefinitions": {"parameters": {"a": {}, "b": {}}}}}
    assert set(pipeline_root_parameters({"pipeline_spec": inner, "platform_spec": {}})) == {"a", "b"}
    assert pipeline_root_parameters({"root": {}}) == {}
    assert pipeline_root_parameters(None) == {}
