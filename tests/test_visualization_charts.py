"""Tests for drawing pipeline plot tables as Plotly charts in the chat."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app
from agents.code import _visualization_figure, get_pipeline_visualization
from app import process_responses_stream, result_for_model
from tests.test_pipeline_results_run_lookup import BASE, OBJECTS, PIPELINE, RUN_ID, _call, _kfp, _minio


def _page(title, header, rows):
    """Same layout as table_page() in kubeflow/smart_cities/auto_correction_pipeline.py."""
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>'
            f"<body><h3>{title}</h3><table><tr>{head}</tr>{body}</table><p>note</p></body></html>")


CONFUSION_PAGE = _page(
    "Random Forest - confusion matrix (test set, rows = actual)",
    ["actual \\ predicted", "needs operator", "auto-correctable"],
    [["needs operator", 183, 1], ["auto-correctable", 0, 182]],
)
ROC_PAGE = _page(
    "Random Forest - ROC curve (AUC 1.0000)",
    ["threshold", "false_positive_rate", "true_positive_rate"],
    [["1.000", "0.000", "0.000"], ["0.500", "0.005", "1.000"], ["0.000", "1.000", "1.000"]],
)
FEATURE_PAGE = _page(
    "Random Forest - feature importance",
    ["feature", "weight"],
    [["levenshtein", "0.4100"], ["char_multiset_equal", "0.2200"]],
)


def test_confusion_matrix_page_becomes_heatmap():
    fig = _visualization_figure(CONFUSION_PAGE)
    heatmap = fig.data[0]
    assert heatmap.type == "heatmap"
    assert [list(r) for r in heatmap.z] == [[183, 1], [0, 182]]
    assert list(heatmap.x) == ["needs operator", "auto-correctable"]
    assert list(heatmap.y) == ["needs operator", "auto-correctable"]
    assert fig.layout.xaxis.title.text == "Predicted"
    assert fig.layout.yaxis.title.text == "Actual"
    assert fig.layout.title.text.startswith("Random Forest - confusion matrix")


def test_roc_page_becomes_curve_with_diagonal():
    fig = _visualization_figure(ROC_PAGE)
    assert [t.type for t in fig.data] == ["scatter", "scatter"]
    assert list(fig.data[0].x) == [0.0, 0.005, 1.0]
    assert list(fig.data[0].y) == [0.0, 1.0, 1.0]


def test_feature_page_becomes_bar_chart_most_important_on_top():
    fig = _visualization_figure(FEATURE_PAGE)
    bar = fig.data[0]
    assert bar.type == "bar" and bar.orientation == "h"
    assert list(bar.y) == ["levenshtein", "char_multiset_equal"]
    assert fig.layout.yaxis.autorange == "reversed"


def test_unrecognised_pages_give_no_chart():
    assert _visualization_figure("<html><body><img src='plot.png'></body></html>") is None
    assert _visualization_figure(_page("Other", ["a", "b"], [["x", "y"]])) is None
    assert _visualization_figure(_page("Bad", ["feature", "weight"], [["f", "not-a-number"]])) is None


def test_visualization_tool_returns_figure_for_table_pages():
    objects = {**OBJECTS, f"{BASE}/plots/random_forest_confusion_matrix.html": CONFUSION_PAGE.encode()}
    result = _call(
        get_pipeline_visualization, _minio(objects), _kfp(), bucket_name="smart-cities-data",
        pipeline_name=PIPELINE, run_name=RUN_ID, visualization_type="confusion_matrix", model_name="random_forest",
    )
    assert json.loads(result["figure_json"])["data"][0]["type"] == "heatmap"
    assert result["title"].startswith("Random Forest - confusion matrix")


def test_result_for_model_drops_figure_and_flags_chart():
    result = {"html_content": "<table/>", "figure_json": "{}", "title": "t"}
    trimmed = result_for_model("get_pipeline_visualization", result)
    assert "figure_json" not in trimmed
    assert trimmed["chart_shown_to_user"] is True
    assert trimmed["html_content"] == "<table/>"
    assert "figure_json" in result  # the original still feeds the chart


def test_result_for_model_leaves_other_tools_alone():
    result = {"figure_json": "{}", "success": True}
    assert result_for_model("plot_data", result) is result
    assert result_for_model("get_docs", "docs") == "docs"


class FakeMessage:
    def __init__(self, content="", **kwargs):
        self.content = content
        self.elements = []
        self.streaming = False

    async def stream_token(self, token, is_sequence=False):
        self.content = token if is_sequence else self.content + (token or "")

    async def send(self):
        return self

    async def update(self):
        return self


async def _visualization_call_stream():
    item = SimpleNamespace(
        type="function_call", id="item_1", call_id="call_1", name="get_pipeline_visualization",
        arguments='{"bucket_name": "b", "pipeline_name": "p", "run_name": "r", "visualization_type": "confusion_matrix"}',
    )
    yield SimpleNamespace(type="response.output_item.added", item=item)


async def _text_stream():
    yield SimpleNamespace(type="response.output_text.delta", delta="Here is the matrix.")


class TestChartRendering(unittest.IsolatedAsyncioTestCase):
    async def test_visualization_chart_shown_and_kept_out_of_history(self):
        figure_json = _visualization_figure(CONFUSION_PAGE).to_json()

        async def fake_visualization(**kwargs):
            return {"html_content": CONFUSION_PAGE, "figure_json": figure_json, "title": "RF confusion matrix",
                    "visualization_type": "confusion_matrix"}

        async def fake_create_response_stream(message_history):
            return _text_stream()

        plotly = MagicMock(side_effect=lambda **kw: SimpleNamespace(**kw))
        history = []
        with (
            patch.object(app, "create_response_stream", fake_create_response_stream),
            patch.object(app, "function_map", {"get_pipeline_visualization": fake_visualization}),
            patch.object(app, "is_tool_enabled", lambda name, packs: True),
            patch.object(app, "get_enabled_pack_ids", lambda: []),
            patch.object(app, "UserSessionManager", MagicMock()),
            patch.object(app.cl, "Plotly", plotly),
            patch.object(app.cl, "Message", FakeMessage),
        ):
            await process_responses_stream(_visualization_call_stream(), history, FakeMessage())

        plotly.assert_called_once()
        assert plotly.call_args.kwargs["name"] == "confusion_matrix_RF confusion matrix"
        assert plotly.call_args.kwargs["figure"].data[0].type == "heatmap"
        output = next(i for i in history if i.get("type") == "function_call_output")
        model_view = json.loads(output["output"])
        assert "figure_json" not in model_view
        assert model_view["chart_shown_to_user"] is True
