"""Unit tests for HAIC pilot mapping and response projection."""

import json
import unittest
from unittest.mock import MagicMock, patch

from utils.haic_pilot_map import (
    assert_configuration_allowed,
    resolve_haic_pilot_context,
)
from utils.haic_client import (
    _dedupe_warnings,
    _extract_haic_metrics,
    _project_configuration,
    format_haic_result_markdown,
    list_evaluations,
    get_holistic,
    query_haic_benchmark,
)
from utils.haic_routing import detect_haic_live_query, infer_haic_action


class TestHaicPilotMap(unittest.TestCase):
    def test_email_pattern_match(self):
        ctx = resolve_haic_pilot_context(
            user_email="cities-pilot@humaine.com",
            groups=[],
            policies=[],
        )
        self.assertTrue(ctx["has_access"])
        self.assertEqual(ctx["pilot"], "smart_cities")
        self.assertEqual(ctx["configuration_ids"], [3])

    def test_group_match(self):
        ctx = resolve_haic_pilot_context(
            user_email="user@example.com",
            groups=["smart-energy"],
            policies=[],
        )
        self.assertTrue(ctx["has_access"])
        self.assertEqual(ctx["pilot"], "smart_energy")
        self.assertEqual(ctx["configuration_ids"], [6])

    def test_policy_match(self):
        ctx = resolve_haic_pilot_context(
            user_email="user@example.com",
            groups=[],
            policies=["smart-healthcare"],
        )
        self.assertTrue(ctx["has_access"])
        self.assertEqual(ctx["configuration_ids"], [2])

    def test_no_match(self):
        ctx = resolve_haic_pilot_context(
            user_email="random@example.com",
            groups=["developers"],
            policies=["developers"],
        )
        self.assertFalse(ctx["has_access"])
        self.assertEqual(ctx["configuration_ids"], [])

    def test_new_pilot_without_configuration_ids(self):
        ctx = resolve_haic_pilot_context(
            user_email="user@example.com",
            groups=["new-pilot"],
            policies=[],
        )
        self.assertFalse(ctx["has_access"])

    def test_allowlist_rejection(self):
        err = assert_configuration_allowed(99, [3])
        self.assertIsNotNone(err)
        self.assertIn("Access denied", err)

    def test_allowlist_accepts(self):
        self.assertIsNone(assert_configuration_allowed(3, [3]))


class TestHaicProjection(unittest.TestCase):
    def test_dedupe_warnings(self):
        warnings = [
            {"metric": "Tr", "warning": "no events"},
            {"metric": "Tr", "warning": "no events"},
            {"metric": "HCL", "warning": "auto rt_max"},
        ]
        out = _dedupe_warnings(warnings)
        self.assertEqual(len(out), 2)

    def test_extract_haic_from_holistic(self):
        metrics = _extract_haic_metrics({"haic": {"Tr": 0.7, "HCL": 0.58, "F": 1.2}})
        self.assertEqual(metrics["Tr"], 0.7)
        self.assertEqual(metrics["HCL"], 0.58)

    def test_extract_haic_from_interaction(self):
        payload = {"aggregates": {"interaction": {"Tr": 0.61, "D": 10.0}}}
        metrics = _extract_haic_metrics(payload)
        self.assertEqual(metrics["Tr"], 0.61)

    def test_project_configuration_filters_fields(self):
        row = {
            "id": 3,
            "application_name": "Smart Cities",
            "ai_model_name": "Screener",
            "pilot_tag": "applications",
            "evaluation_status": "completed",
            "evaluation_date": "2026-08-30",
            "description": "test",
            "extra_field": "dropped",
        }
        projected = _project_configuration(row)
        self.assertEqual(set(projected.keys()), {
            "id", "application_name", "ai_model_name", "pilot_tag",
            "evaluation_status", "evaluation_date", "description",
        })


class TestHaicRouting(unittest.TestCase):
    def test_find_my_haic_results(self):
        self.assertEqual(
            detect_haic_live_query("can you find my haic results?"),
            "get_holistic",
        )

    def test_evaluations_list(self):
        self.assertEqual(
            detect_haic_live_query("What HAIC evaluations do I have?"),
            "list_evaluations",
        )

    def test_conceptual_not_routed(self):
        self.assertIsNone(detect_haic_live_query("What does HCL mean in the HAIC framework?"))

    def test_minio_style_query_without_haic(self):
        self.assertIsNone(detect_haic_live_query("what data do I have in my buckets?"))

    def test_format_holistic_markdown(self):
        md = format_haic_result_markdown({
            "success": True,
            "action": "get_holistic",
            "haic": {"Tr": 0.692, "HCL": 0.585},
            "per_model_results": [
                {"ai_model_version": "v0", "haic": {"Tr": 0.701, "HCL": 0.539}},
            ],
        })
        self.assertIn("HAIC Benchmark Suite", md)
        self.assertIn("Tr=0.692", md)
        self.assertIn("v0", md)


class TestHaicClientAsync(unittest.IsolatedAsyncioTestCase):
    async def test_list_evaluations_filters_allowlist(self):
        configs = [
            {"id": 3, "application_name": "A", "ai_model_name": "m", "pilot_tag": "p",
             "evaluation_status": "completed", "evaluation_date": "d", "description": None},
            {"id": 99, "application_name": "Other", "ai_model_name": "m", "pilot_tag": "x",
             "evaluation_status": "completed", "evaluation_date": "d", "description": None},
        ]
        with patch("utils.haic_client._aget", return_value=configs):
            result = await list_evaluations([3])
        self.assertTrue(result["success"])
        self.assertEqual(len(result["evaluations"]), 1)
        self.assertEqual(result["evaluations"][0]["id"], 3)

    async def test_query_rejects_foreign_configuration(self):
        result = await query_haic_benchmark(
            action="list_results",
            allowlist=[3],
            configuration_id=99,
        )
        self.assertFalse(result["success"])
        self.assertIn("Access denied", result["error"])

    async def test_get_holistic_projects_compact_payload(self):
        holistic = {
            "configuration_id": 3,
            "haic": {"Tr": 0.69, "HCL": 0.58, "F": 1.0, "D": 90.0},
            "warnings": [{"metric": "Tr", "warning": "dup"}, {"metric": "Tr", "warning": "dup"}],
            "evaluation_date": "2026-08-30",
            "ai_model_version": "v1",
        }
        result_index = [{"id": 15, "ai_model_version": "v0", "configuration_id": 3}]
        full_result = {
            "configuration_id": 3,
            "ai_model_version": "v0",
            "aggregates": {"interaction": {"Tr": 0.7, "HCL": 0.72}},
            "warnings": [{"metric": "A", "warning": "short session"}] * 100,
            "metric_timeseries": [{"t": i} for i in range(500)],
        }

        async def fake_aget(path, params=None):
            if path.endswith("/holistic"):
                return holistic
            if path == "/v1/results/3":
                return result_index
            if path.endswith("/15"):
                return full_result
            raise AssertionError(f"unexpected path {path}")

        with patch("utils.haic_client._aget", side_effect=fake_aget):
            result = await get_holistic(3, [3])

        self.assertTrue(result["success"])
        self.assertEqual(result["haic"]["Tr"], 0.69)
        self.assertEqual(len(result["warnings"]), 1)
        self.assertEqual(len(result["per_model_results"]), 1)
        self.assertEqual(result["per_model_results"][0]["haic"]["Tr"], 0.7)
        payload_size = len(json.dumps(result))
        self.assertLess(payload_size, 5000)


if __name__ == "__main__":
    unittest.main()
