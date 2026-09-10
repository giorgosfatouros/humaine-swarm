"""Unit tests for multi-namespace Kubeflow pipeline listing."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agents.code import (
    _kubeflow_pipeline_queries,
    _merge_pipeline_listings,
    get_kf_pipelines,
    resolve_kubeflow_shared_namespace,
)


class TestKubeflowPipelines(unittest.TestCase):
    def test_resolve_shared_namespace_empty_means_unscoped(self):
        self.assertIsNone(resolve_kubeflow_shared_namespace(""))
        self.assertIsNone(resolve_kubeflow_shared_namespace("   "))

    def test_resolve_shared_namespace_explicit_value(self):
        self.assertEqual(resolve_kubeflow_shared_namespace("kubeflow"), "kubeflow")

    def test_pipeline_queries_include_unscoped_shared(self):
        queries = _kubeflow_pipeline_queries("kubeflow-g-fatouros-dev", include_shared=True)
        self.assertEqual(
            queries,
            [
                ("kubeflow-g-fatouros-dev", "kubeflow-g-fatouros-dev", "private"),
                (None, "", "shared"),
            ],
        )

    @patch("agents.code.resolve_kubeflow_shared_namespace", return_value="kubeflow")
    def test_pipeline_queries_dedupe_when_user_equals_shared(self, _mock_resolve):
        queries = _kubeflow_pipeline_queries("kubeflow", include_shared=True)
        self.assertEqual(queries, [("kubeflow", "kubeflow", "private")])

    def test_pipeline_queries_unscoped_only_without_user_namespace(self):
        queries = _kubeflow_pipeline_queries(None, include_shared=True)
        self.assertEqual(queries, [(None, "", "shared")])

    def test_pipeline_queries_private_only(self):
        queries = _kubeflow_pipeline_queries("kubeflow-g-fatouros-dev", include_shared=False)
        self.assertEqual(
            queries,
            [("kubeflow-g-fatouros-dev", "kubeflow-g-fatouros-dev", "private")],
        )

    def test_merge_pipeline_listings_tags_and_dedupes(self):
        private_pipeline = SimpleNamespace(
            pipeline_id="p1",
            display_name="Private",
            description="desc",
            created_at="2026-01-01",
        )
        shared_pipeline = SimpleNamespace(
            pipeline_id="p2",
            display_name="Shared",
            description="shared desc",
            created_at="2026-01-02",
            namespace="kubeflow",
        )
        duplicate = SimpleNamespace(
            pipeline_id="p2",
            display_name="Shared duplicate",
            description="ignored",
            created_at="2026-01-03",
            namespace="kubeflow",
        )

        merged = _merge_pipeline_listings([
            ("kubeflow-user", "private", [private_pipeline]),
            ("", "shared", [shared_pipeline, duplicate]),
        ])

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["namespace_type"], "private")
        self.assertEqual(merged[1]["namespace"], "kubeflow")
        self.assertEqual(merged[1]["namespace_type"], "shared")


class TestKubeflowPipelinePagination(unittest.IsolatedAsyncioTestCase):
    async def test_unscoped_catalog_uses_and_returns_page_token(self):
        client = MagicMock()
        client.list_pipelines.return_value = SimpleNamespace(
            pipelines=[],
            next_page_token="next-shared-page",
        )

        with patch(
            "agents.code.get_user_kubeflow_client",
            new=AsyncMock(return_value=client),
        ), patch(
            "agents.code.UserSessionManager.get_kubeflow_namespace",
            return_value=None,
        ), patch(
            "agents.code.resolve_kubeflow_shared_namespace",
            return_value=None,
        ), patch.dict(
            "os.environ",
            {"KUBEFLOW_NAMESPACE": "legacy-private-namespace"},
        ):
            result = await get_kf_pipelines.__wrapped__(page_token="shared-page-2")

        client.list_pipelines.assert_called_once_with(
            page_token="shared-page-2",
            page_size=20,
            sort_by="created_at desc",
            namespace=None,
        )
        self.assertEqual(result["next_page_token"], "next-shared-page")


if __name__ == "__main__":
    unittest.main()
