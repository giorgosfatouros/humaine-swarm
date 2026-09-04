"""Unit tests for Smart Manufacturing semantic-inference client."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from utils.semantic_inference_client import (
    get_entity,
    query_manufacturing_knowledge,
    semantic_search,
)


def _mock_response(status_code: int, json_data):
    response = MagicMock()
    response.ok = 200 <= status_code < 300
    response.status_code = status_code
    response.json.return_value = json_data
    response.text = str(json_data)
    return response


class TestSemanticSearch(unittest.IsolatedAsyncioTestCase):
    @patch("utils.semantic_inference_client.requests.post")
    async def test_basic_search_maps_payload(self, mock_post):
        mock_post.return_value = _mock_response(
            200,
            {
                "query": "list all drilling machines",
                "meta": {
                    "requested_n": 10,
                    "returned": 1,
                    "reranking_enabled": True,
                },
                "results": [
                    {
                        "id": "4:abc:271",
                        "name": "Machine #271",
                        "type": "Asset",
                        "labels": ["Drilling"],
                        "score": 0.82,
                        "properties": {"category": "Drilling"},
                        "relationships": [],
                    }
                ],
            },
        )

        result = await semantic_search("list all drilling machines", n=10)

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "semantic_search")
        self.assertEqual(result["query"], "list all drilling machines")
        self.assertEqual(result["meta"]["returned"], 1)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["name"], "Machine #271")

        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(call_kwargs["json"]["query"], "list all drilling machines")
        self.assertEqual(call_kwargs["json"]["n"], 10)
        self.assertTrue(call_kwargs["json"]["include_relationships"])
        self.assertTrue(call_kwargs["json"]["include_properties"])

    @patch("utils.semantic_inference_client.requests.post")
    async def test_truncated_relationships_passthrough(self, mock_post):
        mock_post.return_value = _mock_response(
            200,
            {
                "query": "drilling machines",
                "meta": {"returned": 1},
                "results": [
                    {
                        "id": "4:abc:1",
                        "name": "EcoDrive elite",
                        "relationships_truncated": True,
                        "total_relationships": 120,
                        "relationships": [],
                    }
                ],
            },
        )

        result = await semantic_search("drilling machines")

        self.assertTrue(result["success"])
        entity = result["results"][0]
        self.assertTrue(entity["relationships_truncated"])
        self.assertEqual(entity["total_relationships"], 120)

    @patch("utils.semantic_inference_client.requests.post")
    async def test_empty_results(self, mock_post):
        mock_post.return_value = _mock_response(
            200,
            {"query": "unknown machine", "meta": {"returned": 0}, "results": []},
        )

        result = await semantic_search("unknown machine")

        self.assertTrue(result["success"])
        self.assertEqual(result["results"], [])

    @patch("utils.semantic_inference_client.requests.post")
    async def test_error_envelope(self, mock_post):
        mock_post.return_value = _mock_response(
            400,
            {"error": {"message": "query is required", "code": "bad_request"}},
        )

        result = await semantic_search("test")

        self.assertFalse(result["success"])
        self.assertIn("query is required", result["error"])

    @patch("utils.semantic_inference_client.requests.post")
    async def test_timeout(self, mock_post):
        mock_post.side_effect = requests.Timeout("timed out")

        result = await semantic_search("drilling machines")

        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])


class TestEntityLookup(unittest.IsolatedAsyncioTestCase):
    @patch("utils.semantic_inference_client.requests.get")
    async def test_entity_id_get_path(self, mock_get):
        entity_id = "4:65968995-1a1b-4765-9dab-e9dfd5daef59:7"
        mock_get.return_value = _mock_response(
            200,
            {
                "id": entity_id,
                "name": "EcoDrive elite",
                "type": "Asset",
                "relationships": [{"relation": "HAS_DRILLING", "target_name": "Drilling"}],
            },
        )

        result = await get_entity(entity_id)

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "get_entity")
        self.assertEqual(result["entity_id"], entity_id)
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["name"], "EcoDrive elite")
        mock_get.assert_called_once()
        self.assertIn(entity_id, mock_get.call_args.args[0])

    @patch("utils.semantic_inference_client.requests.get")
    async def test_query_manufacturing_knowledge_entity_id_routes_to_get(self, mock_get):
        entity_id = "4:abc:99"
        mock_get.return_value = _mock_response(
            200,
            {"id": entity_id, "name": "Machine #99", "type": "Asset"},
        )

        result = await query_manufacturing_knowledge(entity_id=entity_id)

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "get_entity")
        mock_get.assert_called_once()

    @patch("utils.semantic_inference_client.requests.post")
    async def test_query_manufacturing_knowledge_search_routes_to_post(self, mock_post):
        mock_post.return_value = _mock_response(
            200,
            {"query": "titancraft", "meta": {}, "results": []},
        )

        result = await query_manufacturing_knowledge(query="titancraft")

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "semantic_search")
        mock_post.assert_called_once()

    async def test_missing_query_and_entity_id(self):
        result = await query_manufacturing_knowledge()

        self.assertFalse(result["success"])
        self.assertIn("required", result["error"])


if __name__ == "__main__":
    unittest.main()
