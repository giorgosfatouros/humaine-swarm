"""Unit tests for Kubeflow profile namespace discovery and resolution."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agents.code import list_experiments, list_kubeflow_namespaces, list_runs, resolve_kubeflow_namespace
from classes.user_handler import UserSessionManager
from utils.helper_functions import (
    fetch_kubeflow_profile_namespaces,
    kubeflow_dashboard_root,
    select_default_kubeflow_namespace,
)

ENV_INFO = {
    "user": "gfatouros@gmail.com",
    "namespaces": [
        {"user": "gfatouros@gmail.com", "namespace": "smart-cities", "role": "contributor"},
    ],
    "isClusterAdmin": False,
}


def _session(namespace=None, namespaces=None):
    """Patch the session namespace getters."""
    return (
        patch("agents.code.UserSessionManager.get_kubeflow_namespace", return_value=namespace),
        patch("agents.code.UserSessionManager.get_kubeflow_namespaces", return_value=namespaces),
    )


class TestNamespaceDiscovery(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            "os.environ", {"KUBEFLOW_HOST": "https://kubeflow.example/pipeline"}
        )
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_dashboard_root_strips_pipeline(self):
        self.assertEqual(kubeflow_dashboard_root(), "https://kubeflow.example")

    @patch("utils.helper_functions.requests.get")
    def test_fetch_parses_env_info(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: ENV_INFO)

        namespaces = fetch_kubeflow_profile_namespaces("token")

        self.assertEqual(namespaces, [{"namespace": "smart-cities", "role": "contributor"}])
        url = mock_get.call_args.args[0]
        self.assertEqual(url, "https://kubeflow.example/api/workgroup/env-info")
        self.assertEqual(
            mock_get.call_args.kwargs["headers"], {"Authorization": "Bearer token"}
        )

    @patch("utils.helper_functions.requests.get")
    def test_fetch_raises_on_non_200(self, mock_get):
        mock_get.return_value = MagicMock(status_code=302)
        with self.assertRaises(ValueError):
            fetch_kubeflow_profile_namespaces("token")

    def test_default_prefers_owned_namespace(self):
        namespaces = [
            {"namespace": "smart-cities", "role": "contributor"},
            {"namespace": "kubeflow-alice", "role": "owner"},
        ]
        self.assertEqual(select_default_kubeflow_namespace(namespaces), "kubeflow-alice")
        self.assertEqual(select_default_kubeflow_namespace(namespaces[:1]), "smart-cities")
        self.assertIsNone(select_default_kubeflow_namespace([]))


class TestDiscoverAndStore(unittest.IsolatedAsyncioTestCase):
    async def test_discovered_namespace_becomes_default(self):
        with patch.object(UserSessionManager, "get_oauth_token", return_value="token"), patch.object(
            UserSessionManager, "get_kubeflow_namespace", return_value=None
        ), patch(
            "utils.helper_functions.fetch_kubeflow_profile_namespaces",
            return_value=[{"namespace": "smart-cities", "role": "contributor"}],
        ), patch.object(UserSessionManager, "set_kubeflow_namespaces") as set_list, patch.object(
            UserSessionManager, "set_kubeflow_namespace"
        ) as set_default:
            await UserSessionManager.discover_and_store_kubeflow_namespaces()

        set_list.assert_called_once_with([{"namespace": "smart-cities", "role": "contributor"}])
        set_default.assert_called_once_with("smart-cities")

    async def test_discovery_failure_keeps_claim_namespace(self):
        with patch.object(UserSessionManager, "get_oauth_token", return_value="token"), patch(
            "utils.helper_functions.fetch_kubeflow_profile_namespaces",
            side_effect=ValueError("HTTP 500"),
        ), patch.object(UserSessionManager, "set_kubeflow_namespaces") as set_list, patch.object(
            UserSessionManager, "set_kubeflow_namespace"
        ) as set_default:
            await UserSessionManager.discover_and_store_kubeflow_namespaces()

        set_list.assert_called_once_with(None)
        set_default.assert_not_called()


class TestResolveNamespace(unittest.TestCase):
    def test_defaults_to_session_namespace(self):
        a, b = _session("smart-cities", [{"namespace": "smart-cities", "role": "contributor"}])
        with a, b:
            self.assertEqual(resolve_kubeflow_namespace(None), ("smart-cities", None))

    def test_rejects_inaccessible_namespace(self):
        a, b = _session("smart-cities", [{"namespace": "smart-cities", "role": "contributor"}])
        with a, b:
            namespace, error = resolve_kubeflow_namespace("humaine")
        self.assertIsNone(namespace)
        self.assertIn("humaine", error)
        self.assertIn("smart-cities", error)

    def test_no_profile_namespace_is_a_clear_error(self):
        a, b = _session(None, [])
        with a, b:
            namespace, error = resolve_kubeflow_namespace(None)
        self.assertIsNone(namespace)
        self.assertIn("no Kubeflow profile namespace", error)

    def test_explicit_namespace_allowed_when_discovery_failed(self):
        a, b = _session(None, None)
        with a, b:
            self.assertEqual(resolve_kubeflow_namespace("kubeflow-bob"), ("kubeflow-bob", None))


class TestNamespacedTools(unittest.IsolatedAsyncioTestCase):
    async def test_list_runs_uses_default_namespace(self):
        client = MagicMock()
        client.list_runs.return_value = SimpleNamespace(runs=[], next_page_token="")
        a, b = _session("smart-cities", [{"namespace": "smart-cities", "role": "contributor"}])
        with a, b, patch("agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=client)):
            result = await list_runs.__wrapped__()

        self.assertEqual(client.list_runs.call_args.kwargs["namespace"], "smart-cities")
        self.assertEqual(result["namespace"], "smart-cities")

    async def test_list_experiments_rejects_guessed_namespace(self):
        client = MagicMock()
        a, b = _session("smart-cities", [{"namespace": "smart-cities", "role": "contributor"}])
        with a, b, patch("agents.code.get_user_kubeflow_client", new=AsyncMock(return_value=client)):
            result = await list_experiments.__wrapped__(namespace="humaine")

        self.assertIn("error", result)
        client.list_experiments.assert_not_called()

    async def test_list_kubeflow_namespaces(self):
        a, b = _session("smart-cities", [{"namespace": "smart-cities", "role": "contributor"}])
        with a, b:
            result = await list_kubeflow_namespaces.__wrapped__()

        self.assertEqual(result["default_namespace"], "smart-cities")
        self.assertEqual(result["namespaces"][0]["role"], "contributor")


if __name__ == "__main__":
    unittest.main()
