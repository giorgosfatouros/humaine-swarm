"""Unit tests for Kubeflow authentication helpers."""

import unittest
from unittest.mock import MagicMock, patch

import jwt

from utils.helper_functions import (
    extract_user_namespace_from_token,
    get_kubeflow_client,
    normalize_kubeflow_host,
)


class TestKubeflowAuth(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            "os.environ",
            {"KUBEFLOW_HOST": "http://test-kubeflow/pipeline"},
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_does_not_invent_namespace_from_email(self):
        token = jwt.encode(
            {"preferred_username": "gfatouros@gmail.com", "email": "gfatouros@gmail.com"},
            "secret",
            algorithm="HS256",
        )
        self.assertIsNone(extract_user_namespace_from_token(token))

    def test_extracts_explicit_namespace_claim(self):
        token = jwt.encode(
            {"namespace": "kubeflow-alice"},
            "secret",
            algorithm="HS256",
        )
        self.assertEqual(extract_user_namespace_from_token(token), "kubeflow-alice")

    def test_normalize_kubeflow_host_appends_pipeline(self):
        self.assertEqual(
            normalize_kubeflow_host("https://kubeflow.humaine-horizon.eu/"),
            "https://kubeflow.humaine-horizon.eu/pipeline",
        )
        self.assertEqual(
            normalize_kubeflow_host("https://kubeflow.humaine-horizon.eu/pipeline"),
            "https://kubeflow.humaine-horizon.eu/pipeline",
        )

    @patch("utils.helper_functions.kfp.Client")
    def test_bearer_token_creates_client(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        result = get_kubeflow_client(
            user_namespace="kubeflow-alice",
            user_token="keycloak-access-token",
        )

        self.assertEqual(result, mock_client)
        mock_client_cls.assert_called_once_with(
            host="http://test-kubeflow/pipeline",
            namespace="kubeflow-alice",
            existing_token="keycloak-access-token",
        )

    @patch("utils.helper_functions.kfp.Client")
    def test_unscoped_client_ignores_legacy_namespace_env(self, mock_client_cls):
        with patch.dict(
            "os.environ",
            {"KUBEFLOW_NAMESPACE": "legacy-private-namespace"},
        ):
            get_kubeflow_client(user_token="keycloak-access-token")

        mock_client_cls.assert_called_once_with(
            host="http://test-kubeflow/pipeline",
            namespace=None,
            existing_token="keycloak-access-token",
        )

    def test_missing_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bearer token"):
            get_kubeflow_client(user_namespace="kubeflow-bob")


class TestGetUserKubeflowClient(unittest.IsolatedAsyncioTestCase):
    async def test_missing_session_token_is_rejected(self):
        from agents.code import get_user_kubeflow_client

        with patch("agents.code.UserSessionManager.get_kubeflow_namespace", return_value=None), patch(
            "agents.code.UserSessionManager.get_oauth_token", return_value=None
        ):
            with self.assertRaisesRegex(ValueError, "access token"):
                await get_user_kubeflow_client()

    async def test_bearer_client_is_created_from_session(self):
        from agents.code import get_user_kubeflow_client

        mock_client = MagicMock()
        with patch("agents.code.UserSessionManager.get_kubeflow_namespace", return_value="kubeflow-alice"), patch(
            "agents.code.UserSessionManager.get_oauth_token", return_value="keycloak-access-token"
        ), patch(
            "agents.code.get_kubeflow_client",
            return_value=mock_client,
        ) as mock_factory:
            result = await get_user_kubeflow_client()

        self.assertIs(result, mock_client)
        mock_factory.assert_called_once_with(
            user_namespace="kubeflow-alice",
            user_token="keycloak-access-token",
        )


if __name__ == "__main__":
    unittest.main()
