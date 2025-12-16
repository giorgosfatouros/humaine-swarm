"""
Custom Keycloak OAuth provider that handles missing email gracefully.
This extends Chainlit's built-in Keycloak provider to handle cases where
email is not included in the userinfo response.
"""

import os
from typing import Dict, Tuple
from chainlit.oauth_providers import KeycloakOAuthProvider
from chainlit import User
import logging

logger = logging.getLogger(__name__)


class CustomKeycloakProvider(KeycloakOAuthProvider):
    """
    Custom Keycloak OAuth provider that handles missing email field.
    Falls back to preferred_username or sub (subject) if email is not available.
    """
    
    async def get_user_info(self, token: str) -> Tuple[Dict, User]:
        """
        Get user information from Keycloak, with fallback for missing email.
        
        Args:
            token: The OAuth access token
            
        Returns:
            Tuple of (raw_user_data, User object)
        """
        # Get the raw user data from Keycloak
        async with self.session.get(
            self.userinfo_endpoint,
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            response.raise_for_status()
            kc_user = await response.json()
            logger.info(f"Keycloak user data received: {kc_user.keys()}")
        
        # Extract identifier with fallback chain: email -> preferred_username -> sub
        identifier = None
        if "email" in kc_user and kc_user["email"]:
            identifier = kc_user["email"]
            logger.info(f"Using email as identifier: {identifier}")
        elif "preferred_username" in kc_user and kc_user["preferred_username"]:
            identifier = kc_user["preferred_username"]
            logger.info(f"Email not found, using preferred_username as identifier: {identifier}")
        elif "sub" in kc_user and kc_user["sub"]:
            identifier = kc_user["sub"]
            logger.info(f"Email and username not found, using sub as identifier: {identifier}")
        else:
            raise ValueError("Cannot determine user identifier: no email, preferred_username, or sub found in Keycloak response")
        
        # Extract user metadata
        metadata = {
            "provider": "keycloak",
            "keycloak_sub": kc_user.get("sub"),
            "preferred_username": kc_user.get("preferred_username"),
            "email": kc_user.get("email"),
            "email_verified": kc_user.get("email_verified", False),
            "name": kc_user.get("name"),
            "given_name": kc_user.get("given_name"),
            "family_name": kc_user.get("family_name"),
            "groups": kc_user.get("groups", []),
            "realm_access": kc_user.get("realm_access", {}),
        }
        
        # Create the Chainlit User object
        user = User(
            identifier=identifier,
            metadata=metadata,
        )
        
        logger.info(f"Created user with identifier: {identifier}")
        
        return kc_user, user


def get_custom_keycloak_provider():
    """
    Create and return a custom Keycloak OAuth provider.
    
    Returns:
        CustomKeycloakProvider instance configured from environment variables
    """
    # Get configuration from environment
    client_id = os.environ.get("OAUTH_KEYCLOAK_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_KEYCLOAK_CLIENT_SECRET")
    base_url = os.environ.get("OAUTH_KEYCLOAK_BASE_URL")
    realm = os.environ.get("OAUTH_KEYCLOAK_REALM")
    
    if not all([client_id, client_secret, base_url, realm]):
        raise ValueError("Missing required Keycloak OAuth environment variables")
    
    # The provider will be initialized by Chainlit
    # This function is just a helper for configuration
    return {
        "id": "keycloak",
        "name": os.environ.get("OAUTH_KEYCLOAK_NAME", "Keycloak"),
        "client_id": client_id,
        "client_secret": client_secret,
        "authorize_url": f"{base_url}/realms/{realm}/protocol/openid-connect/auth",
        "token_url": f"{base_url}/realms/{realm}/protocol/openid-connect/token",
        "userinfo_endpoint": f"{base_url}/realms/{realm}/protocol/openid-connect/userinfo",
    }

