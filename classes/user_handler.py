import tiktoken
import chainlit as cl
from utils.helper_functions import setup_logging
from utils.config import settings, MAX_INPUT_TOKENS
import logging
import uuid
import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import jwt

logger = setup_logging('USER', level=logging.INFO)

# Helper class for managing user session state
class UserSessionManager:
    @staticmethod
    def set_user_id(user_id):
        cl.user_session.set("user_id", user_id)
        logger.info(f"Set user ID: {user_id}")

    @staticmethod
    def get_user_id():
        app_user = cl.user_session.get("user")
        if not app_user:
            logger.warning("No user found in session")
            return None
        
        # Try multiple sources for user ID
        user_id = None
        
        # First try the identifier (most reliable in Chainlit)
        if hasattr(app_user, 'identifier') and app_user.identifier:
            user_id = app_user.identifier
        # Then try metadata email
        elif hasattr(app_user, 'metadata') and app_user.metadata and 'email' in app_user.metadata:
            user_id = app_user.metadata["email"]
        # Then try metadata sub (subject claim from OAuth)
        elif hasattr(app_user, 'metadata') and app_user.metadata and 'sub' in app_user.metadata:
            user_id = app_user.metadata["sub"]
        # Finally try metadata preferred_username
        elif hasattr(app_user, 'metadata') and app_user.metadata and 'preferred_username' in app_user.metadata:
            user_id = app_user.metadata["preferred_username"]
        else:
            logger.warning(f"Could not find user ID in app_user: {app_user}")
            user_id = "unknown_user"
        
        logger.info(f"Retrieved user ID: {user_id}")  
        return user_id
    
    @staticmethod
    def get_message_history() -> list:
        message_history = cl.user_session.get("message_history", [])
        if not isinstance(message_history, list):
            return []
        return message_history

    @staticmethod
    def increment_function_call_count():
        count = cl.user_session.get("function_call_count", 0)
        cl.user_session.set("function_call_count", (count or 0) + 1)

    @staticmethod
    def reset_function_call_count():
        cl.user_session.set("function_call_count", 0)

    @staticmethod
    def get_function_call_count() -> int:
        return cl.user_session.get("function_call_count", 0) or 0

    @staticmethod
    def get_thread_id():
        return cl.context.session.thread_id

    @staticmethod
    def set_message_history(message_history):
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')
        total_tokens = 0
        truncated_messages = []
        system_message = message_history[0] if message_history and message_history[0]['role'] == 'system' else None

        for message in reversed(message_history[1:]):
            message_tokens = len(encoding.encode(message.get('content', '') or ''))
            total_tokens += message_tokens
            if total_tokens > MAX_INPUT_TOKENS:
                break
            truncated_messages.insert(0, message)
        
        if system_message:
            truncated_messages.insert(0, system_message)
        
        cl.user_session.set("message_history", truncated_messages)

    @staticmethod
    def print_stored_info():
        user_id = UserSessionManager.get_user_id()
        function_call_count = UserSessionManager.get_function_call_count()
        thread_id = UserSessionManager.get_thread_id()
        logger.info(f"User ID: {user_id}")
        logger.info(f"Function Call Count: {function_call_count}")
        logger.info(f"Thread ID: {thread_id}")
    
    # OAuth Token Management
    @staticmethod
    def set_oauth_token(token: str):
        """Store the user's OAuth access token in the session."""
        cl.user_session.set("oauth_token", token)
        logger.info("OAuth token stored in session")
    
    @staticmethod
    def get_oauth_token() -> Optional[str]:
        """Retrieve the user's OAuth access token from the session."""
        return cl.user_session.get("oauth_token")
    
    # MinIO Credentials Management
    @staticmethod
    def set_minio_credentials(credentials: Dict[str, str]):
        """
        Store MinIO credentials in the session.
        Expected keys: access_key, secret_key, session_token, expiry
        """
        cl.user_session.set("minio_credentials", credentials)
        logger.info("MinIO credentials stored in session")
    
    @staticmethod
    def get_minio_credentials() -> Optional[Dict[str, str]]:
        """Retrieve MinIO credentials from the session."""
        return cl.user_session.get("minio_credentials")
    
    @staticmethod
    def are_minio_credentials_valid() -> bool:
        """Check if MinIO credentials exist and are not expired."""
        creds = UserSessionManager.get_minio_credentials()
        if not creds:
            return False
        
        expiry = creds.get("expiry")
        if not expiry:
            return False
        
        try:
            expiry_time = datetime.fromisoformat(expiry)
            # Consider credentials expired 5 minutes before actual expiry
            return datetime.now() < (expiry_time - timedelta(minutes=5))
        except:
            return False
    
    @staticmethod
    async def fetch_and_store_minio_credentials():
        """
        Fetch MinIO credentials using the OAuth token and store them in the session.
        """
        from utils.helper_functions import fetch_minio_credentials_from_keycloak
        
        oauth_token = UserSessionManager.get_oauth_token()
        if not oauth_token:
            logger.error("No OAuth token available to fetch MinIO credentials")
            raise ValueError("OAuth token not found in session")
        
        credentials = await fetch_minio_credentials_from_keycloak(oauth_token)
        UserSessionManager.set_minio_credentials(credentials)
        logger.info("MinIO credentials fetched and stored successfully")
    
    @staticmethod
    async def get_or_refresh_minio_credentials() -> Optional[Dict[str, str]]:
        """
        Get MinIO credentials, refreshing them if expired.
        Proactively refreshes if token is about to expire.
        Returns None if unable to get credentials.
        If OAuth token is expired, clears the session to force logout.
        """
        # Check if token should be refreshed proactively
        if UserSessionManager.should_refresh_token():
            logger.info("Token expiring soon, proactively refreshing MinIO credentials")
            try:
                await UserSessionManager.fetch_and_store_minio_credentials()
            except Exception as e:
                logger.warning(f"Proactive refresh failed, will try on next access: {e}")
        
        if UserSessionManager.are_minio_credentials_valid():
            return UserSessionManager.get_minio_credentials()
        
        try:
            await UserSessionManager.fetch_and_store_minio_credentials()
            return UserSessionManager.get_minio_credentials()
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to refresh MinIO credentials: {error_msg}")
            
            # Check if the error is due to expired OAuth token
            if "expired" in error_msg.lower() or "token is expired" in error_msg.lower():
                logger.warning("OAuth token expired - clearing session to force logout")
                UserSessionManager.clear_session()
                # Send a message to the user
                try:
                    await cl.Message(
                        content="Your session has expired. Please refresh the page to log in again.",
                        author="System"
                    ).send()
                except Exception as msg_error:
                    logger.error(f"Failed to send logout message: {msg_error}")
            
            return None
    
    @staticmethod
    def check_bucket_access(bucket_name: str) -> bool:
        """
        Check if user has access to a specific bucket based on their policies.
        
        Args:
            bucket_name: Name of the bucket to check access for
            
        Returns:
            True if user has access, False otherwise
        """
        policies = UserSessionManager.get_user_policies()
        if not policies:
            # If no policies, allow access (fallback to MinIO's own authorization)
            logger.warning(f"No policies found for user, allowing access to {bucket_name}")
            return True
        
        # Check if any policy grants access to this bucket
        # Policy names might match bucket names or contain bucket patterns
        bucket_lower = bucket_name.lower()
        for policy in policies:
            policy_lower = policy.lower()
            # Check if policy name matches bucket or contains bucket name
            if (policy_lower == bucket_lower or 
                bucket_lower in policy_lower or 
                policy_lower in bucket_lower):
                logger.info(f"User has access to {bucket_name} via policy {policy}")
                return True
        
        logger.warning(f"User does not have explicit policy access to {bucket_name}")
        # Still return True - let MinIO's own authorization handle it
        # This is a soft check, actual authorization happens at MinIO level
        return True
    
    # Kubeflow Namespace Management
    @staticmethod
    def set_kubeflow_namespace(namespace: str):
        """Store the user's Kubeflow namespace in the session."""
        cl.user_session.set("kubeflow_namespace", namespace)
        logger.info(f"Kubeflow namespace set to: {namespace}")
    
    @staticmethod
    def get_kubeflow_namespace() -> Optional[str]:
        """Retrieve the user's Kubeflow namespace from the session."""
        return cl.user_session.get("kubeflow_namespace")
    
    # Kubeflow Credentials Management
    @staticmethod
    def set_kubeflow_credentials(username: str, password: str, namespace: Optional[str] = None):
        """
        Store Kubeflow credentials in the session.
        
        Args:
            username: Kubeflow username
            password: Kubeflow password
            namespace: Optional Kubeflow namespace
        """
        credentials = {
            "username": username,
            "password": password,
            "namespace": namespace
        }
        cl.user_session.set("kubeflow_credentials", credentials)
        logger.info("Kubeflow credentials stored in session")
        # Also update namespace if provided
        if namespace:
            UserSessionManager.set_kubeflow_namespace(namespace)
    
    @staticmethod
    def get_kubeflow_credentials() -> Optional[Dict[str, str]]:
        """Retrieve Kubeflow credentials from the session."""
        return cl.user_session.get("kubeflow_credentials")
    
    @staticmethod
    def has_kubeflow_credentials() -> bool:
        """Check if Kubeflow credentials exist in the session."""
        creds = UserSessionManager.get_kubeflow_credentials()
        if not creds:
            return False
        # Check if username and password are present
        return bool(creds.get("username") and creds.get("password"))
    
    @staticmethod
    def clear_kubeflow_credentials():
        """Clear Kubeflow credentials from the session."""
        cl.user_session.set("kubeflow_credentials", None)
        logger.info("Kubeflow credentials cleared from session")
    
    @staticmethod
    def extract_and_store_namespace():
        """
        Extract Kubeflow namespace from OAuth token claims and store it.
        Falls back to default namespace if not found.
        """
        from utils.helper_functions import extract_user_namespace_from_token
        
        oauth_token = UserSessionManager.get_oauth_token()
        if not oauth_token:
            logger.warning("No OAuth token available to extract namespace")
            # Set default namespace
            UserSessionManager.set_kubeflow_namespace("kubeflow")
            return
        
        try:
            namespace = extract_user_namespace_from_token(oauth_token)
            UserSessionManager.set_kubeflow_namespace(namespace)
            logger.info(f"Extracted namespace from token: {namespace}")
        except Exception as e:
            logger.error(f"Failed to extract namespace from token: {str(e)}")
            # Set default namespace as fallback
            UserSessionManager.set_kubeflow_namespace("kubeflow")
    
    # User Roles Management
    @staticmethod
    def set_user_roles(roles: list):
        """Store the user's roles from token claims in the session."""
        cl.user_session.set("user_roles", roles)
        logger.info(f"User roles stored: {roles}")
    
    @staticmethod
    def get_user_roles() -> list:
        """Retrieve the user's roles from the session."""
        return cl.user_session.get("user_roles", [])
    
    # User Policies Management (MinIO access policies)
    @staticmethod
    def set_user_policies(policies: list):
        """Store the user's MinIO policies from token claims in the session."""
        cl.user_session.set("user_policies", policies)
        logger.info(f"User policies stored: {policies}")
    
    @staticmethod
    def get_user_policies() -> list:
        """Retrieve the user's MinIO policies from the session."""
        return cl.user_session.get("user_policies", [])
    
    # Token Metadata Management
    @staticmethod
    def set_token_metadata(metadata: Dict):
        """Store comprehensive token metadata (email, username, expiration, etc.)."""
        cl.user_session.set("token_metadata", metadata)
        logger.info(f"Token metadata stored for user: {metadata.get('email', metadata.get('preferred_username', 'unknown'))}")
    
    @staticmethod
    def get_token_metadata() -> Optional[Dict]:
        """Retrieve comprehensive token metadata."""
        return cl.user_session.get("token_metadata")
    
    @staticmethod
    def extract_and_store_token_info():
        """
        Extract and store comprehensive information from OAuth token:
        - Policies (for MinIO bucket access)
        - Roles (realm_access + resource_access)
        - User metadata (email, username, etc.)
        - Token expiration info
        """
        oauth_token = UserSessionManager.get_oauth_token()
        if not oauth_token:
            logger.warning("No OAuth token available to extract token info")
            return
        
        try:
            import jwt
            # Decode token without verification (we trust it came from OAuth flow)
            decoded = jwt.decode(oauth_token, options={"verify_signature": False})
            
            # Extract policies (MinIO access policies)
            policies = decoded.get("policy", [])
            if isinstance(policies, list) and policies:
                UserSessionManager.set_user_policies(policies)
                logger.info(f"Extracted policies: {policies}")
            
            # Extract all roles (realm_access + resource_access)
            all_roles = []
            
            # Realm roles
            realm_access = decoded.get("realm_access", {})
            realm_roles = realm_access.get("roles", [])
            if isinstance(realm_roles, list):
                all_roles.extend(realm_roles)
            
            # Resource-specific roles
            resource_access = decoded.get("resource_access", {})
            for resource, data in resource_access.items():
                resource_roles = data.get("roles", [])
                if isinstance(resource_roles, list):
                    all_roles.extend([f"{resource}:{role}" for role in resource_roles])
            
            if all_roles:
                UserSessionManager.set_user_roles(all_roles)
                logger.info(f"Extracted roles: {all_roles}")
            
            # Store comprehensive token metadata
            token_metadata = {
                "email": decoded.get("email"),
                "preferred_username": decoded.get("preferred_username"),
                "given_name": decoded.get("given_name"),
                "family_name": decoded.get("family_name"),
                "email_verified": decoded.get("email_verified", False),
                "sub": decoded.get("sub"),  # Subject (user ID)
                "exp": decoded.get("exp"),  # Expiration timestamp
                "iat": decoded.get("iat"),  # Issued at timestamp
                "auth_time": decoded.get("auth_time"),  # Authentication time
                "scope": decoded.get("scope", "").split() if decoded.get("scope") else [],  # Scopes as list
                "session_state": decoded.get("session_state"),
            }
            
            UserSessionManager.set_token_metadata(token_metadata)
            
            # Log token expiration info for proactive refresh
            if token_metadata.get("exp"):
                exp_time = datetime.fromtimestamp(token_metadata["exp"])
                time_until_expiry = exp_time - datetime.now()
                logger.info(f"Token expires at {exp_time}, {time_until_expiry.total_seconds() / 3600:.2f} hours remaining")
            
        except jwt.DecodeError as e:
            logger.error(f"Failed to decode JWT token for info extraction: {str(e)}")
        except Exception as e:
            logger.error(f"Error extracting token info: {str(e)}")
    
    @staticmethod
    def should_refresh_token() -> bool:
        """
        Check if token should be refreshed proactively.
        Returns True if token expires within the next 5 minutes.
        """
        metadata = UserSessionManager.get_token_metadata()
        if not metadata or not metadata.get("exp"):
            return False
        
        try:
            exp_time = datetime.fromtimestamp(metadata["exp"])
            time_until_expiry = exp_time - datetime.now()
            # Refresh if less than 5 minutes remaining
            return time_until_expiry.total_seconds() < 300
        except Exception as e:
            logger.error(f"Error checking token expiration: {e}")
            return False
    
    @staticmethod
    def clear_session():
        """
        Clear all user session data to force logout.
        This is called when the OAuth token expires.
        """
        logger.info("Clearing user session due to expired token")
        try:
            # Clear all session data
            cl.user_session.clear()
            logger.info("Session cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
