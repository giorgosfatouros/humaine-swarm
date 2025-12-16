import logging
from typing import Optional, Dict, List, Any, NamedTuple
import os
import re
from urllib.parse import urlsplit, urlencode
import kfp
from kfp.dsl import Input, Output, Artifact, Dataset, Model, Metrics, ClassificationMetrics
import requests
import urllib3
# import matplotlib.pyplot as plt
from chainlit import User
from agents.definition import functions
import json
from minio import Minio
from utils.config import MINIO_ENDPOINT, MINIO_SECURE, MINIO_API_ENDPOINT
import datetime
import xml.etree.ElementTree as ET
from datetime import datetime as dt, timedelta
import jwt
import asyncio



def setup_logging(logger_name, level=logging.INFO):
    # Check for LOG_LEVEL environment variable to override the default level
    env_log_level = os.getenv('LOG_LEVEL', '').upper()
    if env_log_level:
        # Map string level names to logging constants
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        if env_log_level in level_map:
            level = level_map[env_log_level]
        else:
            # If invalid level, use default and log a warning
            logging.warning(f"Invalid LOG_LEVEL '{env_log_level}'. Using provided level {level}")
    
    logger = logging.getLogger(logger_name)
    if not logger.handlers:  # Check if the logger already has handlers
        # Create a console handler and set the level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    logger.setLevel(level)
    return logger

logger = setup_logging('HELPER-FUNCTIONS', level=logging.ERROR)


def decode_jwt(token: str) -> Optional[Dict]:
    import jwt, os
    # Load the secret key securely from environment variables
    SECRET_KEY = os.getenv("CHAINLIT_AUTH_SECRET")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("JWT token has expired")
    except jwt.InvalidTokenError:
        logger.error("Invalid JWT token")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None


def extract_token_from_headers(headers: Dict) -> Optional[str]:
    auth_header = headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    referer = headers.get("referer")
    if referer and "=" in referer:
        return referer.split("=")[1].split('&')[0]
    return None


def extract_user_from_payload(payload: Dict) -> Optional[User]:
    user_id = payload.get("userId")
    user_email = payload.get("email")
    user_name = payload.get("name", '')
    token_provider = payload.get("provider", 'wix')
    plan = payload.get("plan", 'free')

    if not user_id or not user_email:
        return None

    metadata = {"id": user_id, "email": user_email, "name": user_name, "plan": plan, "role": "user", "provider": token_provider}
    return User(identifier=user_email, metadata=metadata)



def get_required_arguments(function_name: str):
    for func in functions:
        if func['name'] == function_name:
            required_params = func['parameters'].get('required', [])
            return required_params
    return []


def manage_chat_history(message_history):      
    total_length = sum(len(json.dumps(msg)) for msg in message_history)
    while total_length > 100000:
        if len(message_history) > 2:
            removed_msg = message_history.pop(2)  # Remove the third message (index 2)
            total_length -= len(json.dumps(removed_msg))
            logging.info(f"Reducing conversation history. Current length: {total_length} characters")
        else:
            break  # If we only have two or fewer messages, stop removing
    
    return message_history


def rag_extract_deliverables(retrieved_nodes):
    return [
        {
            "text": f"{node.get_content()}",
       
        }
        for node in retrieved_nodes
    ]

# def create_plot(data_dict):
#     """
#     Create a plot for the given data dictionary.

#     Args:
#         data_dict (dict): A dictionary where keys are indicators and values are dictionaries with dates as keys and values as data points.

#     Returns:
#         matplotlib.figure.Figure: The created plot figure.
#     """
#     fig, ax = plt.subplots(figsize=(12, 8))  # Make the figure larger
#     for indicator, data in data_dict.items():
#         dates = list(data.keys())
#         values = [entry['Value'] for entry in data.values()]
#         ax.plot(dates, values, label=indicator)

#     ax.legend()
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Value")
#     ax.grid(True)

#     # # Set y-axis to log scale to ensure the scale of each plot doesn't affect the visualization of the other
#     # if len(data_dict) > 1:
#     #     ax.set_yscale('log')

#     # Improve date visualization on x-axis
#     fig.autofmt_xdate()
#     ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Limit the number of x-axis labels

#     return fig


async def fetch_minio_credentials_from_keycloak(access_token: str) -> Dict[str, str]:
    """
    Fetch temporary MinIO credentials using Keycloak access token via AssumeRoleWithWebIdentity.
    
    Args:
        access_token: The Keycloak OAuth access token
        
    Returns:
        Dict containing: access_key, secret_key, session_token, expiry (ISO format)
        
    Raises:
        Exception if credential fetch fails
    """
    # Check if token is expired before making the request
    try:
        decoded = jwt.decode(access_token, options={"verify_signature": False})
        exp = decoded.get("exp")
        if exp:
            exp_time = dt.fromtimestamp(exp)
            now = dt.now()
            if exp_time < now:
                logger.error(f"Token expired at {exp_time}, current time is {now}")
                raise Exception(f"OAuth token is expired. Please refresh your session.")
    except jwt.DecodeError:
        logger.warning("Could not decode token to check expiration, proceeding anyway")
    except Exception as e:
        if "expired" in str(e).lower():
            raise
        logger.warning(f"Error checking token expiration: {e}, proceeding anyway")
    # Check if a separate STS endpoint is configured (full URL)
    # This allows explicit override of the STS endpoint if needed
    sts_endpoint = os.environ.get("MINIO_STS_ENDPOINT")
    
    if sts_endpoint:
        # Use explicitly configured STS endpoint (can include protocol and port)
        if sts_endpoint.startswith("http://") or sts_endpoint.startswith("https://"):
            sts_url = sts_endpoint
        else:
            # If no protocol specified, use HTTPS if MINIO_SECURE is true
            sts_url = f"https://{sts_endpoint}" if MINIO_SECURE else f"http://{sts_endpoint}"
    else:
        # Use the main MinIO endpoint - match the old working approach from env.sh
        minio_endpoint = MINIO_ENDPOINT
        
        # Remove protocol if present (we'll add it back based on MINIO_SECURE)
        if minio_endpoint.startswith("http://"):
            minio_endpoint = minio_endpoint[7:]
        elif minio_endpoint.startswith("https://"):
            minio_endpoint = minio_endpoint[8:]
        
        # Remove port if present (we'll handle it based on MINIO_SECURE)
        if ":" in minio_endpoint:
            hostname = minio_endpoint.rsplit(":", 1)[0]
        else:
            hostname = minio_endpoint
        
        # Validate hostname - ensure it's the correct endpoint
        if hostname == "minio.humaine-horizon.eu":
            logger.warning(f"Wrong endpoint detected: 'minio.humaine-horizon.eu', overriding to 's3-minio.humaine-horizon.eu'")
            hostname = "s3-minio.humaine-horizon.eu"
        
        # Match the old working approach:
        # - When MINIO_SECURE is true: Use HTTPS on default port (443)
        #   Try root path first, then try /minio/sts if root returns HTML
        # - When MINIO_SECURE is false: Use HTTP on port 9000 - direct API access
        if MINIO_SECURE:
            # Try HTTPS on default port (443) - reverse proxy should route POST to API
            # Some reverse proxy configs need the root path, others need /minio/sts
            sts_url = f"https://{hostname}"
        else:
            # HTTP on explicit API port (9000) - direct access
            sts_url = f"http://{hostname}:9000"
    
    # Build params for STS request (matches old curl command)
    params = {
        "Action": "AssumeRoleWithWebIdentity",
        "Version": "2011-06-15",
        "DurationSeconds": "43200",  # 12 hours
        "WebIdentityToken": access_token
    }
    
    # Set headers explicitly to match the old working curl command
    # Add headers to help reverse proxy distinguish API requests from browser requests
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "MinIO-STS-Client/1.0",
        "Accept": "application/xml, text/xml, */*",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    try:
        # Use asyncio to run the synchronous request in a thread pool
        # Match the old working approach: POST with form data
        # Try with form data first (matches old curl command)
        loop = asyncio.get_event_loop()
        
        # POST with form data (matches old curl -d approach)
        response = await loop.run_in_executor(
            None, 
            lambda: requests.post(sts_url, data=params, headers=headers, verify=MINIO_SECURE, timeout=10)
        )
        
        # Check for HTTP errors and parse error responses
        if response.status_code != 200:
            # Try to parse error response from MinIO STS (it returns XML ErrorResponse)
            error_message = None
            try:
                xml_response = ET.fromstring(response.content)
                namespace = {'sts': 'https://sts.amazonaws.com/doc/2011-06-15/'}
                error_elem = xml_response.find(".//sts:Error", namespace)
                if error_elem is None:
                    error_elem = xml_response.find(".//Error")
                
                if error_elem is not None:
                    code_elem = error_elem.find(".//sts:Code", namespace)
                    if code_elem is None:
                        code_elem = error_elem.find(".//Code")
                    msg_elem = error_elem.find(".//sts:Message", namespace)
                    if msg_elem is None:
                        msg_elem = error_elem.find(".//Message")
                    
                    if code_elem is not None and msg_elem is not None:
                        error_message = f"{code_elem.text}: {msg_elem.text}"
            except Exception:
                pass  # If we can't parse the error, use the default message
            
            if error_message:
                logger.error(f"MinIO STS API error: {error_message}")
                raise Exception(f"MinIO STS error: {error_message}")
            else:
                logger.error(f"MinIO STS API returned status {response.status_code}: {response.text}")
                raise Exception(f"Failed to get MinIO credentials: HTTP {response.status_code}")
        
        # Check for HTTP errors - match reference implementation
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            logger.error(f"MinIO STS API returned error: {e}")
            logger.error(f"Response: {response.text}")
            raise Exception(f"Failed to get MinIO credentials: HTTP {response.status_code}")
        
        # Check if response is empty
        if not response.text or not response.text.strip():
            logger.error("MinIO STS API returned empty response")
            raise Exception("Empty response from MinIO STS API")
        
        # Check if we got HTML (MinIO Console) instead of XML
        # This happens when reverse proxy routes to console instead of API
        if response.text.strip().startswith('<!') or 'text/html' in response.headers.get('Content-Type', '').lower():
            logger.error(f"Received HTML response (MinIO Console) instead of XML. This suggests the reverse proxy is not routing POST requests to the API.")
            logger.error(f"Response URL: {response.url}")
            
            # Try alternative STS paths that some reverse proxy configs use
            if MINIO_SECURE and sts_url.startswith('https://') and ':9000' not in sts_url:
                # Extract hostname from the endpoint we used
                endpoint_for_fallback = MINIO_ENDPOINT
                if endpoint_for_fallback.startswith("http://"):
                    endpoint_for_fallback = endpoint_for_fallback[7:]
                elif endpoint_for_fallback.startswith("https://"):
                    endpoint_for_fallback = endpoint_for_fallback[8:]
                
                # Remove port if present
                if ":" in endpoint_for_fallback:
                    fallback_hostname = endpoint_for_fallback.rsplit(":", 1)[0]
                else:
                    fallback_hostname = endpoint_for_fallback
                
                # Try alternative paths that some MinIO reverse proxy configs use
                alternative_paths = ['/minio/sts', '/sts', '/api/sts']
                
                for alt_path in alternative_paths:
                    fallback_url = f"https://{fallback_hostname}{alt_path}"
                    logger.info(f"Trying alternative STS path: {fallback_url}")
                    
                    try:
                        # Retry with alternative path
                        response = await loop.run_in_executor(
                            None, 
                            lambda url=fallback_url: requests.post(url, data=params, headers=headers, verify=MINIO_SECURE, timeout=10)
                        )
                        
                        logger.info(f"Alternative path Response Status: {response.status_code}")
                        logger.info(f"Alternative path Response Headers: {dict(response.headers)}")
                        logger.info(f"Alternative path Response Text (first 500 chars): {response.text[:500]}")
                        
                        if response.status_code == 200:
                            # Check if we got XML this time
                            if not (response.text.strip().startswith('<!') or 'text/html' in response.headers.get('Content-Type', '').lower()):
                                logger.info(f"Successfully got XML response from {fallback_url}")
                                break  # Success, exit the loop
                    
                    except Exception as e:
                        logger.warning(f"Alternative path {fallback_url} failed: {str(e)}")
                        continue
                
                # If we still have HTML after trying alternatives, raise error
                if response.text.strip().startswith('<!') or 'text/html' in response.headers.get('Content-Type', '').lower():
                    raise Exception("MinIO STS endpoint returned HTML (Console) instead of XML. Tried root and alternative paths. Check reverse proxy configuration to route POST requests to MinIO API.")
                
                if response.status_code != 200:
                    logger.error(f"All alternative paths failed. Last status: {response.status_code}: {response.text}")
                    raise Exception(f"Failed to get MinIO credentials: HTTP {response.status_code}")
        
        # Parse XML response - use response.content like the reference implementation
        try:
            xml_response = ET.fromstring(response.content)
        except ET.ParseError as parse_err:
            logger.error(f"Failed to parse XML. Response content: {response.text[:1000]}")
            logger.error(f"Response content type: {response.headers.get('Content-Type', 'unknown')}")
            raise
        
        # Define namespace - match the reference implementation
        namespace = {'sts': 'https://sts.amazonaws.com/doc/2011-06-15/'}
        
        # Extract credentials directly from XML tree - match reference implementation
        access_key_elem = xml_response.find(".//sts:AccessKeyId", namespace)
        secret_key_elem = xml_response.find(".//sts:SecretAccessKey", namespace)
        session_token_elem = xml_response.find(".//sts:SessionToken", namespace)
        expiration_elem = xml_response.find(".//sts:Expiration", namespace)
        
        # Try without namespace if not found (fallback)
        if access_key_elem is None:
            access_key_elem = xml_response.find(".//AccessKeyId")
        if secret_key_elem is None:
            secret_key_elem = xml_response.find(".//SecretAccessKey")
        if session_token_elem is None:
            session_token_elem = xml_response.find(".//SessionToken")
        if expiration_elem is None:
            expiration_elem = xml_response.find(".//Expiration")
        
        # Extract text values
        access_key = access_key_elem.text if access_key_elem is not None else None
        secret_key = secret_key_elem.text if secret_key_elem is not None else None
        session_token = session_token_elem.text if session_token_elem is not None else None
        expiration = expiration_elem.text if expiration_elem is not None else None
        
        if not all([access_key, secret_key, session_token]):
            logger.error(f"Missing credential fields in response: {response.text}")
            raise Exception("Incomplete credentials in MinIO STS response")
        
        logger.info("Successfully fetched MinIO credentials from Keycloak token")
        
        return {
            "access_key": access_key,
            "secret_key": secret_key,
            "session_token": session_token,
            "expiry": expiration or (dt.now() + timedelta(hours=12)).isoformat()
        }
        
    except requests.RequestException as e:
        logger.error(f"Request failed when fetching MinIO credentials: {str(e)}")
        raise Exception(f"Failed to connect to MinIO STS API: {str(e)}")
    except ET.ParseError as e:
        logger.error(f"Failed to parse MinIO STS response: {str(e)}")
        raise Exception(f"Invalid XML response from MinIO: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching MinIO credentials: {str(e)}")
        raise


def extract_user_namespace_from_token(access_token: str) -> str:
    """
    Extract Kubeflow namespace from Keycloak OAuth token claims.
    Looks for namespace in groups, roles, or custom claims.
    
    Args:
        access_token: The Keycloak OAuth access token
        
    Returns:
        The user's Kubeflow namespace, or "kubeflow" as default
    """
    try:
        # Decode token without verification (we trust it came from OAuth flow)
        decoded = jwt.decode(access_token, options={"verify_signature": False})
        
        # Try to extract namespace from various possible claim locations
        # 1. Check for direct namespace claim
        if "namespace" in decoded:
            namespace = decoded["namespace"]
            logger.info(f"Found namespace in token claims: {namespace}")
            return namespace
        
        # 2. Check groups for namespace-like patterns
        groups = decoded.get("groups", [])
        if isinstance(groups, list):
            for group in groups:
                if group.startswith("kubeflow-"):
                    namespace = group
                    logger.info(f"Found namespace in groups: {namespace}")
                    return namespace
        
        # 3. Check realm_access roles
        realm_access = decoded.get("realm_access", {})
        roles = realm_access.get("roles", [])
        if isinstance(roles, list):
            for role in roles:
                if role.startswith("kubeflow-"):
                    namespace = role
                    logger.info(f"Found namespace in realm roles: {namespace}")
                    return namespace
        
        # 4. Check resource_access roles
        resource_access = decoded.get("resource_access", {})
        for resource, data in resource_access.items():
            resource_roles = data.get("roles", [])
            if isinstance(resource_roles, list):
                for role in resource_roles:
                    if role.startswith("kubeflow-"):
                        namespace = role
                        logger.info(f"Found namespace in resource roles: {namespace}")
                        return namespace
        
        # 5. Use preferred_username to construct namespace
        username = decoded.get("preferred_username") or decoded.get("email")
        if username:
            # Convert email/username to valid namespace format
            namespace = f"kubeflow-{username.replace('@', '-').replace('.', '-').lower()}"
            logger.info(f"Constructed namespace from username: {namespace}")
            return namespace
        
        # Default fallback
        logger.warning("Could not extract namespace from token, using default 'kubeflow'")
        return "kubeflow"
        
    except jwt.DecodeError as e:
        logger.error(f"Failed to decode JWT token: {str(e)}")
        return "kubeflow"
    except Exception as e:
        logger.error(f"Error extracting namespace from token: {str(e)}")
        return "kubeflow"


def get_minio_client(user_credentials: Optional[Dict[str, str]] = None) -> Minio:
    """
    Initialize and return a MinIO client.
    
    Args:
        user_credentials: Optional dict with user-specific credentials
                         Expected keys: access_key, secret_key, session_token
                         If not provided, falls back to environment variables
    
    Returns:
        Configured MinIO client instance
    """
    # Determine the correct endpoint for S3 API operations
    # Match the test script behavior:
    # - When MINIO_SECURE=true: Use endpoint as-is (no port), secure=True → HTTPS on port 443 (reverse proxy)
    # - When MINIO_SECURE=false: Use endpoint:9000, secure=False → HTTP on port 9000 (direct access)
    if MINIO_API_ENDPOINT:
        # Use explicitly configured API endpoint
        api_endpoint = MINIO_API_ENDPOINT
        # Extract protocol if present to determine secure flag
        if api_endpoint.startswith("https://"):
            api_secure = True
            api_endpoint = api_endpoint[8:]
        elif api_endpoint.startswith("http://"):
            api_secure = False
            api_endpoint = api_endpoint[7:]
        else:
            # No protocol specified, use MINIO_SECURE setting
            api_secure = MINIO_SECURE
    else:
        # Construct endpoint based on MINIO_ENDPOINT
        # Remove protocol if present
        endpoint = MINIO_ENDPOINT
        if endpoint.startswith("http://"):
            endpoint = endpoint[7:]
        elif endpoint.startswith("https://"):
            endpoint = endpoint[8:]
        
        # Remove port if present
        if ":" in endpoint:
            hostname = endpoint.rsplit(":", 1)[0]
        else:
            hostname = endpoint
        
        # Validate hostname - ensure it's the correct endpoint
        if hostname == "minio.humaine-horizon.eu":
            logger.error(f"ERROR: Wrong endpoint detected! Got 'minio.humaine-horizon.eu' but expected 's3-minio.humaine-horizon.eu'")
            logger.error(f"ERROR: Overriding to correct endpoint: s3-minio.humaine-horizon.eu")
            hostname = "s3-minio.humaine-horizon.eu"
        
        # Match test script behavior:
        # When secure=True: use endpoint as-is (HTTPS on port 443 via reverse proxy)
        # When secure=False: use endpoint:9000 (HTTP on port 9000 direct access)
        if MINIO_SECURE:
            # Use endpoint as-is - MinIO client will use HTTPS on port 443
            api_endpoint = hostname
            api_secure = True
        else:
            # Use port 9000 for direct HTTP access
            api_endpoint = f"{hostname}:9000"
            api_secure = False
    
    if user_credentials:
        # Use user-specific credentials from session
        return Minio(
            endpoint=api_endpoint,
            access_key=user_credentials["access_key"],
            secret_key=user_credentials["secret_key"],
            session_token=user_credentials.get("session_token"),
            secure=api_secure
        )
    else:
        # Fall back to environment variables (for backward compatibility)
        logger.info("Using MinIO credentials from environment variables")
        return Minio(
            endpoint=api_endpoint,
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            session_token=os.getenv("MINIO_SESSION_TOKEN"),
            secure=api_secure
        )



class KFPClientManager:
    """
    A class that creates `kfp.Client` instances with Dex authentication.
    """

    def __init__(
        self,
        api_url: str,
        dex_username: str,
        dex_password: str,
        dex_auth_type: str = "local",
        skip_tls_verify: bool = False,
    ):
        """
        Initialize the KfpClient

        :param api_url: the Kubeflow Pipelines API URL
        :param skip_tls_verify: if True, skip TLS verification
        :param dex_username: the Dex username
        :param dex_password: the Dex password
        :param dex_auth_type: the auth type to use if Dex has multiple enabled, one of: ['ldap', 'local']
        """
        self._api_url = api_url
        self._skip_tls_verify = skip_tls_verify
        self._dex_username = dex_username
        self._dex_password = dex_password
        self._dex_auth_type = dex_auth_type
        self._client = None

        # disable SSL verification, if requested
        if self._skip_tls_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # ensure `dex_default_auth_type` is valid
        if self._dex_auth_type not in ["ldap", "local"]:
            raise ValueError(
                f"Invalid `dex_auth_type` '{self._dex_auth_type}', must be one of: ['ldap', 'local']"
            )

    def _get_session_cookies(self) -> str:
        """
        Get the session cookies by authenticating against Dex
        :return: a string of session cookies in the form "key1=value1; key2=value2"
        """

        # use a persistent session (for cookies)
        s = requests.Session()

        # GET the api_url, which should redirect to Dex
        resp = s.get(
            self._api_url, allow_redirects=True, verify=not self._skip_tls_verify
        )
        if resp.status_code == 200:
            pass
        elif resp.status_code == 403:
            # if we get 403, we might be at the oauth2-proxy sign-in page
            # the default path to start the sign-in flow is `/oauth2/start?rd=<url>`
            url_obj = urlsplit(resp.url)
            url_obj = url_obj._replace(
                path="/oauth2/start", query=urlencode({"rd": url_obj.path})
            )
            resp = s.get(
                url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
            )
        else:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {self._api_url}"
            )

        # if we were NOT redirected, then the endpoint is unsecured
        if len(resp.history) == 0:
            # no cookies are needed
            return ""

        # if we are at `../auth` path, we need to select an auth type
        url_obj = urlsplit(resp.url)
        if re.search(r"/auth$", url_obj.path):
            url_obj = url_obj._replace(
                path=re.sub(r"/auth$", f"/auth/{self._dex_auth_type}", url_obj.path)
            )

        # if we are at `../auth/xxxx/login` path, then we are at the login page
        if re.search(r"/auth/.*/login$", url_obj.path):
            dex_login_url = url_obj.geturl()
        else:
            # otherwise, we need to follow a redirect to the login page
            resp = s.get(
                url_obj.geturl(), allow_redirects=True, verify=not self._skip_tls_verify
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {url_obj.geturl()}"
                )
            dex_login_url = resp.url

        # attempt Dex login
        resp = s.post(
            dex_login_url,
            data={"login": self._dex_username, "password": self._dex_password},
            allow_redirects=True,
            verify=not self._skip_tls_verify,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for POST against: {dex_login_url}"
            )

        # if we were NOT redirected, then the login credentials were probably invalid
        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials are probably invalid - "
                f"No redirect after POST to: {dex_login_url}"
            )

        # if we are at `../approval` path, we need to approve the login
        url_obj = urlsplit(resp.url)
        if re.search(r"/approval$", url_obj.path):
            dex_approval_url = url_obj.geturl()

            # approve the login
            resp = s.post(
                dex_approval_url,
                data={"approval": "approve"},
                allow_redirects=True,
                verify=not self._skip_tls_verify,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for POST against: {url_obj.geturl()}"
                )

        return "; ".join([f"{c.name}={c.value}" for c in s.cookies])

    def _create_kfp_client(self, namespace: Optional[str] = None) -> kfp.Client:
        try:
            session_cookies = self._get_session_cookies()
        except Exception as ex:
            raise RuntimeError(f"Failed to get Dex session cookies") from ex

        # monkey patch the kfp.Client to support disabling SSL verification
        # kfp only added support in v2: https://github.com/kubeflow/pipelines/pull/7174
        original_load_config = kfp.Client._load_config

        def patched_load_config(client_self, *args, **kwargs):
            config = original_load_config(client_self, *args, **kwargs)
            config.verify_ssl = not self._skip_tls_verify
            return config

        patched_kfp_client = kfp.Client
        patched_kfp_client._load_config = patched_load_config

        # Use provided namespace or fall back to environment variable
        kf_namespace = namespace or os.getenv("KUBEFLOW_NAMESPACE")

        return patched_kfp_client(
            host=self._api_url,
            cookies=session_cookies,
            namespace=kf_namespace
        )

    def create_kfp_client(self, namespace: Optional[str] = None) -> kfp.Client:
        """
        Get a newly authenticated Kubeflow Pipelines client.
        
        Args:
            namespace: Optional namespace override
        """
        return self._create_kfp_client(namespace=namespace)
            
def get_kubeflow_client(
    user_namespace: Optional[str] = None, 
    user_token: Optional[str] = None,
    user_username: Optional[str] = None,
    user_password: Optional[str] = None
) -> kfp.Client:
    """
    Gets the Kubeflow client with proper authentication.
    
    Args:
        user_namespace: Optional user-specific namespace. If not provided, uses env var or default
        user_token: Optional OAuth token for the user. If provided, assumes SSO authentication
        user_username: Optional username for authentication. If provided, uses this instead of env vars
        user_password: Optional password for authentication. If provided, uses this instead of env vars
        
    Returns:
        Configured Kubeflow client instance
        
    Note: When Kubeflow is integrated with Keycloak SSO, the OAuth token should be used.
        For now, uses username/password authentication. Credentials should be provided via
        user_username/user_password parameters (from session) rather than environment variables.
    """
    # Suppress the specific KFP client warning about version compatibility
    import warnings
    warnings.filterwarnings("ignore", message="This client only works with Kubeflow Pipeline.*", category=FutureWarning)
    
    # Use provided credentials if available, otherwise use empty strings (will fail gracefully)
    kubeflow_username = user_username or ""
    kubeflow_password = user_password or ""
    
    if not kubeflow_username or not kubeflow_password:
        logger.warning("Kubeflow credentials not provided. Authentication may fail.")
    
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KUBEFLOW_HOST"),
        skip_tls_verify=True,
        dex_username=kubeflow_username,
        dex_password=kubeflow_password,
        dex_auth_type="local",
    )

    kfp_client = kfp_client_manager.create_kfp_client(namespace=user_namespace)
    
    # Log namespace usage
    if user_namespace:
        logger.info(f"Using Kubeflow namespace: {user_namespace}")
    
    return kfp_client


def get_kubeflow_old_client(host: str, username: str, password: str, namespace: str) -> kfp.Client:
    """Initialize and return a Kubeflow Pipelines client."""
    session = requests.Session()
    response = session.get(host)
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }
    
    data = {"login": username, "password": password}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]
    
    return kfp.Client(
        host=f"{host}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=namespace,
    )

# New functions for artifact metadata handling

def create_dataset_artifact(uri: str, metadata: Dict[str, Any] = None) -> Dataset:
    """
    Create a Dataset artifact with proper metadata.
    
    Args:
        uri: The URI where the dataset is stored
        metadata: Dictionary of metadata to attach to the dataset
        
    Returns:
        Dataset: A properly configured dataset artifact
    """
    if metadata is None:
        metadata = {}
        
    dataset = Dataset(uri=uri)
    
    # Add standard metadata fields
    dataset.metadata.update({
        'creation_time': str(datetime.datetime.now()),
        'format': metadata.get('format', 'unknown'),
        'size': metadata.get('size', 0),
        'sample_count': metadata.get('sample_count', 0),
        'description': metadata.get('description', ''),
        'source': metadata.get('source', ''),
    })
    
    # Add any additional metadata
    for key, value in metadata.items():
        if key not in dataset.metadata:
            dataset.metadata[key] = value
            
    return dataset

def create_model_artifact(uri: str, metadata: Dict[str, Any] = None) -> Model:
    """
    Create a Model artifact with proper metadata.
    
    Args:
        uri: The URI where the model is stored
        metadata: Dictionary of metadata to attach to the model
        
    Returns:
        Model: A properly configured model artifact
    """
    if metadata is None:
        metadata = {}
        
    model = Model(uri=uri)
    
    # Add standard metadata fields
    model.metadata.update({
        'framework': metadata.get('framework', 'unknown'),
        'creation_time': str(datetime.datetime.now()),
        'version': metadata.get('version', '0.1'),
        'hyperparameters': json.dumps(metadata.get('hyperparameters', {})),
        'training_dataset': metadata.get('training_dataset', ''),
        'description': metadata.get('description', ''),
    })
    
    # Add any additional metadata
    for key, value in metadata.items():
        if key not in model.metadata:
            model.metadata[key] = value
            
    return model

def create_metrics_artifact(uri: str, metrics_data: Dict[str, Any] = None) -> Metrics:
    """
    Create a Metrics artifact with proper metadata.
    
    Args:
        uri: The URI where the metrics are stored
        metrics_data: Dictionary of metrics to include
        
    Returns:
        Metrics: A properly configured metrics artifact
    """
    if metrics_data is None:
        metrics_data = {}
        
    metrics = Metrics(uri=uri)
    
    # Add standard metadata fields
    metrics.metadata.update({
        'creation_time': str(datetime.datetime.now()),
    })
    
    # Add metrics as metadata
    for key, value in metrics_data.items():
        metrics.metadata[key] = value
            
    return metrics

def create_classification_metrics_artifact(
    uri: str, 
    confusion_matrix: Optional[List[List[int]]] = None,
    roc_data: Optional[Dict[str, List[float]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ClassificationMetrics:
    """
    Create a ClassificationMetrics artifact with proper metadata.
    
    Args:
        uri: The URI where the metrics are stored
        confusion_matrix: Confusion matrix as a list of lists
        roc_data: ROC curve data with 'fpr' and 'tpr' keys
        metadata: Additional metadata
        
    Returns:
        ClassificationMetrics: A properly configured classification metrics artifact
    """
    if metadata is None:
        metadata = {}
        
    metrics = ClassificationMetrics(uri=uri)
    
    # Add standard metadata fields
    metrics.metadata.update({
        'creation_time': str(datetime.datetime.now()),
    })
    
    # Add confusion matrix if provided
    if confusion_matrix:
        metrics.metadata['confusion_matrix'] = json.dumps(confusion_matrix)
    
    # Add ROC data if provided
    if roc_data:
        metrics.metadata['roc_data'] = json.dumps(roc_data)
    
    # Add any additional metadata
    for key, value in metadata.items():
        if key not in metrics.metadata:
            metrics.metadata[key] = value
            
    return metrics

def read_artifact_metadata(artifact: Artifact) -> Dict[str, Any]:
    """
    Read and return the metadata from an artifact.
    
    Args:
        artifact: The artifact to read metadata from
        
    Returns:
        Dict: The artifact's metadata
    """
    return artifact.metadata

def log_artifact_properties(artifact: Artifact):
    """
    Log the properties of an artifact for debugging or monitoring.
    
    Args:
        artifact: The artifact to log
    """
    logger.info(f"Artifact Name: {artifact.name}")
    logger.info(f"Artifact URI: {artifact.uri}")
    logger.info(f"Artifact Path: {artifact.path}")
    logger.info(f"Artifact Metadata: {json.dumps(artifact.metadata, indent=2)}")

# Helper function to read dataset from an artifact
def read_dataset_from_artifact(dataset_artifact: Input[Dataset]) -> Any:
    """
    Read a dataset from an artifact.
    
    Args:
        dataset_artifact: The dataset artifact to read
        
    Returns:
        Any: The loaded dataset
    """
    with open(dataset_artifact.path, 'r') as f:
        # This is a simple example; adjust based on your dataset format
        data = f.read()
    
    log_artifact_properties(dataset_artifact)
    return data

# Helper function to save model to an artifact
def save_model_to_artifact(model, model_artifact: Output[Model], metadata: Dict[str, Any] = None):
    """
    Save a model to an artifact.
    
    Args:
        model: The model to save
        model_artifact: The model artifact to save to
        metadata: Dictionary of metadata to attach to the model
    """
    if metadata is None:
        metadata = {}
    
    # Save model to the artifact path
    model_dir = os.path.dirname(model_artifact.path)
    os.makedirs(model_dir, exist_ok=True)
    
    # This is a placeholder - replace with your actual model saving code
    # For example: model.save(model_artifact.path)
    
    # Update artifact metadata
    model_artifact.metadata.update({
        'framework': metadata.get('framework', 'unknown'),
        'version': metadata.get('version', '0.1'),
        'creation_time': str(datetime.datetime.now()),
    })
    
    # Add any additional metadata
    for key, value in metadata.items():
        if key not in model_artifact.metadata:
            model_artifact.metadata[key] = value
    
    log_artifact_properties(model_artifact)

# Helper function to save metrics to an artifact
def save_metrics_to_artifact(metrics_data: Dict[str, Any], metrics_artifact: Output[Metrics]):
    """
    Save metrics to a metrics artifact.
    
    Args:
        metrics_data: Dictionary of metrics to save
        metrics_artifact: The metrics artifact to save to
    """
    # Save metrics as JSON
    with open(metrics_artifact.path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    # Update artifact metadata with metrics
    for key, value in metrics_data.items():
        metrics_artifact.metadata[key] = value
    
    metrics_artifact.metadata['creation_time'] = str(datetime.datetime.now())
    log_artifact_properties(metrics_artifact)