import logging
from typing import Optional, Dict
import os
import re
from urllib.parse import urlsplit, urlencode
import kfp
import requests
import urllib3
# import matplotlib.pyplot as plt
from chainlit import User
from typing import List, Any
from agents.definition import functions
import json
from minio import Minio
from utils.config import MINIO_ENDPOINT, MINIO_SECURE
# from literalai import AsyncLiteralClient

# lai = AsyncLiteralClient()



def setup_logging(logger_name, level=logging.INFO, log_file='error.log'):
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

        # Create a file handler for logging errors to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

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

# async def get_literal_participant_id(user_id):
#     if user_id:
#         lai_user = await lai.api.get_or_create_user(identifier=user_id)
#         return lai_user.id
#     else:
#         return None

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


def get_minio_client() -> Minio:
    """Initialize and return a MinIO client."""
    return Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        session_token=os.getenv("MINIO_SESSION_TOKEN"),
        secure=MINIO_SECURE
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

    def _create_kfp_client(self) -> kfp.Client:
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

        return patched_kfp_client(
            host=self._api_url,
            cookies=session_cookies,
        )

    def create_kfp_client(self) -> kfp.Client:
        """Get a newly authenticated Kubeflow Pipelines client."""
        return self._create_kfp_client()
            