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
from utils.config import MINIO_ENDPOINT, MINIO_SECURE
# from literalai import AsyncLiteralClient
import datetime

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
            namespace=os.getenv("KUBEFLOW_NAMESPACE")
        )

    def create_kfp_client(self) -> kfp.Client:
        """Get a newly authenticated Kubeflow Pipelines client."""
        return self._create_kfp_client()
            
def get_kubeflow_client() -> kfp.Client:
    kfp_client_manager = KFPClientManager(
        api_url=os.getenv("KUBEFLOW_HOST"),
        skip_tls_verify=True,

        dex_username=os.getenv("KUBEFLOW_USERNAME"),
        dex_password=os.getenv("KUBEFLOW_PASSWORD"),

        # can be 'ldap' or 'local' depending on your Dex configuration
        dex_auth_type="local",
        )

    kfp_client = kfp_client_manager.create_kfp_client()
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