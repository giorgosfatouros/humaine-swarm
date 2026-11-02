"""
Publish the Smart Cities auto-correction demo assets.

1. Uploads train.csv / test.csv to MinIO (default s3://smart-cities-data/auto-correction/).
   Uses MINIO_ACCESS_KEY / MINIO_SECRET_KEY / MINIO_SESSION_TOKEN from the env or .env when set,
   otherwise exchanges the Keycloak token for temporary MinIO STS keys.
2. Uploads the compiled pipeline YAML as a private pipeline in your Kubeflow profile namespace
   (found from the dashboard, or --namespace) as `smart-cities-auto-correction-classifier`,
   or adds a new version if it exists there. --shared uploads to the shared catalog instead.
   Needs KUBEFLOW_HOST.

The Keycloak token comes from KUBEFLOW_TOKEN when set. Otherwise the script asks for a Keycloak
username and password in the terminal and logs in with the OAUTH_KEYCLOAK_* settings from .env.
The password is read without echo and is never printed or stored.

Usage (from the repo root):
    poetry run python kubeflow/smart_cities/auto_correction_pipeline.py   # compile first
    poetry run python kubeflow/smart_cities/upload_demo_assets.py [--username NAME] [--namespace NS | --shared]
                                                                  [--skip-data] [--skip-pipeline]
"""
import argparse
import asyncio
import datetime
import getpass
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.dotenv_local import load_swarm_dotenv  # noqa: E402

load_swarm_dotenv()

import requests  # noqa: E402

from utils.helper_functions import (  # noqa: E402
    fetch_kubeflow_profile_namespaces,
    fetch_minio_credentials_from_keycloak,
    get_kubeflow_client,
    get_minio_client,
    select_default_kubeflow_namespace,
)

PIPELINE_YAML = HERE / "auto_correction_pipeline.yaml"


def keycloak_login(username: str, password: str, realm: str, client_id: str) -> str:
    """Return a Keycloak access token from a username/password (direct access grant)."""
    base_url = os.getenv("OAUTH_KEYCLOAK_BASE_URL", "").rstrip("/")
    if not base_url:
        raise SystemExit("Set OAUTH_KEYCLOAK_BASE_URL (e.g. in .env)")
    data = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
        "scope": "openid",
    }
    # The .env secret belongs to the app's own client, so only send it for that client
    secret = os.getenv("OAUTH_KEYCLOAK_CLIENT_SECRET")
    if secret and client_id == os.getenv("OAUTH_KEYCLOAK_CLIENT_ID"):
        data["client_secret"] = secret

    response = requests.post(
        f"{base_url}/realms/{realm}/protocol/openid-connect/token",
        data=data,
        timeout=30,
        verify=os.getenv("KEYCLOAK_CA_BUNDLE") or True,
    )
    if not response.ok:
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error", "")
        detail = body.get("error_description") or response.text[:200] or f"HTTP {response.status_code}"
        if error == "invalid_grant":
            raise SystemExit(f"Keycloak login failed: {detail} (check the username and password)")
        if error == "unauthorized_client":
            raise SystemExit(
                f"Keycloak login failed: {detail}. Client '{client_id}' must allow "
                "'Direct access grants', or pass --client-id for one that does."
            )
        raise SystemExit(f"Keycloak login failed: {detail}")
    return response.json()["access_token"]


def get_token(args: argparse.Namespace) -> str:
    token = os.getenv("KUBEFLOW_TOKEN")
    if token:
        print("Using the Keycloak token from KUBEFLOW_TOKEN")
        return token
    username = args.username or input(f"Keycloak username (realm '{args.realm}'): ").strip()
    password = getpass.getpass("Keycloak password: ")
    if not username or not password:
        raise SystemExit("Username and password are required")
    token = keycloak_login(username, password, args.realm, args.client_id)
    print(f"Logged in to Keycloak as {username}")
    return token


def minio_credentials(token: str) -> dict:
    """Exchange a Keycloak token for temporary MinIO STS credentials."""
    try:
        return asyncio.run(fetch_minio_credentials_from_keycloak(token))
    except Exception as exc:
        raise SystemExit(f"Could not get MinIO credentials with the Keycloak token: {exc}")


def upload_data(data_dir: Path, bucket: str, prefix: str, credentials: dict | None = None) -> None:
    client = get_minio_client(user_credentials=credentials)
    if not client.bucket_exists(bucket):
        raise SystemExit(f"Bucket '{bucket}' does not exist or is not visible with these credentials")
    for name in ("train.csv", "test.csv"):
        src = data_dir / name
        if not src.is_file():
            raise SystemExit(f"Missing {src}")
        obj = f"{prefix.rstrip('/')}/{name}"
        client.fput_object(bucket, obj, str(src), content_type="text/csv")
        print(f"Uploaded {src.name} -> s3://{bucket}/{obj}")


def resolve_namespace(token: str, requested: str | None) -> str:
    """Return the profile namespace for a private upload: the requested one, else the user's default."""
    if requested:
        return requested
    try:
        namespaces = fetch_kubeflow_profile_namespaces(token)
    except Exception as exc:
        raise SystemExit(f"Could not look up your Kubeflow namespace ({exc}); pass --namespace or --shared")
    namespace = select_default_kubeflow_namespace(namespaces)
    if not namespace:
        raise SystemExit("Your account has no Kubeflow profile namespace; pass --namespace or --shared")
    return namespace


def find_pipeline_id(client, name: str, namespace: str | None) -> str | None:
    """Find a pipeline by name in one namespace (None = shared catalog)."""
    name_filter = json.dumps({"predicates": [{"operation": "EQUALS", "key": "display_name", "stringValue": name}]})
    pipelines = client.list_pipelines(namespace=namespace, filter=name_filter, page_size=10).pipelines or []
    return pipelines[0].pipeline_id if pipelines else None


def upload_pipeline(token: str, namespace: str | None) -> None:
    # Imported here so the pipeline module (and kfp) loads only when needed
    sys.path.insert(0, str(HERE))
    from auto_correction_pipeline import PIPELINE_DESCRIPTION, PIPELINE_NAME

    if not PIPELINE_YAML.is_file():
        raise SystemExit(f"{PIPELINE_YAML.name} not found; compile the pipeline first")
    client = get_kubeflow_client(user_token=token)
    scope = f"namespace {namespace}" if namespace else "the shared catalog"
    version_name = "v" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pipeline_id = find_pipeline_id(client, PIPELINE_NAME, namespace)
    if pipeline_id:
        version = client.upload_pipeline_version(
            pipeline_package_path=str(PIPELINE_YAML),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_id,
            description=PIPELINE_DESCRIPTION,
        )
        print(f"Added version {version_name} to {PIPELINE_NAME} in {scope}")
    else:
        pipeline = client.upload_pipeline(
            pipeline_package_path=str(PIPELINE_YAML),
            pipeline_name=PIPELINE_NAME,
            description=PIPELINE_DESCRIPTION,
            namespace=namespace,
        )
        pipeline_id = pipeline.pipeline_id
        version = client.list_pipeline_versions(pipeline_id=pipeline_id, page_size=1).pipeline_versions[0]
        print(f"Created pipeline {PIPELINE_NAME} in {scope}")
    print(f"pipeline_id: {pipeline_id}")
    print(f"version_id:  {version.pipeline_version_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "kubeflow" / "Synthetic mistakes",
                    help="Folder with train.csv and test.csv")
    ap.add_argument("--bucket", default="smart-cities-data")
    ap.add_argument("--prefix", default="auto-correction/")
    ap.add_argument("--skip-data", action="store_true", help="Do not upload the CSVs")
    ap.add_argument("--skip-pipeline", action="store_true", help="Do not upload the pipeline")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--namespace", help="Kubeflow profile namespace for the private pipeline "
                                           "(default: your own, from the Kubeflow dashboard)")
    scope.add_argument("--shared", action="store_true", help="Upload to the shared catalog instead")
    ap.add_argument("--username", help="Keycloak username (prompted when omitted)")
    ap.add_argument("--realm", default=os.getenv("OAUTH_KEYCLOAK_REALM", "humaine"),
                    help="Keycloak realm (default: OAUTH_KEYCLOAK_REALM)")
    ap.add_argument("--client-id", default=os.getenv("OAUTH_KEYCLOAK_CLIENT_ID", "minio"),
                    help="Keycloak client for the login (default: OAUTH_KEYCLOAK_CLIENT_ID)")
    a = ap.parse_args()

    env_minio_keys = bool(os.getenv("MINIO_ACCESS_KEY") and os.getenv("MINIO_SECRET_KEY"))
    needs_token = not a.skip_pipeline or (not a.skip_data and not env_minio_keys)
    token = get_token(a) if needs_token else None

    if not a.skip_data:
        credentials = None if env_minio_keys else minio_credentials(token)
        upload_data(a.data_dir, a.bucket, a.prefix, credentials)
    if not a.skip_pipeline:
        namespace = None if a.shared else resolve_namespace(token, a.namespace)
        upload_pipeline(token, namespace)


if __name__ == "__main__":
    main()
