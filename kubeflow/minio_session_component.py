import kfp
from kfp import dsl
from kfp.components import load_component_from_file
from kfp.compiler import Compiler
from kfp.client import Client
import os

# Set your pipeline and component paths
COMPONENT_YAML = "kubeflow/get_minio_session_component.yaml"
PIPELINE_YAML = "minio_auth_pipeline.yaml"
PIPELINE_NAME = "MinIO Session Fetch Pipeline"

# Load the component from YAML
get_minio_session_op = load_component_from_file(COMPONENT_YAML)

# Define the pipeline using the component
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Authenticate with Keycloak and fetch MinIO session token"
)
def minio_auth_pipeline():
    # Use the same STS endpoint as in env.sh
    minio_endpoint = "s3-minio.humaine-horizon.eu"
    sts_endpoint = f"https://{minio_endpoint}"
    
    # Pass string values directly to the component
    get_minio_session_op(
        keycloak_url="https://keycloak.humaine-horizon.eu/realms/humaine/protocol/openid-connect/token",
        keycloak_client_id="minio",
        keycloak_client_secret="CJHIv1jYJfokZc73lUqwtkL12YBi69IB",
        keycloak_username="g.fatouros-dev",
        keycloak_password="g.fatouros-huma1ne!",
        minio_sts_endpoint=sts_endpoint,
        duration_seconds="43200"  # Match the duration from env.sh
    )

# Compile pipeline to a YAML file
Compiler().compile(
    pipeline_func=minio_auth_pipeline,
    package_path=PIPELINE_YAML
)

print(f"✅ Pipeline compiled to {PIPELINE_YAML}")

# Optional: Upload and run it on your Kubeflow Pipelines instance
AUTO_UPLOAD = True  # Set to True when you're ready to upload
KUBEFLOW_HOST = "http://huanew-kubeflow.ddns.net/pipeline"  
if AUTO_UPLOAD:
    SESSION_COOKIE = "2zr8Kvo1KeMzlrtN723tq6vt6Fl2zC0LkOI1UAkR0t43IoUuXMSVJdFHmoQDjvxd4B7UgAlItBHxCz0sNI_DdM7iICel-EdQPBGagsWppbpaNPsPDAT42vIHSDOvnCxFudf3FX5yfxyuY3Cr7EAxEgAC_x3haRQCeCIIbBDJ_u8M0yXTt_H9utX1zK0dfCXdJG-D6LsfJ-5OeTFIdgLISAnbW_wzKWDRkqsAUGN4coE7V8lPF5Ap_ghQN8E8l80S2l8LZu6mfpbzR3QHAW5zUGnnmRbzsEhyZbte4-NziJnLST_O0VCVvHN7CI0RPjUblFKnhJ8NXLkqKNHdUklMKY_GIidhXHZaHvwXGslWrEMsiCMfj_qW3soHqWqFfdMkFrR-rc3qklX_67ELez5o1UgbVOmcp5_qQ6NJ9CRYGkMfdU69cd5yCxCTvYyzKdd1hsFZ8SKsUJj-d3Oflg2No27ThrtFbcXDMFr6RhHRRQeM64kTYhRoIYDBkJ5SuLR6b_NSzY1OSn5eLKlz53xGcHgQRK9EAhu6xunF1z1L3mm201oTFKNbZY10w9BU1d27ygb0VP9sI5dAQduw5gsplbxtaCs6FSWLx-pKQHKNIfv8BJZ_0ivcBiZXchiif55kR0DLOCqHVnqs1JngMZJ8wctN0By7HmJ4ZcNcgfgMhdAqQrA0XfUiSMpy3VoK0i-QIZ2wswjMysLpvv-9ppTsn6UvLn8H5PPBE-NK1kG2Kw7T2gXvXe4EH0iYz90p5KCw3h80pSijlXYQWXcy-rNFw6qAR35AR4dwF9YK_U_2Ip1A3uGP2q2y4bBjq8I9Ml28B9rbwj6b_lbI2PpKjip6M5RJEcC4Ikh4DNBpqZPyEGCSPD5qQiE88YfJDcbW73uh1T-FHLqT7NgPP3tyWtlDZtW087kdQmWryTl_hEspvoLtBHNT9SgEx8Shwv0QkvZxgcb3mTHNSeP05UZZGJN1ogt0zTbfHbqSDtJ80cLcC6xETC4nOvYebdqvUJnLx4aBInHUIACr_BochGLLcYjnqXzgd_Vz1JZJQmu6iWvqBJAP9VSooW9E6cuySssZNnGu9UWwAPD_GerHsgaF08IDV9AI9kvJ_7YqjPe15qUU7Epj1y48pn-OnK3rve45OjA0rzQCEWdYK8rz7SSy06tEmYLPcnVCf6deTKNY4Ojpe_CMHHOjHvG07EF4EiWxpxtwXiHsQ8a0HMjWiX5DT7tL89300n1icllXVdnInTtOcdCRCtJImlNzyj9VJCdARwa_HD3ckJSR2g85JiFC0RqNPz-MvWTefe1WQXlTsbxQzkMjxs50XIqIBn-GMwhtKiGWVxoMo1I_r2zHyKOiSuTj8WEEffdO1S5LALtj0QzIjs0h2LwmwGbCWmMAxQH44Osf-OCTF6CakMXNxJTe3AjxvmSWw_I_WFcOBjjwMzXbQI6QWj_9AslqzKMd42q6IGcFhwEDt3sTa5QKSyRDiGjY1rzsi9HDjjgdxjReqS3mFTLup1W11gusvz8u9U-s_EuOmYxOlctz_JZ36jeGrpffxqVZGZ7uYDSL101wPsF80PH6iSEt7njNPLp6EV69eQh-K7uyuZoPs-hIc3M8yf8tK3rLhqsfIOao9w1iuN1BiFN6Te6taZScr-3oa9kEfbeffc_DH9gJ7ukTO53uFi44S8BicsXorJtAE79IoKiwiJ8eYDxgYr2oQqd8WXld1BsbqnkqobegjdswN4fTGrzN21-VHGzyZPXhdbD_nA0Ej0ioJntP2BVDyzR8Fh-9dIAYr55UfWxJjNWVQPUeEoGzKjB-zlJNFC0e-RjHURbn-bdywqMdbl9xD9OfkVx-Qnw59LEU5r3ZdUmlyTul2TuMmG_G2divstf3iLOwsEkAUUoSlXa4jTdTg2WWy5YZgO4BiCkMI6XZj44IVGHRvxsV8KhMAMhy7pqc5Vleh9V_3X-o9hdHkn6f|1743198925|R4SXfCYXf_Pk7MkBwMqhQsLV-dB9-pPSCWtU6EaYLds="  # Replace with actual token
    KFP_ENDPOINT = "http://huanew-kubeflow.ddns.net/pipeline"
    NAMESPACE = "innovacts"  # Replace with your namespace

    client = Client(
        host=KFP_ENDPOINT,
        cookies=f"oauth2_proxy_kubeflow={SESSION_COOKIE}",
        namespace=NAMESPACE
)

    # Upload or find existing pipeline
    try:
        pipeline = client.get_pipeline_id(PIPELINE_NAME)
        if pipeline is None:
            pipeline = client.upload_pipeline(
                pipeline_package_path=PIPELINE_YAML,
                pipeline_name=PIPELINE_NAME
            )
            print(f"🚀 Uploaded pipeline: {PIPELINE_NAME}")
    except Exception as e:
        print(f"⚠️ Could not find/upload pipeline: {e}")

    # Start a run
    try:
        run = client.create_run_from_pipeline_package(
            pipeline_file=PIPELINE_YAML,
            arguments={},
            run_name="minio-session-test-run"
        )
        print(f"🏃 Run started: {run.run_id}")
    except Exception as e:
        print(f"⚠️ Could not start pipeline run: {e}")
