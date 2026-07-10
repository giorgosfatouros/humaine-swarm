#!/usr/bin/env python3
"""Verify MinIO endpoint auto-correction for wrong MINIO_ENDPOINT value."""

from __future__ import annotations

import logging
import os
import sys

# Ensure project root is on path when run as script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.dotenv_local import load_swarm_dotenv

load_swarm_dotenv()

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


def run_check(endpoint: str) -> str:
    os.environ["MINIO_ENDPOINT"] = endpoint
    # Reload config module values used by helper_functions
    import importlib
    import utils.config as config
    import utils.helper_functions as hf

    importlib.reload(config)
    importlib.reload(hf)

    client = hf.get_minio_client()
    actual_endpoint = client._base_url.host
    return actual_endpoint


def main() -> int:
  wrong = "minio.humaine-horizon.eu"
  correct = "s3-minio.humaine-horizon.eu"

  print(f"=== MINIO_ENDPOINT={wrong} ===")
  actual_wrong = run_check(wrong)
  print(f"Client endpoint netloc: {actual_wrong}")
  assert actual_wrong == correct, f"Expected override to {correct}, got {actual_wrong}"

  print(f"\n=== MINIO_ENDPOINT={correct} ===")
  actual_correct = run_check(correct)
  print(f"Client endpoint netloc: {actual_correct}")
  assert actual_correct == correct, f"Expected {correct}, got {actual_correct}"

  print("\nOK: wrong endpoint is auto-corrected; correct endpoint passes through unchanged.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
