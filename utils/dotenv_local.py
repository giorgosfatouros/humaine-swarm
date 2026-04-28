"""
Load humaine-swarm/.env into os.environ before other imports.

Chainlit does not read Next.js `common-fe/.env`. Put Keycloak JWKS SSL flags here
or export them in the shell that runs `chainlit run`.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_swarm_dotenv() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.is_file():
        return
    try:
        text = env_file.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)
