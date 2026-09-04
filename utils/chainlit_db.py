"""SQLite schema bootstrap for Chainlit SQLAlchemyDataLayer."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from sqlalchemy.engine.url import make_url

logger = logging.getLogger(__name__)

# SQLite-compatible schema (TEXT instead of UUID/JSONB/TEXT[], INTEGER for booleans).
CHAINLIT_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL,
    "createdAt" TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "parentId" TEXT,
    "streaming" INTEGER NOT NULL DEFAULT 0,
    "waitForAnswer" INTEGER,
    "isError" INTEGER,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INTEGER,
    "defaultOpen" INTEGER,
    "modes" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INTEGER,
    "language" TEXT,
    "forId" TEXT,
    "mime" TEXT,
    "props" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT NOT NULL,
    "threadId" TEXT NOT NULL,
    "value" INTEGER NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);
"""


def default_chainlit_db_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".chainlit" / "chat_history.db"


def sqlite_path_from_conninfo(conninfo: str) -> Path | None:
    """Extract a filesystem path from a sqlite SQLAlchemy conninfo string."""
    if not conninfo.startswith("sqlite"):
        return None
    database = make_url(conninfo).database
    if not database:
        return None
    return Path(database)


def _can_write_db_parent(db_path: Path) -> bool:
    parent = db_path.parent
    if parent.exists():
        return os.access(parent, os.W_OK)
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def resolve_chainlit_db_path() -> Path:
    """
    Pick a writable SQLite path.

    Uses CHAINLIT_DB when set and writable; otherwise falls back to
    <project>/.chainlit/chat_history.db (local dev).
    """
    env = os.environ.get("CHAINLIT_DB")
    if env and env.startswith("sqlite"):
        configured = sqlite_path_from_conninfo(env)
        if configured:
            path = configured if configured.is_absolute() else Path.cwd() / configured
            if _can_write_db_parent(path):
                return path
            logger.warning(
                "CHAINLIT_DB path %s is not writable; using local default %s",
                path,
                default_chainlit_db_path(),
            )
    return default_chainlit_db_path()


def get_chainlit_conninfo() -> str:
    env = os.environ.get("CHAINLIT_DB")
    if env and not env.startswith("sqlite"):
        return env
    db_path = resolve_chainlit_db_path()
    return f"sqlite+aiosqlite:///{db_path.as_posix()}"


def ensure_chainlit_sqlite_schema(conninfo: str) -> None:
    """Create Chainlit tables when using SQLite. No-op for other backends."""
    db_path = sqlite_path_from_conninfo(conninfo)
    if db_path is None:
        return
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    if not _can_write_db_parent(db_path):
        raise OSError(f"Cannot create Chainlit database at {db_path}")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(CHAINLIT_SQLITE_SCHEMA)
