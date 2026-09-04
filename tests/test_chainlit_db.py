import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.chainlit_db import (
    CHAINLIT_SQLITE_SCHEMA,
    default_chainlit_db_path,
    ensure_chainlit_sqlite_schema,
    get_chainlit_conninfo,
    resolve_chainlit_db_path,
    sqlite_path_from_conninfo,
)


class TestChainlitDb(unittest.TestCase):
    def test_default_conninfo_uses_project_chainlit_dir(self):
        with patch.dict(os.environ, {}, clear=True):
            conninfo = get_chainlit_conninfo()
        self.assertIn("chat_history.db", conninfo)
        path = sqlite_path_from_conninfo(conninfo)
        self.assertEqual(path, default_chainlit_db_path())

    def test_ensure_schema_creates_users_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            conninfo = f"sqlite+aiosqlite:///{db_path}"
            ensure_chainlit_sqlite_schema(conninfo)
            with sqlite3.connect(db_path) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            self.assertIn("users", tables)
            self.assertIn("threads", tables)
            self.assertIn("steps", tables)
            self.assertIn("elements", tables)
            self.assertIn("feedbacks", tables)

    def test_schema_sql_is_non_empty(self):
        self.assertIn("CREATE TABLE IF NOT EXISTS users", CHAINLIT_SQLITE_SCHEMA)

    def test_unwritable_chainlit_db_falls_back_to_project_default(self):
        with patch.dict(
            os.environ,
            {"CHAINLIT_DB": "sqlite+aiosqlite:////data/chat_history.db"},
            clear=True,
        ):
            with patch("utils.chainlit_db._can_write_db_parent", return_value=False):
                path = resolve_chainlit_db_path()
        self.assertEqual(path, default_chainlit_db_path())


if __name__ == "__main__":
    unittest.main()
