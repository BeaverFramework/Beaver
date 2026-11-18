from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

DEFAULT_CACHE_DIR = "temp/semantic_constraint_cache"


class SemanticConstraintCache:
    def __init__(self, dataset_name: str, cache_dir: str | None = None):
        # BEAVER_SEMANTIC_CACHE_DIR: put the sqlite file on node-local disk.
        # With 32 workers checking Phi on every prefix, a WAL on NFS/Lustre
        # was 99% of wall-clock (dist300 pc_eps1e-3_d2500 runs, 2026-09-16).
        if cache_dir is None:
            cache_dir = os.environ.get("BEAVER_SEMANTIC_CACHE_DIR", DEFAULT_CACHE_DIR)
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_path / f"{dataset_name}.db"
        # SQLite connections cannot be shared across threads. Keep one
        # connection per thread so parallel group workers each get their own.
        self._local = threading.local()
        conn = self._connect()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, result INTEGER NOT NULL)"
        )
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=120.0)
            conn.execute("PRAGMA busy_timeout=120000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    @staticmethod
    def make_key(sequence: str, instance_context: str = "") -> str:
        # Raw sequence: trailing whitespace/newlines are part of what Phi
        # sees (see base_constraints._enforce), so they must be part of the key.
        return hashlib.sha256(
            f"{instance_context}\x00{sequence}".encode()
        ).hexdigest()

    def get(self, key: str) -> Optional[bool]:
        row = (
            self._connect()
            .execute("SELECT result FROM cache WHERE key = ?", (key,))
            .fetchone()
        )
        return bool(row[0]) if row is not None else None

    def set(self, key: str, result: bool):
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, result) VALUES (?, ?)",
            (key, int(result)),
        )
        conn.commit()

    def get_batch(self, keys: list) -> list:
        found = {}
        conn = self._connect()
        for i in range(0, len(keys), 900):
            chunk = keys[i : i + 900]
            placeholders = ",".join("?" * len(chunk))
            for row in conn.execute(
                f"SELECT key, result FROM cache WHERE key IN ({placeholders})", chunk
            ):
                found[row[0]] = bool(row[1])
        return [found.get(k) for k in keys]

    def set_batch(self, keys: list, results: list):
        conn = self._connect()
        conn.executemany(
            "INSERT OR REPLACE INTO cache (key, result) VALUES (?, ?)",
            [(k, int(r)) for k, r in zip(keys, results)],
        )
        conn.commit()

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __del__(self):
        self.close()


_cache_registry: dict = {}


def _get_cache(dataset_name: str) -> SemanticConstraintCache:
    if dataset_name not in _cache_registry:
        _cache_registry[dataset_name] = SemanticConstraintCache(dataset_name)
    return _cache_registry[dataset_name]
