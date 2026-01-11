from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _json_canonical_bytes(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hash_pair(left: bytes, right: bytes) -> bytes:
    return _sha256(left + right)


def _build_merkle_levels(leaves: List[bytes]) -> List[List[bytes]]:
    if len(leaves) == 0:
        return [[_sha256(b"")]]
    levels: List[List[bytes]] = []
    cur = list(leaves)
    levels.append(cur)
    while len(cur) > 1:
        nxt: List[bytes] = []
        i = 0
        while i < len(cur):
            a = cur[i]
            b = cur[i + 1] if i + 1 < len(cur) else cur[i]
            nxt.append(_hash_pair(a, b))
            i += 2
        cur = nxt
        levels.append(cur)
    return levels


def _merkle_root(leaves: List[bytes]) -> bytes:
    levels = _build_merkle_levels(leaves)
    return levels[-1][0]


def _merkle_path(leaves: List[bytes], index: int) -> List[Dict[str, str]]:
    levels = _build_merkle_levels(leaves)
    if index < 0 or index >= len(levels[0]):
        raise IndexError("leaf index out of range")
    path: List[Dict[str, str]] = []
    idx = index
    for level in levels[:-1]:
        sib = idx ^ 1
        if sib >= len(level):
            sib = idx
        side = "left" if sib < idx else "right"
        path.append({"side": side, "hash": level[sib].hex()})
        idx = idx // 2
    return path


def verify_inclusion_proof(leaf_hash_hex: str, root_hash_hex: str, path: List[Dict[str, str]]) -> bool:
    try:
        h = bytes.fromhex(leaf_hash_hex)
        for step in path:
            side = step["side"]
            s = bytes.fromhex(step["hash"])
            if side == "left":
                h = _hash_pair(s, h)
            else:
                h = _hash_pair(h, s)
        return h.hex() == root_hash_hex
    except Exception:
        return False


@dataclass(frozen=True)
class RegistryConfigLike:
    storage_type: str
    storage_path: Optional[Path] = None


class StateRegistry:
    def __init__(self, config):
        self.config = config
        self.root_hash: Optional[str] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._mem_entries: List[Dict[str, Any]] = []
        self._mem_leaf_hashes: List[str] = []
        if self.config.storage_type == "sqlite":
            path = Path(self.config.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._create_tables()
        self._create_genesis()

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS registry_entries (id INTEGER PRIMARY KEY, entry_json TEXT NOT NULL, leaf_hash TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS registry_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
        )
        self._conn.commit()

    def _set_meta(self, k: str, v: str) -> None:
        assert self._conn is not None
        self._conn.execute("INSERT OR REPLACE INTO registry_meta (k, v) VALUES (?, ?)", (k, v))
        self._conn.commit()

    def _get_meta(self, k: str) -> Optional[str]:
        assert self._conn is not None
        cur = self._conn.execute("SELECT v FROM registry_meta WHERE k = ?", (k,))
        row = cur.fetchone()
        if row is None:
            return None
        return str(row[0])

    def _create_genesis(self) -> None:
        genesis = {"type": "genesis", "v": 1}
        if self.config.storage_type == "sqlite":
            assert self._conn is not None
            cur = self._conn.execute("SELECT COUNT(*) FROM registry_entries WHERE id = 0")
            n = int(cur.fetchone()[0])
            if n == 0:
                leaf = _sha256(_json_canonical_bytes(genesis)).hex()
                self._conn.execute(
                    "INSERT INTO registry_entries (id, entry_json, leaf_hash) VALUES (?, ?, ?)",
                    (0, json.dumps(genesis), leaf),
                )
                self._conn.commit()
            self._recompute_root()
        else:
            if len(self._mem_entries) == 0:
                self._mem_entries.append(genesis)
                self._mem_leaf_hashes.append(_sha256(_json_canonical_bytes(genesis)).hex())
            self._recompute_root()

    def count_entries(self) -> int:
        if self.config.storage_type == "sqlite":
            assert self._conn is not None
            cur = self._conn.execute("SELECT COUNT(*) FROM registry_entries")
            return int(cur.fetchone()[0])
        return len(self._mem_entries)

    def _fetch_leaf_hashes_ordered(self) -> List[str]:
        if self.config.storage_type == "sqlite":
            assert self._conn is not None
            cur = self._conn.execute("SELECT id, leaf_hash FROM registry_entries ORDER BY id ASC")
            rows = cur.fetchall()
            if not rows:
                return []
            max_id = int(rows[-1][0])
            out: List[Optional[str]] = [None] * (max_id + 1)
            for rid, h in rows:
                out[int(rid)] = str(h)
            return [x for x in out if x is not None]
        return list(self._mem_leaf_hashes)

    def _recompute_root(self) -> None:
        leaf_hex = self._fetch_leaf_hashes_ordered()
        leaves = [bytes.fromhex(h) for h in leaf_hex]
        root = _merkle_root(leaves).hex()
        self.root_hash = root
        if self.config.storage_type == "sqlite":
            self._set_meta("root_hash", root)

    def append(self, entry: Dict[str, Any]) -> str:
        leaf = _sha256(_json_canonical_bytes(entry)).hex()
        if self.config.storage_type == "sqlite":
            assert self._conn is not None
            cur = self._conn.execute("SELECT COALESCE(MAX(id), -1) FROM registry_entries")
            max_id = int(cur.fetchone()[0])
            new_id = max_id + 1
            self._conn.execute(
                "INSERT INTO registry_entries (id, entry_json, leaf_hash) VALUES (?, ?, ?)",
                (new_id, json.dumps(entry, ensure_ascii=False), leaf),
            )
            self._conn.commit()
        else:
            self._mem_entries.append(entry)
            self._mem_leaf_hashes.append(leaf)
        self._recompute_root()
        return leaf

    def get_inclusion_proof(self, leaf_index: int) -> Dict[str, Any]:
        leaf_hex = self._fetch_leaf_hashes_ordered()
        if leaf_index < 0 or leaf_index >= len(leaf_hex):
            raise IndexError("leaf index out of range")
        leaves = [bytes.fromhex(h) for h in leaf_hex]
        root = _merkle_root(leaves).hex()
        path = _merkle_path(leaves, leaf_index)
        leaf_hash = leaf_hex[leaf_index]
        verified = verify_inclusion_proof(leaf_hash, root, path)
        return {
            "leaf_index": int(leaf_index),
            "leaf_hash": leaf_hash,
            "root_hash": root,
            "path": path,
            "verified": bool(verified),
        }

    def get_proof(self, start_index: int, end_index: int) -> Dict[str, Any]:
        if end_index < start_index:
            raise ValueError("end_index must be >= start_index")
        leaf_hex = self._fetch_leaf_hashes_ordered()
        leaves = [bytes.fromhex(h) for h in leaf_hex]
        root = _merkle_root(leaves).hex()
        items: List[Dict[str, Any]] = []
        ok = True
        for i in range(start_index, end_index + 1):
            p = self.get_inclusion_proof(i)
            items.append(p)
            ok = ok and bool(p["verified"])
        return {
            "start_index": int(start_index),
            "end_index": int(end_index),
            "leaf_count": int(end_index - start_index + 1),
            "root_hash": root,
            "items": items,
            "verified": bool(ok),
        }

    def get_audit_summary(self) -> Dict[str, Any]:
        return {
            "entry_count": self.count_entries(),
            "root_hash": self.root_hash,
        }
