from valm.core.registry import StateRegistry
from valm.core.config import RegistryConfig


def test_registry_append_and_root(tmp_path):
    cfg = RegistryConfig(storage_type="sqlite", storage_path=tmp_path/"reg.db")
    reg = StateRegistry(cfg)

    h1 = reg.append({"step": 1, "x": 1})
    h2 = reg.append({"step": 2, "x": 2})

    assert h1 != h2
    assert reg.root_hash is not None
    proof = reg.get_proof(1, 2)
    assert proof["root_hash"] == reg.root_hash
    assert proof["leaf_count"] == 2
