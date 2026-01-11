from __future__ import annotations

from valm.unified.witness import make_unified_load_witness


def test_unified_witness_has_expected_shape() -> None:
    w = make_unified_load_witness("configs/unified.yaml")
    assert w["witness_type"] == "valm_unified_load_witness"
    assert w["witness_version"] == "0.1.0"
    assert "config" in w and "sha256" in w["config"]
    assert "systems" in w
    for k in ["adaptation", "capability", "specialization", "verification"]:
        assert k in w["systems"]
        s = w["systems"][k]
        assert "status" in s
        assert "imports" in s
