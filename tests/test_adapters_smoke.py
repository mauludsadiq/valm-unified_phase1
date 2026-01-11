from __future__ import annotations

from valm.unified.integrated import IntegratedVALM


def test_adapters_construct_and_anchor_imports() -> None:
    integ = IntegratedVALM.from_yaml("configs/unified.yaml")
    st = integ.adapter_status()

    assert st["adaptation"]["ok"] is True
    assert st["adaptation"]["module"] == "surprise_ttt.ttt"
    assert st["adaptation"]["symbol"] == "run_ttt"

    assert st["capability"]["ok"] is True
    assert st["capability"]["module"] == "llm_nature"

    assert st["specialization"]["ok"] is True

    assert st["verification"]["ok"] is True
    assert st["verification"]["module"] == "cbench.verifier_v0_1"
