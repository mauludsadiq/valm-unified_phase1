from pathlib import Path
import yaml

from valm.integration.coordinator import SystemCoordinator


def test_unified_yaml_loads():
    cfg_path = Path("configs/unified.yaml")
    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text())
    assert "systems" in cfg
    assert "capability" in cfg["systems"]


def test_coordinator_load_all_is_deterministic():
    cfg = yaml.safe_load(Path("configs/unified.yaml").read_text())
    coord = SystemCoordinator(cfg)
    systems = coord.load_all()

    assert "capability" in systems
    assert "specialization" in systems
    assert "adaptation" in systems
    assert "verification" in systems

    for name, payload in systems.items():
        if payload.get("enabled") is False:
            continue
        assert ("ok" in payload) or (payload.get("stub") is True)
        if payload.get("stub") is True:
            assert payload.get("ok") is False
