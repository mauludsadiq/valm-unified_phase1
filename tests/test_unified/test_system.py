from valm.unified.system import VALMUnified


def test_valm_unified_loads_config_and_reports():
    u = VALMUnified.from_yaml("configs/unified.yaml")
    rpt = u.report_loaded()
    assert "capability" in rpt
    assert "verification" in rpt
    assert "repo_path" in rpt["capability"]
