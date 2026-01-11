import pytest
from valm.core.config import VALMConfig, KVMemoryConfig, MemoryType


class TestVALMConfig:
    def test_default_config(self):
        config = VALMConfig()
        assert config.system_name == "valm-unified"
        assert config.memory_type == MemoryType.HYBRID
        assert config.kv_memory.M == 256
        assert config.param_memory.trust_radius == 0.05

    def test_yaml_serialization(self, tmp_path):
        config = VALMConfig(system_name="test-system", memory_type=MemoryType.KV)
        yaml_path = tmp_path / "test_config.yaml"
        config.to_yaml(yaml_path)
        loaded = VALMConfig.from_yaml(yaml_path)
        assert loaded.system_name == "test-system"
        assert loaded.memory_type == MemoryType.KV
        assert loaded.kv_memory.M == 256

    def test_validation(self):
        with pytest.raises(ValueError):
            KVMemoryConfig(tau_novel=0.9, tau_reuse=0.8)

        with pytest.raises(ValueError):
            VALMConfig(param_memory={"trust_radius": -0.1})

        config = VALMConfig(
            kv_memory={"tau_novel": 0.3, "tau_reuse": 0.8},
            param_memory={"trust_radius": 0.05},
        )
        assert config.kv_memory.tau_novel == 0.3
        assert config.param_memory.trust_radius == 0.05

    def test_update_from_dict(self):
        config = VALMConfig()
        updated = config.update_from_dict({"system_name": "updated-system", "kv_memory": {"M": 512}})
        assert updated.system_name == "updated-system"
        assert updated.kv_memory.M == 512
        assert updated.param_memory.trust_radius == 0.05
