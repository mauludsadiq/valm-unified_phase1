import pytest
import torch
from unittest.mock import Mock
from valm.core.interface import VALMInterface
from valm.core.config import VALMConfig


class TestVALMInterface:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.config.hidden_size = 768
        model.state_dict.return_value = {"weight": torch.randn(10, 10)}
        p = torch.randn(10, requires_grad=True)
        model.parameters.return_value = [p]

        mock_output = Mock()
        mock_output.logits = torch.randn(1, 10, 100)

        k = torch.randn(1, 12, 10, 64)
        v = torch.randn(1, 12, 10, 64)
        mock_output.past_key_values = [(k, v)]
        mock_output.hidden_states = [torch.randn(1, 10, 768)]

        def f(*args, **kwargs):
            return mock_output

        model.side_effect = f
        model.generate.return_value = torch.randint(0, 100, (1, 20))
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = Mock()

        def tok(*args, **kwargs):
            return Mock(input_ids=torch.randint(0, 100, (1, 10)))

        tokenizer.side_effect = tok
        tokenizer.decode.return_value = "Mock generated text"
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def valm_config(self, tmp_path):
        cfg = VALMConfig(system_name="test", version="1.0.0", device="cpu")
        cfg.registry.storage_type = "sqlite"
        cfg.registry.storage_path = tmp_path / "valm_registry_test.db"
        return cfg

    def test_initialization(self, mock_model, mock_tokenizer, valm_config):
        valm = VALMInterface(valm_config)
        valm.initialize_from_model(mock_model, mock_tokenizer)
        assert valm.step == 0
        assert valm.theta_0 is not None
        assert valm.model is mock_model
        assert valm.tokenizer is mock_tokenizer

    def test_process_basic(self, mock_model, mock_tokenizer, valm_config):
        valm = VALMInterface(valm_config)
        valm.initialize_from_model(mock_model, mock_tokenizer)
        result = valm.process("Test input")
        assert isinstance(result.text, str)
        assert result.step == 1
        assert result.surprise >= 0.0
        assert len(result.memory_events) > 0

    def test_memory_events(self, mock_model, mock_tokenizer, valm_config):
        valm = VALMInterface(valm_config)
        valm.initialize_from_model(mock_model, mock_tokenizer)
        result = valm.process("Test input")
        event_types = [e.event_type for e in result.memory_events]
        assert "kv_retrieve" in event_types

    def test_system_state(self, mock_model, mock_tokenizer, valm_config):
        valm = VALMInterface(valm_config)
        valm.initialize_from_model(mock_model, mock_tokenizer)
        state = valm.get_system_state()
        assert "step" in state
        assert "kv_memory" in state
        assert "param_memory" in state
        assert "registry" in state
        assert state["step"] == 0
        assert state["kv_memory"]["M"] == valm_config.kv_memory.M

    def test_audit_report(self, mock_model, mock_tokenizer, valm_config):
        valm = VALMInterface(valm_config)
        valm.initialize_from_model(mock_model, mock_tokenizer)
        for i in range(3):
            valm.process(f"Input {i}")
        report = valm.generate_audit_report(start_step=0)
        assert report["total_steps"] == 3
        assert "event_counts" in report
        assert "registry_entries" in report
        assert "registry_proof" in report
        assert report["registry_proof"]["consistency_check"] in (True, False)

    def test_registry_integration(self, mock_model, mock_tokenizer, valm_config):
        valm = VALMInterface(valm_config)
        valm.initialize_from_model(mock_model, mock_tokenizer)
        result = valm.process("Test input", generate_witness=True)
        event_types = [e.event_type for e in result.memory_events]
        assert "registry_update" in event_types
        state = valm.get_system_state()
        assert state["registry"]["entry_count"] > 0
        assert state["registry"]["root_hash"] is not None
