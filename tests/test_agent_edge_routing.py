"""Tests for agent integration with Google-edge routing policy."""

from pathlib import Path

from hermes.agent import ModelManager
from hermes.edge_policy import DeviceTier
from hermes.router import INTENT_CHAT, INTENT_REASONING, INTENT_TOOLS


def _model_file(path: Path) -> str:
    path.write_bytes(b"LITERTLM" + b"\0" * 32)
    return str(path)


def test_model_manager_prefers_registered_google_edge_profile(tmp_path):
    manager = ModelManager(device_tier=DeviceTier.HIGH, available_ram_mb=4096)
    manager.register("gemma-3n-e4b-int4-litert", _model_file(tmp_path / "e4b.litertlm"))
    manager.register("_hot", _model_file(tmp_path / "hot.litertlm"))

    model = manager.resolve_edge(INTENT_CHAT)

    assert manager.current_key == "gemma-3n-e4b-int4-litert"
    assert manager.last_route_decision is not None
    assert manager.last_route_decision.profile.id == "gemma-3n-e4b-int4-litert"
    assert Path(model.model_path).name == "e4b.litertlm"


def test_model_manager_uses_hot_model_when_profile_not_registered(tmp_path):
    manager = ModelManager(device_tier=DeviceTier.LOW, available_ram_mb=2400)
    manager.register("_hot", _model_file(tmp_path / "hot.litertlm"))

    model = manager.resolve_edge(INTENT_CHAT)

    assert manager.current_key == "_hot"
    assert manager.last_route_decision is not None
    assert manager.last_route_decision.profile.id == "gemma-3n-e2b-int4-litert"
    assert Path(model.model_path).name == "hot.litertlm"


def test_model_manager_tool_intent_avoids_llm_profile_but_falls_back_to_tool_model(tmp_path):
    manager = ModelManager(device_tier=DeviceTier.LOW, available_ram_mb=512)
    manager.register(INTENT_TOOLS, _model_file(tmp_path / "tools.litertlm"))

    model = manager.resolve_edge(INTENT_TOOLS)

    assert manager.current_key == INTENT_TOOLS
    assert manager.last_route_decision is not None
    assert manager.last_route_decision.profile.id == "tool-first"
    assert Path(model.model_path).name == "tools.litertlm"


def test_model_manager_can_select_benchmark_gated_mtp_profile(tmp_path):
    manager = ModelManager(
        device_tier=DeviceTier.HIGH,
        available_ram_mb=4096,
        mtp_available=True,
    )
    manager.register("gemma-4-e2b-mtp-litert", _model_file(tmp_path / "mtp.litertlm"))
    manager.register(INTENT_REASONING, _model_file(tmp_path / "reasoning.litertlm"))

    model = manager.resolve_edge(INTENT_REASONING)

    assert manager.current_key == "gemma-4-e2b-mtp-litert"
    assert manager.last_route_decision is not None
    assert manager.last_route_decision.profile.id == "gemma-4-e2b-mtp-litert"
    assert Path(model.model_path).name == "mtp.litertlm"
