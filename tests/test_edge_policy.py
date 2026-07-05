"""Tests for Google-edge runtime policy."""

from hermes.edge_policy import (
    DeviceTier,
    Runtime,
    TaskClass,
    assert_local_first,
    choose_profile,
    profiles,
)


def test_required_profiles_are_local_first():
    assert_local_first()


def test_tool_tasks_skip_llm():
    decision = choose_profile(TaskClass.TOOL, DeviceTier.LOW, available_ram_mb=512)
    assert decision.profile.id == "tool-first"
    assert decision.profile.runtime is Runtime.DETERMINISTIC


def test_gemini_nano_only_when_preferred_and_npu():
    decision = choose_profile(
        TaskClass.CHAT,
        DeviceTier.NPU,
        available_ram_mb=1024,
        prefer_system_model=True,
    )
    assert decision.profile.id == "gemini-nano-aicore"


def test_mtp_route_wins_when_available_and_fits():
    decision = choose_profile(
        TaskClass.REASONING,
        DeviceTier.HIGH,
        available_ram_mb=4096,
        mtp_available=True,
    )
    assert decision.profile.id == "gemma-4-e2b-mtp-litert"


def test_gemma_3n_e2b_baseline_for_low_memory():
    decision = choose_profile(TaskClass.CHAT, DeviceTier.LOW, available_ram_mb=2400)
    assert decision.profile.id == "gemma-3n-e2b-int4-litert"
    assert decision.profile.requires_network is False


def test_cloud_fallback_disabled_when_no_local_fit():
    decision = choose_profile(TaskClass.CHAT, DeviceTier.LOW, available_ram_mb=256)
    assert decision.profile.id == "cloud-fallback-disabled"
    assert decision.profile.optional is True
    assert decision.profile.requires_network is True


def test_registry_has_google_edge_profiles():
    ids = {p.id for p in profiles()}
    assert "gemma-3n-e2b-int4-litert" in ids
    assert "gemma-3n-e4b-int4-litert" in ids
    assert "gemini-nano-aicore" in ids
