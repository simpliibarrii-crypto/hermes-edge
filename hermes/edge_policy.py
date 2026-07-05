"""Google-edge runtime policy for Hermes Edge.

This module keeps Hermes fast by making the largest model the last resort.
It is intentionally dependency-free so it can run on phones, test hosts, and CI.

Policy goals:
- local-first: no required cloud or paid API
- Google-edge aware: LiteRT-LM, Gemma, Gemini Nano/AICore when available
- benchmarkable: every route carries expected latency/memory intent
- reasonable: prefer deterministic tools and small models before larger models
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Runtime(str, Enum):
    """Supported runtime families, ordered from cheapest to heaviest."""

    DETERMINISTIC = "deterministic"
    LITERT_LM = "litert_lm"
    GEMINI_NANO = "gemini_nano"
    CLOUD_DISABLED = "cloud_disabled"


class DeviceTier(str, Enum):
    """Coarse device tiers used before real-device benchmarking exists."""

    LOW = "low"          # <= 4 GB RAM, older Android, low-end laptop
    MID = "mid"          # 4-8 GB RAM
    HIGH = "high"        # flagship phone / modern laptop
    NPU = "npu"          # Android device exposing NPU/AICore/GPU path


class TaskClass(str, Enum):
    """Task classes that drive model/runtime choice."""

    TOOL = "tool"
    CHAT = "chat"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"
    CODE = "code"


@dataclass(frozen=True)
class ModelProfile:
    """A benchmarkable model/runtime option."""

    id: str
    runtime: Runtime
    device_tiers: tuple[DeviceTier, ...]
    task_classes: tuple[TaskClass, ...]
    local_first: bool
    requires_network: bool
    optional: bool
    model_family: str
    quantization: str
    memory_mb: int
    notes: str


@dataclass(frozen=True)
class RouteDecision:
    """Resolved route for one prompt/task."""

    profile: ModelProfile
    reason: str
    fallback_profile_id: str | None = None


MODEL_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        id="tool-first",
        runtime=Runtime.DETERMINISTIC,
        device_tiers=(DeviceTier.LOW, DeviceTier.MID, DeviceTier.HIGH, DeviceTier.NPU),
        task_classes=(TaskClass.TOOL,),
        local_first=True,
        requires_network=False,
        optional=False,
        model_family="none",
        quantization="n/a",
        memory_mb=8,
        notes="Use deterministic tools, calculators, retrieval, cache, or rules before LLM.",
    ),
    ModelProfile(
        id="gemma-3n-e2b-int4-litert",
        runtime=Runtime.LITERT_LM,
        device_tiers=(DeviceTier.LOW, DeviceTier.MID, DeviceTier.HIGH, DeviceTier.NPU),
        task_classes=(TaskClass.CHAT, TaskClass.REASONING, TaskClass.MULTIMODAL),
        local_first=True,
        requires_network=False,
        optional=False,
        model_family="gemma-3n-e2b",
        quantization="int4",
        memory_mb=2200,
        notes="Reliable Google edge baseline: MatFormer/PLE design, mobile multimodal-ready.",
    ),
    ModelProfile(
        id="gemma-3n-e4b-int4-litert",
        runtime=Runtime.LITERT_LM,
        device_tiers=(DeviceTier.MID, DeviceTier.HIGH, DeviceTier.NPU),
        task_classes=(TaskClass.CHAT, TaskClass.REASONING, TaskClass.MULTIMODAL, TaskClass.CODE),
        local_first=True,
        requires_network=False,
        optional=True,
        model_family="gemma-3n-e4b",
        quantization="int4",
        memory_mb=3600,
        notes="Higher quality local route when RAM allows; still edge-first.",
    ),
    ModelProfile(
        id="gemma-4-e2b-mtp-litert",
        runtime=Runtime.LITERT_LM,
        device_tiers=(DeviceTier.MID, DeviceTier.HIGH, DeviceTier.NPU),
        task_classes=(TaskClass.CHAT, TaskClass.REASONING, TaskClass.CODE),
        local_first=True,
        requires_network=False,
        optional=True,
        model_family="gemma-4-e2b",
        quantization="int4/mtp",
        memory_mb=2600,
        notes="Fast path candidate when LiteRT-LM MTP/speculative decoding support is available.",
    ),
    ModelProfile(
        id="gemini-nano-aicore",
        runtime=Runtime.GEMINI_NANO,
        device_tiers=(DeviceTier.NPU,),
        task_classes=(TaskClass.CHAT, TaskClass.REASONING, TaskClass.MULTIMODAL),
        local_first=True,
        requires_network=False,
        optional=True,
        model_family="gemini-nano",
        quantization="system-managed",
        memory_mb=0,
        notes="Use Android system model only when exposed by device/AICore; never required.",
    ),
    ModelProfile(
        id="cloud-fallback-disabled",
        runtime=Runtime.CLOUD_DISABLED,
        device_tiers=(DeviceTier.LOW, DeviceTier.MID, DeviceTier.HIGH, DeviceTier.NPU),
        task_classes=(TaskClass.CHAT, TaskClass.REASONING, TaskClass.MULTIMODAL, TaskClass.CODE),
        local_first=False,
        requires_network=True,
        optional=True,
        model_family="none",
        quantization="n/a",
        memory_mb=0,
        notes="Placeholder only. Cloud fallback is disabled by default by project policy.",
    ),
)


def profiles() -> tuple[ModelProfile, ...]:
    """Return immutable profile registry."""

    return MODEL_PROFILES


def _fits(profile: ModelProfile, tier: DeviceTier, task: TaskClass, available_ram_mb: int) -> bool:
    return (
        tier in profile.device_tiers
        and task in profile.task_classes
        and profile.memory_mb <= max(available_ram_mb, 0)
        and not profile.requires_network
    )


def choose_profile(
    task: TaskClass,
    device_tier: DeviceTier,
    available_ram_mb: int,
    *,
    prefer_system_model: bool = False,
    mtp_available: bool = False,
) -> RouteDecision:
    """Choose fastest reasonable local model/runtime for task/device.

    The decision order encodes current Google-edge guidance:
    deterministic tools first, then system Gemini Nano when explicitly available,
    then MTP/speculative Gemma candidate, then mature Gemma 3n LiteRT profiles.
    """

    if task is TaskClass.TOOL:
        return RouteDecision(MODEL_PROFILES[0], "tool-class task: avoid model call")

    if prefer_system_model:
        nano = _by_id("gemini-nano-aicore")
        if _fits(nano, device_tier, task, available_ram_mb):
            return RouteDecision(nano, "device exposes Gemini Nano/AICore local path")

    if mtp_available:
        mtp = _by_id("gemma-4-e2b-mtp-litert")
        if _fits(mtp, device_tier, task, available_ram_mb):
            return RouteDecision(mtp, "MTP/speculative LiteRT-LM path available")

    # Quality route if memory allows.
    e4b = _by_id("gemma-3n-e4b-int4-litert")
    if _fits(e4b, device_tier, task, available_ram_mb):
        return RouteDecision(e4b, "Gemma 3n E4B fits memory budget", "gemma-3n-e2b-int4-litert")

    e2b = _by_id("gemma-3n-e2b-int4-litert")
    if _fits(e2b, device_tier, task, available_ram_mb):
        return RouteDecision(e2b, "Gemma 3n E2B is safest local baseline")

    return RouteDecision(
        _by_id("cloud-fallback-disabled"),
        "no local profile fits current memory budget; cloud remains disabled by default",
        "gemma-3n-e2b-int4-litert",
    )


def _by_id(profile_id: str) -> ModelProfile:
    for profile in MODEL_PROFILES:
        if profile.id == profile_id:
            return profile
    raise KeyError(profile_id)


def assert_local_first(registry: Iterable[ModelProfile] = MODEL_PROFILES) -> None:
    """Fail if a required profile needs network/cloud access."""

    offenders = [p.id for p in registry if not p.optional and (p.requires_network or not p.local_first)]
    if offenders:
        raise AssertionError(f"required profiles must be local-first: {offenders}")
