"""Deterministic workflow-agent benchmarks for Hermes Edge.

The helpers in this module use a virtual clock and local artifact hashes. They
exercise orchestration policy without starting real jobs, requiring a scheduler,
or introducing cloud dependencies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

JobStatus = Literal["queued", "running", "succeeded", "failed"]
MonitorStrategy = Literal["event", "poll"]
VALID_STATUSES = frozenset({"queued", "running", "succeeded", "failed"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
DEFAULT_ALLOWED_BACKENDS = frozenset({"cpu", "gpu", "npu", "litert-lm"})
DEFAULT_ALLOWED_AFFINITIES = frozenset({"auto", "balanced", "performance", "efficiency"})
SCHEMA = "hermes.workflow_agent_benchmark.v1"


@dataclass(frozen=True, order=True)
class JobEvent:
    """A state transition at a deterministic virtual-clock offset."""

    offset_ms: int
    status: JobStatus

    def __post_init__(self) -> None:
        if self.offset_ms < 0:
            raise ValueError("Job event offsets must be non-negative.")


@dataclass(frozen=True)
class MonitorMetrics:
    """JSON-safe orchestration measurements for one monitor strategy."""

    monitor_strategy: MonitorStrategy
    status_checks: int
    idle_wait_ms: int
    completion_latency_ms: int
    timeout_count: int
    final_status: JobStatus

    def to_dict(self) -> dict[str, int | str]:
        return {
            "monitor_strategy": self.monitor_strategy,
            "status_checks": self.status_checks,
            "idle_wait_ms": self.idle_wait_ms,
            "completion_latency_ms": self.completion_latency_ms,
            "timeout_count": self.timeout_count,
            "final_status": self.final_status,
        }


@dataclass(frozen=True)
class WorkflowBenchmarkRecord:
    """One JSONL row for the optional ``workflow-agent`` scenario."""

    scenario_id: str
    backend: str
    affinity: str
    metrics: MonitorMetrics
    output_equivalent: bool
    terminal_artifact_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "scenario": "workflow-agent",
            "scenario_id": self.scenario_id,
            "backend": self.backend,
            "affinity": self.affinity,
            **self.metrics.to_dict(),
            "output_equivalent": self.output_equivalent,
            "terminal_artifact_hash": self.terminal_artifact_hash,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _validated_events(events: Iterable[JobEvent]) -> tuple[JobEvent, ...]:
    timeline = tuple(events)
    if not timeline:
        raise ValueError("A workflow benchmark requires at least one job event.")
    if any(event.status not in VALID_STATUSES for event in timeline):
        raise ValueError("Job events contain an unsupported status.")
    if timeline[0].offset_ms != 0 or timeline[0].status != "queued":
        raise ValueError("A workflow timeline must start with queued at offset 0.")
    if any(left.offset_ms >= right.offset_ms for left, right in zip(timeline, timeline[1:])):
        raise ValueError("Job event offsets must be strictly increasing.")
    if any(event.status in TERMINAL_STATUSES for event in timeline[:-1]):
        raise ValueError("A terminal job status must be the final event.")
    return timeline


def monitor_event_driven(events: Iterable[JobEvent], timeout_ms: int) -> MonitorMetrics:
    """Observe state transitions directly, without active polling."""

    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative.")
    timeline = _validated_events(events)
    visible = tuple(event for event in timeline if event.offset_ms <= timeout_ms)
    terminal = next((event for event in visible if event.status in TERMINAL_STATUSES), None)
    if terminal is not None:
        return MonitorMetrics(
            monitor_strategy="event",
            status_checks=len(visible),
            idle_wait_ms=terminal.offset_ms,
            completion_latency_ms=terminal.offset_ms,
            timeout_count=0,
            final_status=terminal.status,
        )
    final_status = visible[-1].status if visible else "queued"
    return MonitorMetrics(
        monitor_strategy="event",
        status_checks=len(visible),
        idle_wait_ms=timeout_ms,
        completion_latency_ms=timeout_ms,
        timeout_count=1,
        final_status=final_status,
    )


def _status_at(timeline: Sequence[JobEvent], offset_ms: int) -> JobStatus:
    status: JobStatus = "queued"
    for event in timeline:
        if event.offset_ms > offset_ms:
            break
        status = event.status
    return status


def monitor_polling(
    events: Iterable[JobEvent], timeout_ms: int, poll_interval_ms: int
) -> MonitorMetrics:
    """Poll a virtual job state at a fixed interval until terminal or timeout."""

    if timeout_ms < 0:
        raise ValueError("timeout_ms must be non-negative.")
    if poll_interval_ms <= 0:
        raise ValueError("poll_interval_ms must be positive.")
    timeline = _validated_events(events)
    clock_ms = 0
    checks = 0
    final_status: JobStatus = "queued"
    while clock_ms <= timeout_ms:
        checks += 1
        final_status = _status_at(timeline, clock_ms)
        if final_status in TERMINAL_STATUSES:
            return MonitorMetrics(
                monitor_strategy="poll",
                status_checks=checks,
                idle_wait_ms=clock_ms,
                completion_latency_ms=clock_ms,
                timeout_count=0,
                final_status=final_status,
            )
        clock_ms += poll_interval_ms

    return MonitorMetrics(
        monitor_strategy="poll",
        status_checks=checks,
        idle_wait_ms=timeout_ms,
        completion_latency_ms=timeout_ms,
        timeout_count=1,
        final_status=final_status,
    )


def validate_execution_options(
    backend: str,
    affinity: str,
    *,
    allowed_backends: Iterable[str] = DEFAULT_ALLOWED_BACKENDS,
    allowed_affinities: Iterable[str] = DEFAULT_ALLOWED_AFFINITIES,
) -> tuple[str, str]:
    """Reject syntactically plausible but unsupported backend/affinity options."""

    normalized_backend = backend.strip().lower()
    normalized_affinity = affinity.strip().lower()
    backend_allowlist = {item.strip().lower() for item in allowed_backends}
    affinity_allowlist = {item.strip().lower() for item in allowed_affinities}
    if normalized_backend not in backend_allowlist:
        raise ValueError(f"Unsupported backend option: {backend!r}.")
    if normalized_affinity not in affinity_allowlist:
        raise ValueError(f"Unsupported affinity option: {affinity!r}.")
    return normalized_backend, normalized_affinity


def terminal_artifact_hash(artifacts: Mapping[str, bytes | str]) -> str | None:
    """Hash terminal artifact content while intentionally ignoring path names."""

    if not artifacts:
        return None
    content_hashes: list[str] = []
    for content in artifacts.values():
        payload = content if isinstance(content, bytes) else content.encode("utf-8")
        content_hashes.append(hashlib.sha256(payload).hexdigest())
    canonical = json.dumps(sorted(content_hashes), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def terminal_artifacts_equivalent(
    left: Mapping[str, bytes | str], right: Mapping[str, bytes | str]
) -> bool:
    """Compare scientifically terminal outputs, not workflow names or layout."""

    left_hash = terminal_artifact_hash(left)
    right_hash = terminal_artifact_hash(right)
    return left_hash is not None and left_hash == right_hash


def build_benchmark_record(
    *,
    scenario_id: str,
    backend: str,
    affinity: str,
    metrics: MonitorMetrics,
    actual_terminal_artifacts: Mapping[str, bytes | str],
    expected_terminal_artifacts: Mapping[str, bytes | str],
) -> WorkflowBenchmarkRecord:
    """Create a benchmark row after option and terminal-output validation."""

    if not scenario_id.strip():
        raise ValueError("scenario_id must be non-empty.")
    normalized_backend, normalized_affinity = validate_execution_options(backend, affinity)
    equivalent = terminal_artifacts_equivalent(
        actual_terminal_artifacts, expected_terminal_artifacts
    )
    return WorkflowBenchmarkRecord(
        scenario_id=scenario_id,
        backend=normalized_backend,
        affinity=normalized_affinity,
        metrics=metrics,
        output_equivalent=equivalent,
        terminal_artifact_hash=terminal_artifact_hash(actual_terminal_artifacts),
    )
