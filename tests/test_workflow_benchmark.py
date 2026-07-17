"""Tests for deterministic workflow-agent orchestration benchmarks."""

import json

import pytest

from hermes.workflow_benchmark import (
    JobEvent,
    build_benchmark_record,
    monitor_event_driven,
    monitor_polling,
    terminal_artifacts_equivalent,
    validate_execution_options,
)


def _events():
    return (
        JobEvent(0, "queued"),
        JobEvent(15, "running"),
        JobEvent(37, "succeeded"),
    )


def test_event_subscription_uses_fewer_status_checks_than_polling():
    event_metrics = monitor_event_driven(_events(), timeout_ms=100)
    poll_metrics = monitor_polling(_events(), timeout_ms=100, poll_interval_ms=10)

    assert event_metrics.final_status == "succeeded"
    assert poll_metrics.final_status == "succeeded"
    assert event_metrics.status_checks < poll_metrics.status_checks
    assert event_metrics.completion_latency_ms < poll_metrics.completion_latency_ms


@pytest.mark.parametrize(
    ("backend", "affinity"),
    [("cuda-magic", "balanced"), ("cpu", "socket:99")],
)
def test_invalid_backend_or_affinity_is_rejected(backend, affinity):
    with pytest.raises(ValueError):
        validate_execution_options(backend, affinity)


def test_structurally_different_workflows_pass_when_terminal_outputs_match():
    left = {"run-a/results/final.vcf": "chr1\t101\t.\tA\tG\t60\tPASS"}
    right = {"workflow-b/output/calls.vcf": "chr1\t101\t.\tA\tG\t60\tPASS"}

    assert terminal_artifacts_equivalent(left, right) is True


def test_partial_workflow_fails_without_terminal_artifact():
    expected = {"expected/final.vcf": "chr1\t101\t.\tA\tG\t60\tPASS"}

    assert terminal_artifacts_equivalent({}, expected) is False


def test_jsonl_record_contains_required_orchestration_fields():
    record = build_benchmark_record(
        scenario_id="fixture-1",
        backend="CPU",
        affinity="Balanced",
        metrics=monitor_event_driven(_events(), timeout_ms=100),
        actual_terminal_artifacts={"a/final.txt": "same"},
        expected_terminal_artifacts={"b/result.txt": "same"},
    )
    payload = json.loads(record.to_json_line())

    assert payload["schema"] == "hermes.workflow_agent_benchmark.v1"
    assert payload["scenario"] == "workflow-agent"
    assert payload["backend"] == "cpu"
    assert payload["affinity"] == "balanced"
    assert payload["output_equivalent"] is True
    assert {
        "monitor_strategy",
        "status_checks",
        "idle_wait_ms",
        "completion_latency_ms",
        "timeout_count",
    } <= payload.keys()
