#!/usr/bin/env python3
"""Run Hermes Edge's deterministic workflow-agent benchmark fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes.workflow_benchmark import (  # noqa: E402
    JobEvent,
    JobStatus,
    build_benchmark_record,
    monitor_event_driven,
    monitor_polling,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _artifacts(items: list[dict[str, str]]) -> dict[str, str]:
    return {item["path"]: item["content"] for item in items}


def run(fixture_path: Path, output_path: Path) -> int:
    fixture: dict[str, Any] = json.loads(fixture_path.read_text(encoding="utf-8"))
    events = tuple(
        JobEvent(int(item["offset_ms"]), cast(JobStatus, item["status"]))
        for item in fixture["events"]
    )
    timeout_ms = int(fixture["timeout_ms"])
    actual = _artifacts(fixture["actual_terminal_artifacts"])
    expected = _artifacts(fixture["expected_terminal_artifacts"])

    records = [
        build_benchmark_record(
            scenario_id=fixture["scenario_id"],
            backend=fixture["backend"],
            affinity=fixture["affinity"],
            metrics=monitor_event_driven(events, timeout_ms),
            actual_terminal_artifacts=actual,
            expected_terminal_artifacts=expected,
        ),
        build_benchmark_record(
            scenario_id=fixture["scenario_id"],
            backend=fixture["backend"],
            affinity=fixture["affinity"],
            metrics=monitor_polling(
                events, timeout_ms, int(fixture["poll_interval_ms"])
            ),
            actual_terminal_artifacts=actual,
            expected_terminal_artifacts=expected,
        ),
    ]
    lines = "\n".join(record.to_json_line() for record in records) + "\n"
    output_path.write_text(lines, encoding="utf-8")
    print(lines, end="")
    return 0 if all(record.output_equivalent for record in records) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=_REPO_ROOT / "data" / "workflow_agent_fixture.json",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("workflow_benchmark_results.jsonl")
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args.fixture, args.output))
