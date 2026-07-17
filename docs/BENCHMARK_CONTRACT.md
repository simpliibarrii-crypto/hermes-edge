# Benchmark Contract

Hermes Edge is only as fast as measured. This file defines required proof before performance claims.

## Required metrics

| Metric | Meaning |
|---|---|
| TTFT ms | time to first generated token |
| decode tok/s | steady-state generation speed |
| prefill tok/s | prompt ingestion speed |
| peak RSS MB | max process memory |
| model size MB | on-device bundle size |
| route | tool, Gemini Nano, Gemma 3n E2B, Gemma 3n E4B, Gemma 4 MTP |
| backend | CPU, GPU, NPU, AICore, LiteRT-LM |
| device | exact phone/laptop/edge box |
| thermal note | cold/warm/throttled |

## Claim levels

- **Design claim:** architecture expected to be fast, no benchmark needed.
- **Local benchmark claim:** measured on developer machine or emulator.
- **Device benchmark claim:** measured on named physical device.
- **Comparative claim:** requires same prompt, same token budget, same device, same runtime class.

## Forbidden claims without data

- fastest ever
- beats Gemini Nano
- beats Gemma/Gemma 3n
- production-ready mobile runtime
- battery efficient

## Default benchmark set

1. tool route: arithmetic, unit conversion, current date
2. chat route: 64-token prompt, 64-token decode
3. reasoning route: 256-token prompt, 128-token decode
4. retrieval route: 3 snippets, 128-token answer
5. tool-agent route: one function call + final answer
6. workflow-agent route: deterministic orchestration and terminal-output equivalence

## Workflow-agent scenario

The optional `workflow-agent` scenario measures orchestration correctness without starting real schedulers or cloud jobs. It uses a virtual clock and deterministic events such as `queued -> running -> succeeded|failed`.

Every workflow-agent JSONL row must include:

| Field | Meaning |
|---|---|
| `monitor_strategy` | `event` or fixed-interval `poll` |
| `status_checks` | number of state inspections or delivered transitions |
| `idle_wait_ms` | virtual time spent awaiting completion |
| `completion_latency_ms` | virtual time when terminal status was observed |
| `timeout_count` | zero when completed, one when the scenario timed out |
| `backend` | allowlisted execution backend |
| `affinity` | allowlisted processor-affinity policy |
| `output_equivalent` | whether terminal artifact contents match the reference |
| `terminal_artifact_hash` | canonical hash of terminal artifact contents |

Rules:

- Prefer event subscriptions when they produce fewer checks or lower completion latency than polling.
- Reject backend or affinity values outside explicit allowlists, even when a generated command is syntactically valid.
- Compare canonical terminal artifact contents, not directory names, task labels, or workflow layout.
- A partial workflow with no valid terminal artifact fails output equivalence.
- Keep the scenario local, deterministic, dependency-free, and safe for CI.
- Do not turn fixture results into public performance claims.

Run the bundled fixture with:

```bash
python scripts/workflow_benchmark.py \
  --fixture data/workflow_agent_fixture.json \
  --output workflow_benchmark_results.jsonl
```

## Output format

Write JSON lines:

```json
{"device":"Pixel class TBD","route":"gemma-3n-e2b-int4-litert","ttft_ms":0,"decode_tok_s":0,"peak_rss_mb":0,"prompt_tokens":64,"decode_tokens":64}
{"schema":"hermes.workflow_agent_benchmark.v1","scenario":"workflow-agent","scenario_id":"local-variant-calling-equivalence","backend":"cpu","affinity":"balanced","monitor_strategy":"event","status_checks":3,"idle_wait_ms":37,"completion_latency_ms":37,"timeout_count":0,"final_status":"succeeded","output_equivalent":true,"terminal_artifact_hash":"sha256-value"}
```
