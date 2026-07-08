# Raven Evidence Graph Integration

Hermes Edge should use Raven Evidence Graph as the compact provenance packet for edge runs, benchmark claims, and offline agent outputs. The graph lives in Raven AI and serializes as `raven.evidence_graph.v1` JSON so Hermes can emit traceable results without adding a heavy runtime dependency.

## Role in Hermes Edge

Hermes owns local model routing, edge policy, benchmark gates, and device-aware execution. Raven Evidence Graph owns source, claim, confidence, risk, and answer trace structure.

| Hermes area | Evidence Graph use |
|---|---|
| Benchmark reports | Attach evidence traces to public performance claims such as TTFT, tokens/sec, backend, and memory. |
| Edge agent runs | Emit compact trace packets for tool-first answers and local model outputs. |
| Offline mode | Store trace JSON alongside local artifacts so results remain reviewable without cloud access. |
| Release claims | Link README or model-card claims to benchmark evidence IDs before publishing speed statements. |

## Compact packet shape

```json
{
  "device": "android-high-tier-demo",
  "backend": "gpu",
  "evidence_trace": {
    "schema": "raven.evidence_graph.v1",
    "question": "What route did Hermes choose?",
    "answer": "Hermes selected the GPU-first local route.",
    "claim_ids": ["claim:example"],
    "source_ids": ["source:example"],
    "confidence": 0.86,
    "risk": "low",
    "explanation": "The trace links route policy, device metadata, and benchmark fields."
  }
}
```

## Guardrails

- Do not publish speed or memory claims without benchmark evidence.
- Keep packets small enough for mobile and offline logs.
- Reference source artifacts by ID or path rather than embedding large benchmark files.
- Treat confidence as evidence quality, not model correctness.

## Adoption path

1. Keep this document as the contract while Raven Evidence Graph lands upstream.
2. Add a benchmark-to-evidence adapter for `docs/BENCHMARK_CONTRACT.md` outputs.
3. Emit trace packets from edge route decisions and local model runs.
4. Add tests that public benchmark claims require traceable source fields.
