# Raven Token Economy Integration

Hermes Edge should provide the local lanes and benchmark evidence for Raven Token Economy. This is about AI inference token saving, not crypto tokenomics.

## Role in Hermes Edge

Hermes owns local routing, deterministic tools, edge model selection, and benchmark gates. Raven Token Economy owns the planning language for cache reuse, cheap drafting, confidence-scheduled verification, and late escalation.

| Hermes area | Token Economy use |
|---|---|
| Tool-first routing | Use deterministic tools before model calls whenever possible. |
| Local lanes | Provide `local-small` and `local-large` draft lanes for private or latency-sensitive work. |
| Benchmarks | Record saved context tokens, draft lane, backend, memory, TTFT, tokens/sec, and device metadata. |
| Evidence packets | Emit token-economy metadata beside `raven.evidence_graph.v1` traces. |

## Guardrails

- Token Economy means model/inference token saving, not blockchain, staking, wallets, or governance tokens.
- Do not publish speed, memory, or cost claims without benchmark evidence.
- Keep packets compact enough for mobile and offline logs.
- Treat local fallback as a safety lane, not a performance excuse.

## Adoption path

1. Add token-economy fields to edge route decision artifacts.
2. Convert benchmark artifacts into evidence sources and claims.
3. Emit compact trace packets for local model runs.
4. Add tests that public performance claims require benchmark evidence IDs.
