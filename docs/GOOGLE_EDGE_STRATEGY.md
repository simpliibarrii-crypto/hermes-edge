# Hermes Edge Google-edge Strategy

Hermes Edge target: fastest reasonable local AI agent on Google-compatible edge runtimes, with every speed claim tied to a benchmark.

## What changed in the edge stack

Primary sources used:

- Google AI Edge: on-device stack across mobile, web, desktop, and embedded devices.
- LiteRT and LiteRT-LM: Google edge runtimes for custom models and LLMs.
- MediaPipe LLM Inference: cross-platform on-device LLM task API and bundling path.
- Gemma 3n: mobile-first Google model family with memory-efficient design.

Practical read: Hermes should not make one model do everything. It should route tasks through the cheapest local path that preserves quality.

## Runtime order

1. **Deterministic tools first**
   - math, time/date, unit conversion, cached facts, local retrieval
   - fastest path is no model call

2. **Gemini Nano / Android AICore when device exposes it**
   - optional system-model shortcut
   - no bundled model size
   - not required, because availability varies by Android device

3. **Gemma 4 E2B/E4B LiteRT-LM with MTP/speculative decoding when available**
   - candidate fastest Google-edge text route
   - must be feature-detected and benchmarked, not assumed

4. **Gemma 3n E2B INT4 LiteRT-LM baseline**
   - safest default local Google-edge route
   - good low/mid device target

5. **Gemma 3n E4B INT4 LiteRT-LM quality route**
   - use only when RAM and thermal budget allow

6. **Cloud fallback disabled by default**
   - project rule: no hidden paid/cloud trap
   - any network model must be explicit, optional, and visible to user

## Agent speed rules

- classify intent before model call
- compress system/tool prompts
- keep tool schemas tiny
- cache stable responses
- reuse sessions/KV cache where runtime supports it
- prefer retrieval snippets over long context
- route simple tasks to tools
- use speculative decoding only behind backend capability checks
- benchmark time-to-first-token, decode tokens/sec, peak RSS, and route success

## Current implementation

`hermes/edge_policy.py` encodes the routing policy as a dependency-free module. It does not claim hardware results. It picks a reasonable route based on:

- task class
- device tier
- available RAM
- Gemini Nano/AICore availability
- MTP/speculative decoding availability

Tests live in `tests/test_edge_policy.py`.

## Benchmark contract

Hermes Edge can claim speed only when backed by:

- device name
- OS/runtime version
- model profile
- quantization
- prompt/token counts
- TTFT
- decode tok/s
- peak memory
- temperature/top-k/top-p
- battery/thermal observation if available

Until then, wording should be: **designed for fast Google-edge routing**, not “fastest ever.”
