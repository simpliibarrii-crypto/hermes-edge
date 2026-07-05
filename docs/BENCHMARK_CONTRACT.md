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

## Output format

Write JSON lines:

```json
{"device":"Pixel class TBD","route":"gemma-3n-e2b-int4-litert","ttft_ms":0,"decode_tok_s":0,"peak_rss_mb":0,"prompt_tokens":64,"decode_tokens":64}
```
