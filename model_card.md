---
language:
- en
license: apache-2.0
tags:
- litert-lm
- google-ai-edge
- gemma
- gemini-nano
- gpu-first
- mobile-ai
- on-device
- local-first
pipeline_tag: text-generation
library_name: custom
base_model: google/gemma-3n-E2B-it
---

# Hermes Edge LiteRT-LM Model Card

Hermes Edge is a GPU-first, local-first edge AI agent runtime designed for LiteRT-LM and Google AI Edge style deployments. It is optimized for phones, tablets, laptops, and small edge boxes.

## Intended use

- Local AI assistant and tool-calling agent
- Android and iOS edge inference experiments
- Gemma 3n / LiteRT-LM deployment testing
- Benchmark-gated speculative decoding and MTP experiments
- Offline demos where cloud inference is not acceptable

## Runtime policy

Hermes uses a dependency-free route policy:

1. deterministic tools before model calls
2. Gemini Nano/AICore only when available and explicitly preferred
3. Gemma 3n E2B/E4B INT4 LiteRT-LM as default local model path
4. Gemma 4 MTP/speculative only when backend support and benchmarks prove benefit
5. cloud fallback disabled by default

## GPU-primary backend behavior

When `backend="auto"`, Hermes tries GPU-class delegates first:

1. `gpu`
2. `vulkan`
3. `metal`
4. `ane`
5. `cpu`

If GPU delegate fails at runtime, Hermes falls back to next local backend. CPU is fallback, not primary.

## Install

```bash
git clone https://github.com/simpliibarrii-crypto/hermes-edge.git
cd hermes-edge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_edge_policy.py tests/test_agent_edge_routing.py tests/test_litert_backend.py -q
```

For LiteRT-LM inference:

```bash
pip install -e ".[runtime]"
hermes --model dist/hermes-mobile-270m-int4.litertlm --backend auto
```

For conversion:

```bash
pip install -e ".[model,conversion,runtime]"
```

## Benchmarks

No performance claim is valid without exact device, model profile, backend used, TTFT, decode tokens/sec, prefill tokens/sec, peak memory, and thermal state.

See `docs/BENCHMARK_CONTRACT.md`.

## Limitations

- `.litertlm` bundles must be built or downloaded separately.
- Hardware delegate support varies by device and runtime version.
- Speculative/MTP acceleration is gated by backend support and measured improvement.
- Hosted demos are optional and not required for local use.

## License

Apache-2.0.
