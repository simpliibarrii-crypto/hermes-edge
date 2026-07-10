<p align="center">\n  <strong><a href="https://barry-ai-public.simpliibarrii.chatgpt.site">Explore the complete AI research & projects portfolio →</a></strong>\n</p>\n\n---
language:
- en
license: apache-2.0
title: Hermes Edge
emoji: 🦊
colorFrom: indigo
colorTo: purple
tags:
- hermes-edge
- mobile-ai
- on-device
- android
- ios
- gpu-first
- litert-lm
- gemma
- gemini-nano
- tool-calling
- local-first
library_name: custom
pipeline_tag: text-generation
short_description: GPU-first local edge AI agent with LiteRT-LM, Gemma/Gemini Nano routing, and benchmark-gated acceleration.
base_model: google/gemma-3n-E2B-it
---

# 🦊 Hermes Edge

**Hermes Edge is a GPU-first, local-first AI agent runtime for phones, tablets, laptops, and edge boxes.**
It routes work to deterministic tools first, then local Google-edge models through LiteRT-LM, Gemma 3n, Gemma 4 MTP when proven, or Gemini Nano/AICore when Android exposes it.

No cloud account is required for the core agent. Heavy model, conversion, Space, and audit packages are explicit optional extras.

<p align="center">
  <img src="assets/hermes-logo.svg" alt="Hermes Edge Logo" width="180" height="180" />
</p>

<p align="center">
  <a href="https://github.com/simpliibarrii-crypto/hermes-edge"><img src="https://img.shields.io/badge/GitHub-Hermes%20Edge-black?style=flat-square" alt="GitHub"></a>
  <a href="https://huggingface.co/bclermo/hermes-edge"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-FFD21E?style=flat-square" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License"></a>
</p>

## Fast install

### Core agent and tests, any desktop

```bash
git clone https://github.com/simpliibarrii-crypto/hermes-edge.git
cd hermes-edge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/test_edge_policy.py tests/test_agent_edge_routing.py tests/test_litert_backend.py -q
```

Windows PowerShell:

```powershell
git clone https://github.com/simpliibarrii-crypto/hermes-edge.git
cd hermes-edge
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest tests/test_edge_policy.py tests/test_agent_edge_routing.py tests/test_litert_backend.py -q
```

### Local model inference with LiteRT-LM

```bash
pip install -e ".[runtime]"
hermes --model dist/hermes-mobile-270m-int4.litertlm --backend auto
```

`--backend auto` is GPU-primary. Hermes tries `gpu`, then `vulkan`, `metal`, `ane`, and finally `cpu` if needed.

### Conversion/build tools

```bash
pip install -e ".[model,conversion,runtime]"
```

### Hugging Face Space demo

```bash
pip install -e ".[space,model]"
python space_app.py
```

### Everything, only if you need the full stack

```bash
pip install -e ".[all]"
```

## Device install guide

| Device | Recommended path | Backend priority |
|---|---|---|
| Android phone/tablet | LiteRT-LM or Google AI Edge Gallery import | GPU, Vulkan, AICore/Gemini Nano when available, CPU fallback |
| iPhone/iPad | Google AI Edge Gallery import when LiteRT-LM bundle is available | GPU/Metal/ANE, CPU fallback |
| macOS | Python package + LiteRT-LM runtime | GPU/Metal first, CPU fallback |
| Linux laptop/edge box | Python package + LiteRT-LM runtime | GPU/Vulkan first, CPU fallback |
| Windows | Python package for routing/dev, LiteRT runtime where available | GPU first, CPU fallback |

## What changed in v0.3

- GPU-primary LiteRT backend selection in `hermes/litert_model.py`
- Google-edge policy in `hermes/edge_policy.py`
- Runtime model selection wired through `ModelManager.resolve_edge()`
- Tool-first routing before expensive model calls
- Lightweight default install, with heavy dependencies moved to explicit extras
- Benchmark contract for public performance claims

## Architecture

Hermes Edge keeps the big model as the last resort:

1. **Deterministic tools first** - math, formatting, retrieval, local utilities.
2. **Tiny routing layer** - classifies tool/chat/reasoning work without model load.
3. **Google-edge local policy** - chooses Gemma 3n E2B/E4B, Gemini Nano/AICore, or benchmark-gated Gemma 4 MTP path.
4. **GPU-primary LiteRT execution** - attempts GPU-class delegates first, with CPU fallback.
5. **Benchmark gates** - no speed claims without TTFT, tok/s, memory, backend, and device proof.

## Google-edge routing policy

`hermes/edge_policy.py` is dependency-free and CI-friendly. It selects:

- `tool-first` for tool tasks
- `gemini-nano-aicore` only when system model is preferred and device support exists
- `gemma-4-e2b-mtp-litert` only when MTP/speculative backend support is available
- `gemma-3n-e4b-int4-litert` when memory allows
- `gemma-3n-e2b-int4-litert` as practical baseline
- `cloud-fallback-disabled` when no local route fits

## Benchmark contract

Before any public speed claim, run and publish:

| Metric | Required |
|---|---|
| TTFT | yes |
| decode tokens/sec | yes |
| prefill tokens/sec | yes |
| peak memory | yes |
| exact device | yes |
| backend used | yes |
| model profile | yes |
| thermal state | yes |

See `docs/BENCHMARK_CONTRACT.md`.

## Build a LiteRT-LM model

```bash
pip install -e ".[model,conversion,runtime]"

litert-torch export_hf   --model=google/gemma-3n-E2B-it   --output_dir=./dist   --quantization=dynamic_wi4_afp32   --cache_length=2048   --prefill_lengths=32
```

## Python usage

```python
from hermes.agent import ModelManager
from hermes.edge_policy import DeviceTier

manager = ModelManager(
    backend="auto",
    device_tier=DeviceTier.HIGH,
    available_ram_mb=4096,
    mtp_available=True,
)

manager.register("gemma-3n-e4b-int4-litert", "dist/gemma-3n-e4b-int4.litertlm")
model = manager.resolve_edge("chat")
print(manager.last_route_decision.profile.id)
```

## Development checks

```bash
pip install -e ".[dev]"
python -m compileall -q hermes tests scripts
pytest tests -q
ruff check hermes/edge_policy.py hermes/agent.py hermes/litert_model.py tests/test_edge_policy.py tests/test_agent_edge_routing.py tests/test_litert_backend.py
```

## Project stance

Hermes Edge is open-source and local-first. Paid APIs, cloud inference, app-store distribution, hosted Spaces, and proprietary services are optional distribution or demo paths, not required core operation.

## License

Apache-2.0. See `LICENSE`.
