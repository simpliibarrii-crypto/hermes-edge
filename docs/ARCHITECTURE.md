# Architecture

This document provides a brief overview. For the full detailed architecture, see [`ARCHITECTURE.md`](../ARCHITECTURE.md) in the project root.

## Components

- **Model Pipeline**: HuggingFace → INT4 quantization → TFLite → `.litertlm` bundle
- **Inference Engine**: LiteRT-LM runtime with streaming, reasoning, and speculative decoding
- **Agent Framework**: Hermes-style tool calling with parallel dispatch and memory
- **Intent Router**: Lightweight regex-based classification (~5μs) for chat/reasoning/tools
- **DSpark**: Speculative decoding with a 30M-parameter draft model for ~2.5× speedup

## Key Modules

| Module | Location | Role |
|--------|----------|------|
| `hermes/litert_model.py` | LiteRT-LM wrapper | Model lifecycle |
| `hermes/inference.py` | Inference engine | Token generation |
| `hermes/agent.py` | Agent loop | Tool orchestration |
| `hermes/router.py` | Intent routing | Query classification |
| `hermes/config.py` | Architecture config | Model presets |
| `scripts/convert_hf_to_litertlm.py` | Conversion | HF → .litertlm |
