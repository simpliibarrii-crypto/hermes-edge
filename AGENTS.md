# Hermes Edge — AI Assistant Guide

## Overview

Hermes Edge is an on-device AI agent for iPhone 16 and Android that runs fully offline via LiteRT-LM. It combines DeepSeek-style reasoning, Hermes tool calling, and DSpark speculative decoding.

## Key Files

| File | Purpose |
|------|---------|
| `hermes/agent.py` | Agent loop with tool orchestration and intent routing |
| `hermes/inference.py` | LiteRT-LM streaming inference engine |
| `hermes/litert_model.py` | LiteRT-LM model wrapper |
| `hermes/config.py` | Model architecture configuration |
| `hermes/chat_template.py` | ChatML + tool calling format |
| `hermes/router.py` | Intent classification (regex-based, ~5μs) |
| `hermes/web_search.py` | DuckDuckGo web search tool |
| `scripts/convert_hf_to_litertlm.py` | HF model to .litertlm converter |
| `scripts/deepseek_reasoning_template.py` | Reasoning prompt templates |
| `scripts/hermes_tool_format.py` | Tool format parser/generator |
| `scripts/dspark_draft.py` | Speculative decoding draft model |

## Architecture

The system uses a multi-model approach:
- **Hot model** (270M INT4, ~180 MB) always loaded for instant chat
- **Reasoning model** loaded on-demand for deep reasoning tasks
- **Tools model** loaded on-demand for tool-heavy queries

Intent routing is done via lightweight regex (~5μs) before model inference.

## Development Patterns

- Python 3.11+, type hints required
- Ruff for linting (line-length 100)
- Pytest for testing

## Model Conversion Flow

HuggingFace model → INT4 quantization → TFLite lowering → .litertlm bundle

See `ARCHITECTURE.md` for detailed pipeline documentation.
