# Changelog

## v0.2.0 (2026-06-30)

### Features
- DeepSeek-style reasoning pipeline with `<think>`/`<answer>` separation
- Intent-based routing (chat, reasoning, tools) with multi-model support
- Web search tool via DuckDuckGo (no API key required)
- DSpark-inspired speculative decoding draft model framework
- LiteRT-LM runtime integration for on-device inference
- CLI interactive mode and HTTP API server
- Docker container with health checks
- HF Space Gradio demo

### Improvements
- CPU-optimized model conversion (2.7 GB RAM budget)
- Stage-wise memory pooling for INT4 quantization
- Qwen3-0.6B architecture support with weight remapping
- Streaming token generation with reasoned output

### Infrastructure
- CI with ruff linting, pytest, and multi-Python testing
- Multi-stage Dockerfile for slim production images
- Comprehensive test suite for model, inference, and agent

## v0.1.0 (2026-05-15)

- Initial release with basic LiteRT-LM model loading
- Hermes tool calling format support
- Simple agent loop
