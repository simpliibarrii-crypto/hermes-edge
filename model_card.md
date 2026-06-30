# Model Card: Hermes Mobile (LiteRT-LM)

## Overview

**Hermes Mobile** is a small, agentic, decoder-only language model built to run
fully on-device inside the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)
app via the **LiteRT-LM** runtime. It is distributed as a single `.litertlm`
bundle and supports **iPhone 16 (A18 Pro ANE)**, **Android (GPU/NPU)**, and
**iPad (M-series)**.

| Field | Value |
|---|---|
| Model name | `hermes-mobile-1b-litertlm` |
| Architecture | Decoder-only transformer, grouped-query attention |
| Parameters | ~270M / ~500M / ~1.0B (selectable preset) |
| Context window | 4096 tokens (8192 for Gemma presets) |
| Position embedding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |
| Tokenizer | SentencePiece BPE, 32k vocab |
| Quantization | INT4 per-channel (weight-only dynamic) |
| File format | `.litertlm` |
| Runtime | LiteRT-LM (CoreML delegate on iOS) |

## Intended use

- On-device chat assistant and **agent** for **iPhone 16** and **Android** via Google AI Edge Gallery
- Tool-calling / function-calling: emits structured `<tool_call>` JSON
- Privacy-sensitive scenarios where prompts must not leave the device
- Offline AI assistant in low-connectivity environments

### Out of scope

- High-stakes decisions (medical, legal, financial) without human review
- Long-document reasoning beyond the context window
- Tasks requiring broad, current world knowledge — pair with the web-search skill

## Targeted Devices

| Device | Chip | Backend | Speed (270m) | Speed (1b) |
|---|---|---|---|---|
| iPhone 16 | A18 | ANE (CoreML) | ~55 tok/s | ~25 tok/s |
| iPhone 16 Pro | A18 Pro | ANE (CoreML) | ~60 tok/s | ~28 tok/s |
| iPhone 15 Pro | A17 Pro | ANE (CoreML) | ~50 tok/s | ~22 tok/s |
| iPad Pro M4 | M4 | ANE (CoreML) | ~70 tok/s | ~35 tok/s |
| Galaxy S24 Ultra | SD 8 Gen 3 | GPU | ~65 tok/s | ~30 tok/s |
| Pixel 9 Pro | Tensor G4 | GPU | ~45 tok/s | ~20 tok/s |

## DeepSeek-Inspired Reasoning

Hermes uses chain-of-thought + tool-calling prompting inspired by DeepSeek-R1:

```
User: Calculate 15% of 340
Assistant (thinking):
  10% of 340 = 34
  5% of 340 = 17
  34 + 17 = 51
Tool: calculator(expression="340*0.15") -> 51
Assistant: 15% of 340 is 51.
```

## Architecture

Standard decoder-only transformer with grouped-query attention, RoPE, SwiGLU, RMSNorm:
- `hermes-270m`: 21 layers, 1024 hidden, 16 heads, 4 KV heads
- `hermes-500m`: 24 layers, 1536 hidden, 24 heads, 6 KV heads
- `hermes-1b`: 22 layers, 2048 hidden, 32 heads, 4 KV heads
- `gemma-3-1b`: 26 layers, 2048 hidden, 16 heads, 8 KV heads, 8192 ctx

## Installation

**iOS:** Open Google AI Edge Gallery → **+** → **Import from URL** → paste:
```
https://huggingface.co/bclermo/hermes-edge/resolve/main/hermes-mobile-270m-int4.litertlm
```

**Android:** Copy to `/sdcard/Download/` → Open Gallery → **+** → Select file.

## Training & Distillation

- Fine-tuned with `scripts/train.py` on agentic chat data
- Knowledge distillation from Gemma 3 1B via `scripts/distill_from_gemma.py`
- DeepSeek-R1 style chain-of-thought supervision for reasoning

## License

Apache 2.0
