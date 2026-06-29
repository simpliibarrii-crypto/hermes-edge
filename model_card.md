# Model Card: Hermes Mobile 1B (LiteRT-LM)

## Overview

**Hermes Mobile** is a small, agentic, decoder-only language model built to run
fully on-device inside the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery)
app via the **LiteRT-LM** runtime. It is distributed as a single `.litertlm`
bundle (TFLite graph + SentencePiece tokenizer + metadata) and is tuned for
tool-calling so it can act as an agent using the Gallery's Agent Skills system.

| Field | Value |
|---|---|
| Model name | `hermes-mobile-1b-litertlm` |
| Architecture | Decoder-only transformer, grouped-query attention |
| Parameters | ~1.0B |
| Layers / heads / KV-heads | 22 / 32 / 4 |
| Hidden / intermediate dim | 2048 / 5632 |
| Context window | 4096 tokens |
| Position embedding | RoPE (θ=10000) |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU (SiLU) |
| Tokenizer | SentencePiece BPE, 32k vocab |
| Quantization | INT4, 4-bit per-channel (weight-only dynamic) |
| File format | `.litertlm` |
| Approx. on-disk size | ~600 MB |
| Runtime | LiteRT-LM |

A smaller **`hermes-270m`** preset (~270M params, FunctionGemma-class) is also
provided for the lowest-end devices.

## Intended use

- On-device chat assistant and **agent** for mobile (Android) via Google AI Edge Gallery.
- Tool-calling / function-calling: emits structured `<tool_call>` JSON that the
  Gallery's Agent Skills runtime dispatches (e.g. offline calculator, web search).
- Privacy-sensitive scenarios where prompts must not leave the device.

### Out of scope

- High-stakes decisions (medical, legal, financial) without human review.
- Long-document reasoning beyond the 4096-token context window.
- Tasks requiring broad, current world knowledge — pair with the web-search skill instead.

## Tool-calling format

Prompts use a ChatML-style template with explicit tool sentinels:

```
<|im_start|>system
You are Hermes... You have access to the following tools: {...}<|im_end|>
<|im_start|>user
What is 12 * 9?<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "calculator", "arguments": {"expression": "12*9"}}</tool_call><|im_end|>
<|im_start|>tool
<tool_response>108</tool_response><|im_end|>
<|im_start|>assistant
12 * 9 = 108.<|im_end|>
```

Constrained decoding in LiteRT-LM can be anchored on `<tool_call>` / `</tool_call>`
to guarantee well-formed JSON.

## Performance targets

These are design targets for the INT4 build; measure on your device.

| Device class | Accelerator | Prefill | Decode (tokens/sec) |
|---|---|---|---|
| Flagship (8+ Gen 2/3) | GPU | fast | ~20–30 tok/s |
| Mid-range | GPU | moderate | ~10–18 tok/s |
| Any (fallback) | CPU | slow | ~4–8 tok/s |

Peak on-device memory target: **< 1.5 GB** with the INT4 build.

## LiteRT-LM compatibility

- Format: `.litertlm` (also loadable anywhere the LiteRT-LM runtime is embedded).
- Google AI Edge Gallery import: copy to `/sdcard/Download/` and add via the **+** button.
- Minimum Android 10; **≥ 6 GB RAM recommended**.

## Training & conversion

- Trained / fine-tuned with `scripts/train.py` on agentic chat data using the
  Hermes tool-calling template.
- Converted to `.litertlm` with `scripts/convert_to_litertlm.py`, which uses
  `ai_edge_torch.generative` to lower the PyTorch model to TFLite, applies INT4
  quantization, and bundles the tokenizer.

```bash
python scripts/convert_to_litertlm.py \
    --checkpoint checkpoints/hermes-1b.pt \
    --tokenizer tokenizer/hermes.model \
    --preset hermes-1b \
    --output dist/hermes-mobile-1b-int4.litertlm
```

## Limitations & biases

- A ~1B model has limited world knowledge and reasoning depth; it can
  hallucinate. Prefer tool calls for facts and math.
- INT4 quantization slightly degrades quality versus the float checkpoint.
- Inherits biases from its training data.

## License

Apache-2.0.
