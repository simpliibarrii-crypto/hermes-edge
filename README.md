# hermes-edge

**Hermes** — a mobile-first, agentic AI model that runs fully on-device inside
the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app via
the **LiteRT-LM** runtime.

This repo contains the full pipeline to build a `.litertlm` model bundle:
the PyTorch model definition, a training/fine-tuning script, the
PyTorch → LiteRT-LM conversion + INT4 quantization pipeline, and a set of
**Agent Skills** that give Hermes tool-calling abilities (offline calculator,
web search).

> Google AI Edge Gallery only loads `.litertlm` or `.task` models — **not** GGUF,
> ONNX, or raw `.tflite`. This project targets `.litertlm` exclusively via the
> LiteRT stack (`ai-edge-torch` + `litert-lm`).

## Architecture

A compact decoder-only transformer in the Gemma / Llama family, sized for
on-device inference.

| | `hermes-1b` (default) | `hermes-270m` |
|---|---|---|
| Parameters | ~1.0B | ~270M |
| Layers | 22 | 21 |
| Hidden dim | 2048 | 1024 |
| Heads / KV-heads (GQA) | 32 / 4 | 16 / 4 |
| Intermediate dim | 5632 | 2816 |
| Context window | 4096 | 4096 |
| Tokenizer | SentencePiece BPE, 32k | same |
| Quantization | INT4 per-channel | INT4 per-channel |
| Approx. file size | ~600 MB | ~180 MB |

Common building blocks: **grouped-query attention** (KV-cache efficiency),
**RoPE** position embeddings, **RMSNorm** pre-norm, **SwiGLU** MLP. The model is
**agentic**: it emits structured `<tool_call>` JSON dispatched by the Gallery's
Agent Skills runtime.

## Repository layout

```
hermes/                      Python package
  config.py                  Architecture presets (single source of truth)
  model.py                   Reference PyTorch model (training + tracing)
  chat_template.py           ChatML + tool-calling prompt format
scripts/
  train.py                   Supervised fine-tuning on agentic chat data
  train_tokenizer.py         Train the bundled SentencePiece tokenizer
  convert_to_litertlm.py     PyTorch -> TFLite -> INT4 -> .litertlm
skills/
  hermes_calculator/SKILL.md Offline calculator Agent Skill (JavaScript)
  hermes_web_search/SKILL.md Web search Agent Skill (JavaScript)
tests/test_model.py          Smoke tests
model_card.md                Model card
hf_model_config.json         HuggingFace (litert-community) publishing metadata
```

## Install

```bash
pip install -r requirements.txt
```

Key deps: `ai-edge-torch`, `litert-lm`, `torch`, `sentencepiece`.

## End-to-end pipeline

### 1. Train a tokenizer (once)

```bash
python scripts/train_tokenizer.py --input corpus.txt --vocab-size 32000 \
    --output tokenizer/hermes.model
```

### 2. Train / fine-tune

```bash
python scripts/train.py \
    --preset hermes-1b \
    --data data/agentic_sft.jsonl \
    --tokenizer tokenizer/hermes.model \
    --output checkpoints/hermes-1b.pt \
    --epochs 1 --batch-size 4 --lr 2e-4
```

Dataset is JSONL, one conversation per line with a `messages` array (and an
optional `tools` array) — see the docstring in `scripts/train.py`.

### 3. Convert to `.litertlm`

```bash
python scripts/convert_to_litertlm.py \
    --checkpoint checkpoints/hermes-1b.pt \
    --tokenizer tokenizer/hermes.model \
    --preset hermes-1b \
    --output dist/hermes-mobile-1b-int4.litertlm
```

This builds an `ai_edge_torch.generative` model, loads the trained weights,
lowers to a TFLite graph with prefill/decode signatures + static KV-cache,
applies **INT4** quantization, and bundles the tokenizer + metadata into the
`.litertlm`.

## Load in Google AI Edge Gallery

1. Push the model to the device:
   ```bash
   adb push dist/hermes-mobile-1b-int4.litertlm /sdcard/Download/
   ```
2. Open the **Google AI Edge Gallery** app → **+** (import) button →
   select the `.litertlm` from Downloads.
3. Start chatting. Hermes runs on the **GPU** if available, else CPU.

## Load the Agent Skills

Hermes is agentic via the Gallery's Agent Skills system. Each skill is a
`SKILL.md` file (JavaScript or native).

- **By URL:** in the Gallery's Agent Skills screen, paste the raw GitHub URL to
  `skills/hermes_calculator/SKILL.md` or `skills/hermes_web_search/SKILL.md`.
- **By file:** copy the `SKILL.md` to `/sdcard/Download/` and import it locally.

Included skills:

- **Hermes Calculator** — fully offline arithmetic/math evaluation (no `eval`,
  safe parser). Keeps the small model accurate on numbers.
- **Hermes Web Search** — fetches up-to-date results (DuckDuckGo Instant Answer)
  inside the Gallery WebView for questions beyond the model's knowledge.

## Requirements (device)

- **Android 10+**
- **≥ 6 GB RAM recommended** (peak on-device memory target < 1.5 GB at INT4)
- GPU acceleration recommended; CPU fallback supported

## Testing

```bash
pip install pytest torch
pytest tests/
```

The tests are intentionally light (no LiteRT stack required) and validate the
model forward/loss/generation paths, config invariants, and the tool-call
prompt round-trip.

## Publishing

`hf_model_config.json` holds metadata for publishing the `.litertlm` to the
`litert-community` HuggingFace org. See `model_card.md` for the full card.

## License

Apache-2.0.
