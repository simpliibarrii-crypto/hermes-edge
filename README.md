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

| | `hermes-270m` | `hermes-500m` | `hermes-1b` (default) |
|---|---|---|---|
| Parameters | ~270M | ~500M | ~1.0B |
| Layers | 21 | 24 | 22 |
| Hidden dim | 1024 | 1536 | 2048 |
| Heads / KV-heads (GQA) | 16 / 4 | 24 / 6 | 32 / 4 |
| Intermediate dim | 2816 | 4096 | 5632 |
| Context window | 4096 | 4096 | 4096 |
| Tokenizer | SentencePiece BPE, 32k | same | same |
| Quantization | INT4 per-channel | INT4 per-channel | INT4 per-channel |
| Approx. INT4 file size | ~180 MB | ~280 MB | ~600 MB |
| Speed target (Snapdragon 8 Gen 3 GPU) | ~2500 tok/s | ~1500 tok/s | ~850 tok/s |

> Speed targets are decode-throughput goals at INT4 on the Snapdragon 8 Gen 3
> GPU, benchmarked relative to Google's published **Gemma 3 1B** baseline of
> **2585 tok/s** on the same SoC. The `hermes-500m` preset is the
> quality/speed sweet spot for mid-range devices.

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
  inference.py               Streaming inference engine (sampling + agentic loop)
  kv_cache.py                Static / sliding-window / paged KV caches
  quantization.py            PTQ calibration + INT4/INT8 fake-quant utilities
scripts/
  train.py                   Supervised fine-tuning on agentic chat data
  train_tokenizer.py         Train the bundled SentencePiece tokenizer
  convert_to_litertlm.py     PyTorch -> TFLite -> INT4 -> .litertlm
  benchmark.py               Mobile speed/memory profiler (pre-conversion)
  eval.py                    Perplexity + tool-call accuracy harness
skills/
  hermes_calculator/SKILL.md Offline calculator Agent Skill (JavaScript)
  hermes_web_search/SKILL.md Web search Agent Skill (JavaScript)
  hermes_memory/SKILL.md     Offline key/value memory Agent Skill (JavaScript)
  hermes_timer/SKILL.md      Offline timer/stopwatch Agent Skill (JavaScript)
data/
  eval.jsonl                 Tiny perplexity eval set (10 chat examples)
  tool_eval.jsonl            Tiny tool-call eval set (10 examples)
tests/                       Smoke tests (model, inference, kv_cache, quantization)
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

Useful conversion flags:

| Flag | Purpose |
|---|---|
| `--backend {cpu,gpu,npu}` | Target compute backend. `npu` logs supported vendors (Qualcomm QNN, Google Tensor, MediaTek NeuroPilot). |
| `--multi-sig` | Export both `prefill` and `decode` signatures in one flatbuffer (Gallery-preferred; avoids a reload). |
| `--calibration-data <jsonl>` | Run PTQ calibration and log per-layer activation ranges before conversion. |
| `--dry-run` | Validate config + checkpoint shapes, then exit (CI-friendly, no heavy deps). |

## Running inference locally

The `HermesInference` engine wraps a checkpoint + tokenizer with streaming
sampling (top-k, nucleus/top-p, repetition penalty), a `chat()` helper, and an
agentic `tool_call_loop()`:

```bash
python -c "
from hermes.config import hermes_270m_config
from hermes.inference import HermesInference
import sentencepiece as spm
tok = spm.SentencePieceProcessor(model_file='tokenizer/hermes.model')
engine = HermesInference.from_checkpoint(hermes_270m_config(), 'checkpoints/hermes-270m.pt', tok, preset_name='hermes-270m')
print(engine)
for delta in engine.generate('Hello!', max_new_tokens=64, stream=True):
    print(delta, end='', flush=True)
"
```

## Benchmarking

Profile prefill/decode throughput, time-to-first-token, and peak memory at the
batch-size-1 mobile target *before* converting:

```bash
python scripts/benchmark.py --preset hermes-270m \
    --seq-lens 64 128 256 512 1024 --runs 5 --device cpu
```

Prints a markdown table and writes `benchmark_results.json`. Add `--device cuda`
to also capture peak VRAM.

## Evaluating

Run the lightweight eval harness (perplexity + tool-call accuracy). It works
with randomly initialized weights (no `--checkpoint`) for CI smoke testing:

```bash
python scripts/eval.py --preset hermes-270m \
    --checkpoint checkpoints/hermes-270m.pt --tokenizer tokenizer/hermes.model
```

Prints `perplexity`, `tool_call_accuracy`, and `avg_latency_ms`, and writes
`eval_results.json`. Eval data lives in `data/eval.jsonl` and `data/tool_eval.jsonl`.

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

| Skill | Status | What it does |
|---|---|---|
| **Hermes Calculator** | Offline | Safe arithmetic/math evaluation (no `eval`) — keeps the small model accurate on numbers. |
| **Hermes Web Search** | Online | Fetches up-to-date results (DuckDuckGo Instant Answer) for questions beyond the model's knowledge. |
| **Hermes Memory** | Offline | Key/value note-taking persisted in `localStorage` — `remember` / `recall` / `forget` / `list_memories`. |
| **Hermes Timer** | Offline | Countdown timers + stopwatches via `setTimeout`/`Date.now()` — `set_timer` / `start_stopwatch` / `stop_stopwatch` / `list_timers`. |

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
