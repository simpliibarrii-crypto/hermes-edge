"""
Hermes Edge — HuggingFace to LiteRT-LM (.litertlm) Converter

Converts any HuggingFace causal LM (Qwen, DeepSeek-Distill, Gemma, Llama)
into a self-contained .litertlm bundle for on-device inference.

Usage:
    python scripts/convert_hf_to_litertlm.py \\
        --model_id Qwen/Qwen2.5-0.5B-Instruct \\
        --output_dir ./dist \\
        --quantization dynamic_wi4_afp32 \\
        --cache_length 2048 \\
        --prefill_lengths 32

Requirements:
    pip install litert-torch torch transformers sentencepiece
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def convert_model(
    model_id: str,
    output_dir: str,
    quantization: str = "dynamic_wi4_afp32",
    cache_length: int = 2048,
    prefill_lengths: list[int] | None = None,
    externalize_embedder: bool = True,
    force: bool = False,
    chat_template: str | None = None,
) -> Path:
    """
    Convert a HuggingFace causal LM to a .litertlm bundle using the real LiteRT-Torch API.

    This uses `litert_torch.generative.export_hf.export()` — the official
    supported path from Google AI Edge. The export function:
      1. Loads the HF model and tokenizer
      2. Traces the forward graph with sample inputs
      3. Applies quantization (INT4 weights, FP16 activations by default)
      4. Lowers through TFLite converter (requires tf-nightly)
      5. Bundles .tflite + tokenizer + metadata into .litertlm

    The .litertlm format is compatible with:
      - Google AI Edge Gallery (iOS / Android)
      - LiteRT-LM Swift SDK (iOS 18+)
      - LiteRT-LM Kotlin SDK (Android 14+)
    """
    prefill_lengths = prefill_lengths or [32]

    out = Path(output_dir)
    if out.exists():
        if force:
            shutil.rmtree(out)
        else:
            raise FileExistsError(f"Output directory {out} already exists (use --force)")
    out.mkdir(parents=True)

    try:
        from litert_torch.generative.export_hf import export
    except ImportError:
        log.error(
            "litert_torch not installed. Install with:\n"
            "  pip install litert-torch torch transformers sentencepiece"
        )
        sys.exit(1)

    export_kwargs = dict(
        model=model_id,
        output_dir=str(out),
        task="text_generation",
        prefill_lengths=prefill_lengths,
        cache_length=cache_length,
        quantization_recipe=quantization,
        externalize_embedder=externalize_embedder,
        single_token_embedder=True,
    )
    if chat_template:
        export_kwargs["jinja_chat_template_override"] = chat_template

    log.info("Starting conversion: model=%s quant=%s cache=%d", model_id, quantization, cache_length)
    log.info("This may take 10-30 minutes and requires 4-8GB RAM.")
    log.info("Output: %s", out.resolve())

    try:
        export.export(**export_kwargs)
    except Exception as exc:
        log.error("Conversion failed: %s", exc)
        log.error(
            "Troubleshooting:\n"
            "  1. Ensure tf-nightly is installed: pip install tf-nightly\n"
            "  2. Reduce prefill_lengths (e.g. --prefill_lengths 16)\n"
            "  3. Increase swap: fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile\n"
            "  4. Use a smaller model (Qwen2.5-0.5B requires ~4GB peak)"
        )
        raise

    lif_files = list(out.glob("*.litertlm"))
    if not lif_files:
        log.error("No .litertlm file produced. Check logs above.")
        log.info("Expected file in: %s", out)
        for p in out.rglob("*"):
            if p.is_file():
                log.info("  %s (%d MB)", p.name, p.stat().st_size // 1024 // 1024)
        raise FileNotFoundError("No .litertlm output")

    result = lif_files[0]
    mb = result.stat().st_size / 1024 / 1024
    log.info("SUCCESS: %s (%.1f MB)", result.name, mb)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF model to LiteRT-LM (.litertlm)")
    parser.add_argument(
        "--model_id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-0.5B-Instruct)",
    )
    parser.add_argument("--output_dir", default="./dist", help="Output directory")
    parser.add_argument(
        "--quantization",
        default="dynamic_wi4_afp32",
        choices=["dynamic_wi4_afp32", "dynamic_wi8_afp32", "fp16"],
        help="Quantization recipe (default: dynamic_wi4_afp32)",
    )
    parser.add_argument("--cache_length", type=int, default=2048, help="KV cache length")
    parser.add_argument(
        "--prefill_lengths",
        type=int,
        nargs="+",
        default=[32],
        help="Prefill lengths for tracing (default: 32)",
    )
    parser.add_argument(
        "--externalize_embedder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Externalize embedding table (reduces peak RAM)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory")
    parser.add_argument("--chat_template", help="Optional Jinja2 chat template override")

    args = parser.parse_args()

    result = convert_model(
        model_id=args.model_id,
        output_dir=args.output_dir,
        quantization=args.quantization,
        cache_length=args.cache_length,
        prefill_lengths=args.prefill_lengths,
        externalize_embedder=args.externalize_embedder,
        force=args.force,
        chat_template=args.chat_template,
    )
    print(f"\nModel ready: {result}")
    print(f"Run with: litert-lm run {result} --prompt 'Hello'")


if __name__ == "__main__":
    main()
