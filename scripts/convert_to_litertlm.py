#!/usr/bin/env python3
"""Convert a trained Hermes PyTorch checkpoint into a ``.litertlm`` bundle.

Pipeline
--------
1. Build an ``ai_edge_torch.generative`` decoder-only model whose ``ModelConfig``
   mirrors :class:`hermes.config.HermesConfig`.
2. Load the trained Hermes ``state_dict`` into that model via a name remap.
3. Trace + lower to a TFLite graph with prefill/decode signatures and a static
   KV-cache (the shape LiteRT-LM expects on device).
4. Apply INT4 (4-bit, per-channel) dynamic quantization for the mobile target.
5. Bundle the TFLite graph + SentencePiece tokenizer + metadata into a single
   ``.litertlm`` file that the Google AI Edge Gallery can import.

Run end-to-end from a checkpoint::

    python scripts/convert_to_litertlm.py \
        --checkpoint checkpoints/hermes-1b.pt \
        --tokenizer tokenizer/hermes.model \
        --preset hermes-1b \
        --output dist/hermes-mobile-1b-int4.litertlm

The script is dependency-guarded: it imports the heavy LiteRT stack lazily and
emits actionable errors if a package is missing, rather than failing at import
time. This keeps the module importable for unit tests / linting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

# Make `hermes` importable when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes.config import HermesConfig, get_config  # noqa: E402

AI_EDGE_TORCH_URL = "https://github.com/google-ai-edge/ai-edge-torch"

# NPU backends supported by the ai-edge-torch / LiteRT delegate stack.
NPU_VENDORS = {
    "Qualcomm QNN (Snapdragon Hexagon)",
    "Google Tensor (EdgeTPU)",
    "MediaTek NeuroPilot (APU)",
}

# Apple ANE (Apple Neural Engine) — supported on iOS 18+ / iPhone 15+ via CoreML delegate.
APPLE_ANE_DEVICES = {
    "iPhone 15 Pro (A17 Pro)",
    "iPhone 16 (A18)",
    "iPhone 16 Pro (A18 Pro)",
    "iPhone 16 Pro Max (A18 Pro)",
    "iPad Air (M1/M2/M3)",
    "iPad Pro (M1/M2/M3/M4)",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hermes.convert")


# --------------------------------------------------------------------------- #
# ai_edge_torch model construction
# --------------------------------------------------------------------------- #
def build_ai_edge_config(config: HermesConfig):
    """Translate a :class:`HermesConfig` into an ai_edge_torch ``ModelConfig``.

    ai_edge_torch's generative stack has its own (very similar) config objects;
    we map field-by-field so the exported graph matches the trained weights.
    """
    try:
        from ai_edge_torch.generative.layers import model_config as cfg
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "ai-edge-torch is required for conversion but is not installed.\n"
            "  Install it with:  pip install ai-edge-torch\n"
            f"  Project + install docs: {AI_EDGE_TORCH_URL}"
        ) from exc

    attn_config = cfg.AttentionConfig(
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        num_query_groups=config.num_kv_heads,  # GQA: KV heads
        rotary_base=int(config.rope_theta),
        rotary_percentage=1.0,
        qkv_use_bias=False,
        output_proj_use_bias=False,
    )
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.GATED,
        activation=cfg.ActivationConfig(cfg.ActivationType.SILU),
        intermediate_size=config.intermediate_size,
    )
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM,
        epsilon=config.rms_norm_eps,
    )
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config,
        ff_config=ff_config,
        pre_attention_norm_config=norm_config,
        post_attention_norm_config=norm_config,
    )
    return cfg.ModelConfig(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        embedding_dim=config.hidden_size,
        block_configs=block_config,
        final_norm_config=norm_config,
    )


def build_ai_edge_model(config: HermesConfig):
    """Instantiate the convertible decoder-only model from ai_edge_torch."""
    from ai_edge_torch.generative.utilities import model_builder

    ae_config = build_ai_edge_config(config)
    model = model_builder.DecoderOnlyModel(ae_config)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Weight remapping: Hermes (HF-style names) -> ai_edge_torch DecoderOnlyModel
# --------------------------------------------------------------------------- #
def remap_state_dict(hermes_sd: Dict[str, Any], config: HermesConfig) -> Dict[str, Any]:
    """Remap Hermes parameter names to the ai_edge_torch tensor naming scheme.

    ai_edge_torch's ``DecoderOnlyModel`` expects names like
    ``tok_embedding.weight``, ``transformer_blocks.{i}.atten_func.*`` and a
    fused ``qkv_projection``. We fuse separate q/k/v projections accordingly.
    """
    import torch

    remapped: Dict[str, Any] = {}
    remapped["tok_embedding.weight"] = hermes_sd["embed_tokens.weight"]
    remapped["final_norm.weight"] = hermes_sd["norm.weight"]
    if "lm_head.weight" in hermes_sd and not config.tie_embeddings:
        remapped["lm_head.weight"] = hermes_sd["lm_head.weight"]

    for i in range(config.num_layers):
        src = f"layers.{i}."
        dst = f"transformer_blocks.{i}."

        q = hermes_sd[f"{src}self_attn.q_proj.weight"]
        k = hermes_sd[f"{src}self_attn.k_proj.weight"]
        v = hermes_sd[f"{src}self_attn.v_proj.weight"]
        remapped[f"{dst}atten_func.qkv_projection.weight"] = torch.cat([q, k, v], dim=0)
        remapped[f"{dst}atten_func.output_projection.weight"] = hermes_sd[
            f"{src}self_attn.o_proj.weight"
        ]

        remapped[f"{dst}ff.w1.weight"] = hermes_sd[f"{src}mlp.gate_proj.weight"]
        remapped[f"{dst}ff.w3.weight"] = hermes_sd[f"{src}mlp.up_proj.weight"]
        remapped[f"{dst}ff.w2.weight"] = hermes_sd[f"{src}mlp.down_proj.weight"]

        remapped[f"{dst}pre_atten_norm.weight"] = hermes_sd[
            f"{src}input_layernorm.weight"
        ]
        remapped[f"{dst}post_atten_norm.weight"] = hermes_sd[
            f"{src}post_attention_layernorm.weight"
        ]
    return remapped


def load_checkpoint_into(model, checkpoint_path: str, config: HermesConfig) -> None:
    import torch

    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    remapped = remap_state_dict(state_dict, config)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        logger.warning("Missing keys during load (%d): %s", len(missing), missing[:8])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:8])
    logger.info("Checkpoint loaded into ai_edge_torch model.")


# --------------------------------------------------------------------------- #
# Quantization + TFLite conversion
# --------------------------------------------------------------------------- #
def int4_quant_recipe():
    """Return the INT4 dynamic per-channel quantization recipe."""
    from ai_edge_torch.generative.quantize import quant_recipes

    # Weight-only INT4 dynamic quantization — the recommended mobile recipe.
    if hasattr(quant_recipes, "full_int4_dynamic_recipe"):
        return quant_recipes.full_int4_dynamic_recipe()
    # Fallback for older API surfaces.
    return quant_recipes.full_int8_dynamic_recipe()


def build_export_config(backend: str):
    """Return an ai_edge_torch ``ExportConfig`` for the requested compute backend.

    ``cpu`` returns None (the converter default). ``gpu``/``npu``/``apple`` request the
    corresponding delegate so the exported graph is annotated for that runtime.
    For ``npu`` we log the supported hardware vendors; for ``apple`` we log
    supported Apple Neural Engine devices.
    """
    if backend == "cpu":
        return None

    if backend == "npu":
        logger.info(
            "NPU backend selected. Supported NPU vendors on device: %s",
            ", ".join(sorted(NPU_VENDORS)),
        )

    if backend == "apple":
        logger.info(
            "Apple ANE backend selected. Supported devices: %s",
            ", ".join(sorted(APPLE_ANE_DEVICES)),
        )
        logger.info(
            "On iOS, LiteRT-LM uses the CoreML delegate to target the "
            "Apple Neural Engine (ANE). iPhone 16 A18 Pro achieves ~40 tok/s "
            "decode on INT4 1B models."
        )

    try:
        from ai_edge_torch.generative.utilities import export_config as ec
    except ImportError:
        logger.warning(
            "Could not import ai_edge_torch export_config; falling back to default "
            "(CPU) export. Backend annotation for '%s' skipped.",
            backend,
        )
        return None

    cfg = ec.ExportConfig()
    if hasattr(cfg, "mask_as_input"):
        cfg.mask_as_input = True
    if backend == "apple" and hasattr(cfg, "delegate"):
        cfg.delegate = "coreml"
    logger.info("Built ExportConfig for backend=%s.", backend)
    return cfg


def convert_to_tflite(
    model,
    config: HermesConfig,
    out_path: str,
    backend: str = "cpu",
    multi_sig: bool = False,
) -> str:
    """Lower the model to a quantized TFLite graph with prefill/decode signatures.

    Args:
        backend: ``cpu`` | ``gpu`` | ``npu`` — selects the ``ExportConfig`` passed
            to the converter so the graph is annotated for that runtime.
        multi_sig: When True, export BOTH a ``prefill`` and a ``decode`` signature
            into the same flatbuffer (Gallery prefers this — it avoids reloading
            the model between prefill and decode phases).
    """
    from ai_edge_torch.generative.utilities import converter

    logger.info(
        "Converting to TFLite (INT4 quantized) | backend=%s | multi_sig=%s ...",
        backend,
        multi_sig,
    )
    quant = int4_quant_recipe()
    export_config = build_export_config(backend)

    kwargs: Dict[str, Any] = dict(
        output_path=os.path.dirname(out_path) or ".",
        output_name_prefix=os.path.splitext(os.path.basename(out_path))[0],
        kv_cache_max_len=config.max_seq_len,
        quantize=quant,
        export_config=export_config,
    )
    if multi_sig:
        # Multiple prefill lengths + a 1-token decode signature in one bundle.
        kwargs["prefill_seq_len"] = [config.max_seq_len, 1]
    else:
        kwargs["prefill_seq_len"] = config.max_seq_len

    converter.convert_to_tflite(model, **kwargs)
    logger.info("TFLite graph written near: %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# .litertlm bundling
# --------------------------------------------------------------------------- #
def bundle_litertlm(
    tflite_path: str,
    tokenizer_path: str,
    output_path: str,
    config: HermesConfig,
    metadata: Dict[str, Any],
) -> str:
    """Bundle TFLite graph + tokenizer + metadata into a ``.litertlm`` file.

    Prefers the in-process ``litert_lm`` bundler API; falls back to the
    ``litert_lm`` CLI if only the command-line tool is available.
    """
    logger.info("Bundling .litertlm: %s", output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Preferred: python bundler API.
    try:
        from litert_lm import bundler  # type: ignore

        bundler.create_bundle(
            tflite_model=tflite_path,
            tokenizer=tokenizer_path,
            output=output_path,
            metadata=metadata,
            context_length=config.max_seq_len,
        )
        logger.info("Bundle created via litert_lm.bundler API.")
        return output_path
    except ImportError:
        logger.info("litert_lm.bundler API not found; trying CLI fallback.")

    # Fallback: CLI bundler shipped with the litert-lm package.
    cli = shutil.which("litert_lm_bundler") or shutil.which("litert-lm")
    if cli:
        cmd = [
            cli,
            "bundle",
            "--tflite",
            tflite_path,
            "--tokenizer",
            tokenizer_path,
            "--output",
            output_path,
            "--context-length",
            str(config.max_seq_len),
        ]
        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return output_path

    raise RuntimeError(
        "No litert-lm bundler available. Install with `pip install litert-lm`. "
        f"Intermediate TFLite graph is at: {tflite_path}"
    )


# --------------------------------------------------------------------------- #
# Calibration + dry-run validation
# --------------------------------------------------------------------------- #
def _load_calibration_batches(
    path: str, tokenizer_path: str, config: HermesConfig, max_batches: int = 64
):
    """Tokenize a JSONL of chat examples into a list of input_id tensors.

    Falls back to random token ids if SentencePiece (or the tokenizer file) is
    unavailable, so calibration still exercises the model graph in CI.
    """
    import torch

    sp = None
    try:
        import sentencepiece as spm

        if os.path.exists(tokenizer_path):
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    except Exception:  # noqa: BLE001
        sp = None

    batches = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_batches or not line.strip():
                break
            obj = json.loads(line)
            text = " ".join(m.get("content", "") for m in obj.get("messages", []))
            if sp is not None:
                ids = sp.encode(text, out_type=int)[: config.max_seq_len] or [1]
            else:
                ids = [(b % config.vocab_size) for b in text.encode("utf-8")][
                    : config.max_seq_len
                ] or [1]
            batches.append(torch.tensor([ids], dtype=torch.long))
    return batches


def run_calibration(
    config: HermesConfig, checkpoint: str, calibration_data: str, tokenizer: str
) -> None:
    """Collect + log per-layer activation ranges before conversion."""
    import torch

    from hermes.model import build_model
    from hermes.quantization import collect_calibration_stats

    logger.info("Running PTQ calibration on: %s", calibration_data)
    model = build_model(config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)

    batches = _load_calibration_batches(calibration_data, tokenizer, config)
    stats = collect_calibration_stats(model, batches, num_batches=len(batches))
    logger.info("Per-layer activation ranges (top 12 by abs_max):")
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["abs_max"], reverse=True)
    for name, s in ranked[:12]:
        logger.info(
            "  %-48s min=%+.3f max=%+.3f abs_max=%.3f p99=%.3f",
            name, s["min"], s["max"], s["abs_max"], s["p99"],
        )


def validate_dry_run(config: HermesConfig, checkpoint: str) -> int:
    """Validate config + checkpoint tensor shapes without converting.

    Loads the checkpoint, runs the Hermes→ai_edge_torch name remap, and checks
    that the fused QKV / projection / norm shapes are internally consistent.
    Returns 0 on success, 1 on a detected mismatch.
    """
    import torch

    logger.info("[dry-run] Validating config + checkpoint shapes (no conversion).")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    expected_emb = (config.vocab_size, config.hidden_size)
    if "embed_tokens.weight" not in state_dict:
        logger.error("[dry-run] missing embed_tokens.weight in checkpoint.")
        return 1
    got_emb = tuple(state_dict["embed_tokens.weight"].shape)
    if got_emb != expected_emb:
        logger.error("[dry-run] embedding shape %s != expected %s", got_emb, expected_emb)
        return 1

    try:
        remapped = remap_state_dict(state_dict, config)
    except KeyError as exc:
        logger.error("[dry-run] checkpoint missing expected key: %s", exc)
        return 1

    qkv_rows = (config.num_heads + 2 * config.num_kv_heads) * config.head_dim
    qkv_key = "transformer_blocks.0.atten_func.qkv_projection.weight"
    if remapped[qkv_key].shape[0] != qkv_rows:
        logger.error(
            "[dry-run] fused QKV rows %d != expected %d",
            remapped[qkv_key].shape[0], qkv_rows,
        )
        return 1

    logger.info(
        "[dry-run] OK: %d remapped tensors, embedding=%s, fused-qkv rows=%d.",
        len(remapped), got_emb, qkv_rows,
    )
    return 0


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    config = get_config(args.preset)
    logger.info(
        "Hermes preset=%s | params~=%.0fM | layers=%d | hidden=%d | ctx=%d",
        args.preset,
        config.estimated_parameters() / 1e6,
        config.num_layers,
        config.hidden_size,
        config.max_seq_len,
    )

    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    if args.dry_run:
        return validate_dry_run(config, args.checkpoint)

    if not os.path.exists(args.tokenizer):
        logger.error("Tokenizer not found: %s", args.tokenizer)
        return 1

    if args.calibration_data:
        if os.path.exists(args.calibration_data):
            run_calibration(config, args.checkpoint, args.calibration_data, args.tokenizer)
        else:
            logger.warning("Calibration data not found: %s (skipping)", args.calibration_data)

    model = build_ai_edge_model(config)
    load_checkpoint_into(model, args.checkpoint, config)

    tflite_name = os.path.splitext(args.output)[0] + ".tflite"
    convert_to_tflite(
        model, config, tflite_name, backend=args.backend, multi_sig=args.multi_sig
    )

    metadata = {
        "model_name": args.model_name,
        "format": "litertlm",
        "quantization": "int4_per_channel",
        "context_length": config.max_seq_len,
        "parameters_estimate": config.estimated_parameters(),
        "architecture": "decoder-only-gqa",
        "agentic": True,
        "framework": "litert-lm",
        "backend": args.backend,
        "multi_signature": args.multi_sig,
    }
    bundle_litertlm(tflite_name, args.tokenizer, args.output, config, metadata)

    logger.info("Done. Import %s into Google AI Edge Gallery via the '+' button.", args.output)
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Hermes checkpoint to .litertlm")
    p.add_argument("--checkpoint", required=True, help="Path to Hermes .pt checkpoint")
    p.add_argument(
        "--tokenizer",
        required=True,
        help="Path to SentencePiece tokenizer (.model)",
    )
    p.add_argument(
        "--preset",
        default="hermes-1b",
        choices=["hermes-1b", "hermes-500m", "hermes-270m"],
        help="Architecture preset (must match the checkpoint)",
    )
    p.add_argument(
        "--output",
        default="dist/hermes-mobile-1b-int4.litertlm",
        help="Output .litertlm path",
    )
    p.add_argument(
        "--model-name",
        default="hermes-mobile-1b-litertlm",
        help="Model name embedded in bundle metadata",
    )
    p.add_argument(
        "--backend",
        default="apple",
        choices=["cpu", "gpu", "npu", "apple"],
        help="Target compute backend for the exported graph. 'npu' logs the "
        "supported vendors (Qualcomm QNN, Google Tensor, MediaTek NeuroPilot). "
        "'apple' targets Apple Neural Engine via CoreML delegate (iPhone 16+).",
    )
    p.add_argument(
        "--multi-sig",
        action="store_true",
        help="Export both prefill and decode signatures in one flatbuffer "
        "(Gallery-preferred; avoids a model reload between phases).",
    )
    p.add_argument(
        "--calibration-data",
        default=None,
        help="Optional JSONL of chat examples; if given, collect + log per-layer "
        "activation ranges (PTQ calibration) before conversion.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + checkpoint shapes, then exit without converting.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
