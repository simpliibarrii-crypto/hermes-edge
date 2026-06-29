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
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict

# Make `hermes` importable when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes.config import HermesConfig, get_config  # noqa: E402

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
            "ai-edge-torch is required for conversion. Install with "
            "`pip install ai-edge-torch`."
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


def convert_to_tflite(model, config: HermesConfig, out_path: str) -> str:
    """Lower the model to a quantized TFLite graph with prefill/decode signatures."""
    from ai_edge_torch.generative.utilities import converter

    logger.info("Converting to TFLite (INT4 quantized)...")
    quant = int4_quant_recipe()
    # The generative converter authors both a `prefill` and `decode` signature
    # backed by a static KV-cache sized to max_seq_len.
    converter.convert_to_tflite(
        model,
        output_path=os.path.dirname(out_path) or ".",
        output_name_prefix=os.path.splitext(os.path.basename(out_path))[0],
        prefill_seq_len=config.max_seq_len,
        kv_cache_max_len=config.max_seq_len,
        quantize=quant,
        export_config=None,
    )
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
    if not os.path.exists(args.tokenizer):
        logger.error("Tokenizer not found: %s", args.tokenizer)
        return 1

    model = build_ai_edge_model(config)
    load_checkpoint_into(model, args.checkpoint, config)

    tflite_name = os.path.splitext(args.output)[0] + ".tflite"
    convert_to_tflite(model, config, tflite_name)

    metadata = {
        "model_name": args.model_name,
        "format": "litertlm",
        "quantization": "int4_per_channel",
        "context_length": config.max_seq_len,
        "parameters_estimate": config.estimated_parameters(),
        "architecture": "decoder-only-gqa",
        "agentic": True,
        "framework": "litert-lm",
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
        choices=["hermes-1b", "hermes-270m"],
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
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
