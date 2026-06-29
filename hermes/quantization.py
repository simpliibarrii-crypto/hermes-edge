"""Post-training quantization (PTQ) analysis + fake-quant utilities.

These helpers are deliberately **standalone** — they have no ``ai_edge_torch``
dependency. They serve two purposes:

1. **Pre-conversion analysis.** :func:`collect_calibration_stats` and
   :func:`quantization_error_report` let you measure activation ranges and the
   weight/perplexity error a given bit-width would introduce, *before* you spend
   minutes lowering the model through the LiteRT stack. Use them to sanity-check
   that INT4 is viable for a checkpoint, or to pick which layers are sensitive.

2. **Training-time fake quantization.** :func:`apply_weight_only_int4` and
   :func:`apply_weight_only_int8` replace each ``nn.Linear`` weight with its
   quantized-then-dequantized value using a straight-through estimator (STE) so
   gradients still flow. This is the quantization-aware-training (QAT) path: fine
   tune with fake-quant on to recover accuracy the real INT4 graph would lose.

Relationship to ``scripts/convert_to_litertlm.py``
--------------------------------------------------
The *real* mobile INT4 graph is produced by ``convert_to_litertlm.py`` via
``ai_edge_torch``'s ``full_int4_dynamic_recipe`` — that is what actually ships in
the ``.litertlm`` bundle. The functions here do **not** replace that conversion:
they approximate the same symmetric per-group INT4 scheme in pure PyTorch so you
can (a) estimate the error offline and (b) QAT-finetune to minimize it. Numbers
from here are guidance; the converter's output is ground truth.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Symmetric per-group quantization core
# --------------------------------------------------------------------------- #
def _quant_levels(bits: int) -> tuple[int, int]:
    """Return ``(qmin, qmax)`` for a signed ``bits``-bit integer."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    return qmin, qmax


def fake_quantize_per_group(
    weight: torch.Tensor, bits: int, group_size: int
) -> torch.Tensor:
    """Symmetric per-group fake quantization of a 2-D weight matrix.

    The weight ``[out_features, in_features]`` is split along ``in_features`` into
    groups of ``group_size``; each group gets its own scale ``max(|w|) / qmax``.
    The result is quantized to the integer grid and dequantized back to float, so
    the returned tensor has the same dtype/shape but only takes representable
    values. Used by both the analysis and STE paths.
    """
    qmin, qmax = _quant_levels(bits)
    out_features, in_features = weight.shape
    gs = group_size if group_size > 0 else in_features
    pad = (gs - in_features % gs) % gs
    w = weight
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    w = w.reshape(out_features, -1, gs)

    max_abs = w.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / qmax).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / scale), qmin, qmax)
    deq = (q * scale).reshape(out_features, -1)
    if pad:
        deq = deq[:, :in_features]
    return deq.to(weight.dtype)


class _STEFakeQuant(torch.autograd.Function):
    """Straight-through estimator: quantize on forward, identity on backward."""

    @staticmethod
    def forward(ctx, weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:  # type: ignore[override]
        return fake_quantize_per_group(weight, bits, group_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Identity gradient w.r.t. the weight; None for the int hyper-params.
        return grad_output, None, None


def _apply_weight_only(model: nn.Module, bits: int, group_size: int) -> nn.Module:
    """In-place STE fake-quant of every ``nn.Linear`` weight in ``model``."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                quantized = _STEFakeQuant.apply(module.weight, bits, group_size)
                module.weight.copy_(quantized)
    return model


def apply_weight_only_int4(model: nn.Module, group_size: int = 128) -> nn.Module:
    """Fake-quantize all ``nn.Linear`` weights to symmetric per-group INT4.

    Each weight is mapped onto the signed 4-bit grid ``[-8, 7]`` (per group of
    ``group_size`` input channels) and dequantized in place. Uses a
    straight-through estimator so the operation is differentiable for QAT.

    This mirrors the per-group INT4 scheme that
    ``ai_edge_torch``'s ``full_int4_dynamic_recipe`` applies during the real
    conversion in ``scripts/convert_to_litertlm.py`` — call this to QAT-finetune
    or to estimate INT4 error offline; the converter produces the shipped graph.

    Returns the same model (mutated in place).
    """
    return _apply_weight_only(model, bits=4, group_size=group_size)


def apply_weight_only_int8(model: nn.Module, group_size: int = 0) -> nn.Module:
    """Fake-quantize all ``nn.Linear`` weights to symmetric INT8 (``[-128, 127]``).

    Per-channel by default (``group_size=0`` → one scale per output row). Same STE
    semantics as :func:`apply_weight_only_int4`; useful as the higher-quality
    fallback recipe when INT4 degrades a sensitive checkpoint too much.

    Returns the same model (mutated in place).
    """
    return _apply_weight_only(model, bits=8, group_size=group_size)


# --------------------------------------------------------------------------- #
# Calibration + error analysis
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_calibration_stats(
    model: nn.Module,
    dataloader: Iterable,
    num_batches: int = 64,
) -> Dict[str, Dict[str, float]]:
    """Run forward passes and collect per-layer activation statistics.

    Forward hooks on every ``nn.Linear`` record the running min/max and a coarse
    99th-percentile estimate of the *output* activations across up to
    ``num_batches`` batches. These ranges are what an activation-quantization
    scheme (or a converter calibration pass) would use to pick scales.

    Args:
        model: The model to profile (set to eval).
        dataloader: Yields either tensors of ``input_ids`` or ``(inputs, _)``
            tuples / dicts with an ``input_ids`` key.
        num_batches: Max number of batches to run.

    Returns:
        ``{layer_name: {"min", "max", "abs_max", "p99", "mean", "num_samples"}}``.
    """
    model.eval()
    stats: Dict[str, Dict[str, float]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, out):
            t = out.detach()
            if not torch.is_floating_point(t):
                return
            flat = t.float().reshape(-1)
            entry = stats.setdefault(
                name,
                {
                    "min": math.inf,
                    "max": -math.inf,
                    "abs_max": 0.0,
                    "p99": 0.0,
                    "mean": 0.0,
                    "num_samples": 0.0,
                },
            )
            entry["min"] = min(entry["min"], float(flat.min()))
            entry["max"] = max(entry["max"], float(flat.max()))
            entry["abs_max"] = max(entry["abs_max"], float(flat.abs().max()))
            # Running mean + percentile (cheap quantile on a subsample).
            n_prev = entry["num_samples"]
            n_new = flat.numel()
            entry["mean"] = (
                entry["mean"] * n_prev + float(flat.sum())
            ) / max(n_prev + n_new, 1)
            sample = flat if flat.numel() <= 16384 else flat[torch.randint(
                0, flat.numel(), (16384,), device=flat.device)]
            entry["p99"] = max(entry["p99"], float(torch.quantile(sample.abs(), 0.99)))
            entry["num_samples"] = n_prev + n_new

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            input_ids = _extract_input_ids(batch)
            model(input_ids)
    finally:
        for h in handles:
            h.remove()

    return stats


def _extract_input_ids(batch) -> torch.Tensor:
    """Pull an ``input_ids`` tensor out of common dataloader batch shapes."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        return batch["input_ids"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    raise TypeError(f"Cannot extract input_ids from batch of type {type(batch)}.")


@torch.no_grad()
def _perplexity(model: nn.Module, dataloader: Iterable, num_batches: int) -> float:
    """Mean token-level perplexity over ``num_batches`` (labels == inputs)."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        input_ids = _extract_input_ids(batch)
        out = model(input_ids, labels=input_ids)
        loss = out["loss"] if isinstance(out, dict) else out
        if loss is None:
            continue
        total_loss += float(loss)
        count += 1
    if count == 0:
        return float("nan")
    return math.exp(total_loss / count)


@torch.no_grad()
def quantization_error_report(
    original_model: nn.Module,
    quantized_model: nn.Module,
    dataloader: Iterable,
    num_batches: int = 8,
) -> Dict[str, object]:
    """Compare a model against its quantized copy.

    Computes, per ``nn.Linear`` layer, the relative L2 error between the original
    and quantized weights, and the model-level perplexity delta on ``dataloader``.

    Returns:
        ``{"per_layer_l2": {name: rel_l2}, "max_layer_l2": float,
           "perplexity_original": float, "perplexity_quantized": float,
           "perplexity_delta": float}``.
    """
    orig_linears = dict(_named_linears(original_model))
    quant_linears = dict(_named_linears(quantized_model))

    per_layer: Dict[str, float] = {}
    for name, orig in orig_linears.items():
        if name not in quant_linears:
            continue
        diff = (orig.weight - quant_linears[name].weight).float()
        denom = orig.weight.float().norm().clamp(min=1e-8)
        per_layer[name] = float(diff.norm() / denom)

    ppl_orig = _perplexity(original_model, dataloader, num_batches)
    ppl_quant = _perplexity(quantized_model, dataloader, num_batches)

    return {
        "per_layer_l2": per_layer,
        "max_layer_l2": max(per_layer.values()) if per_layer else 0.0,
        "perplexity_original": ppl_orig,
        "perplexity_quantized": ppl_quant,
        "perplexity_delta": ppl_quant - ppl_orig,
    }


def _named_linears(model: nn.Module):
    """Yield ``(name, module)`` for every ``nn.Linear`` in ``model``."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import copy

    from hermes.config import HermesConfig
    from hermes.model import build_model

    cfg = HermesConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=16, max_seq_len=32,
    )
    fp_model = build_model(cfg)
    q_model = apply_weight_only_int4(copy.deepcopy(fp_model))
    data = [torch.randint(0, cfg.vocab_size, (1, 8)) for _ in range(4)]
    report = quantization_error_report(fp_model, q_model, data, num_batches=4)
    print("max layer L2 error:", round(report["max_layer_l2"], 4))
    print("perplexity delta:", round(report["perplexity_delta"], 4))
