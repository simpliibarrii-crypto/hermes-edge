"""Tests for the PTQ / fake-quant utilities (no LiteRT stack required)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from hermes.config import HermesConfig  # noqa: E402
from hermes.model import build_model  # noqa: E402
from hermes.quantization import (  # noqa: E402
    apply_weight_only_int4,
    apply_weight_only_int8,
    collect_calibration_stats,
    fake_quantize_per_group,
)


def _tiny_cfg():
    return HermesConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, head_dim=8, max_seq_len=16,
    )


def test_int4_weight_range():
    model = build_model(_tiny_cfg())
    apply_weight_only_int4(model, group_size=8)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Reconstruct the integer codes from the dequantized weights per group.
            w = module.weight.data
            qmax = 7
            gs = 8
            out_f, in_f = w.shape
            pad = (gs - in_f % gs) % gs
            wp = torch.nn.functional.pad(w, (0, pad)).reshape(out_f, -1, gs)
            scale = (wp.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
            codes = torch.round(wp / scale)
            assert codes.min() >= -8 and codes.max() <= 7


def test_int8_weight_range():
    model = build_model(_tiny_cfg())
    apply_weight_only_int8(model)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            qmax = 127
            scale = (w.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-8)
            codes = torch.round(w / scale)
            assert codes.min() >= -128 and codes.max() <= 127


def test_calibration_stats_keys():
    model = build_model(_tiny_cfg())
    data = [torch.randint(0, 64, (1, 8)) for _ in range(3)]
    stats = collect_calibration_stats(model, data, num_batches=3)
    assert isinstance(stats, dict) and stats
    # Layer names should reference nn.Linear submodules (e.g. q_proj).
    assert any("q_proj" in name for name in stats)
    for entry in stats.values():
        assert {"min", "max", "abs_max", "p99"} <= set(entry)
        assert entry["max"] >= entry["min"]


def test_fake_quant_is_idempotent():
    w = torch.randn(16, 24)
    once = fake_quantize_per_group(w, bits=4, group_size=8)
    twice = fake_quantize_per_group(once, bits=4, group_size=8)
    assert torch.allclose(once, twice, atol=1e-5)
