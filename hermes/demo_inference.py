"""Lightweight deterministic inference used by the public Hermes Space.

This module deliberately has no Torch, Transformers, or model-weight dependency.
It demonstrates the interaction and routing surface only. Real inference lives
in :mod:`hermes.inference` and requires the explicit ``model`` extra.
"""

from __future__ import annotations

from .config import PRESETS


class DemoHermesInference:
    """Deterministic architecture demo that mirrors a Hermes model profile."""

    def __init__(self, preset: str):
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        self.preset_name = preset
        self.preset = PRESETS[preset]()

    def chat(self, message: str) -> str:
        """Return a transparent, deterministic routing demonstration."""

        prompt = (message or "").strip()
        if not prompt:
            return "Send a prompt to inspect the demonstration route."

        parameter_count = self.preset.num_parameters()
        return (
            f"[Hermes Edge architecture demo · {self.preset_name}]\n\n"
            f"Prompt received: {prompt}\n\n"
            "Route contract:\n"
            "1. Check registered deterministic tools.\n"
            "2. Compare the model profile with the device memory budget.\n"
            "3. Select an available accelerated backend before CPU fallback.\n"
            "4. Record the route, profile, backend, and fallback state.\n\n"
            f"Reference profile parameters: {parameter_count:,}.\n"
            "This response is deterministic and uses no trained model weights."
        )
