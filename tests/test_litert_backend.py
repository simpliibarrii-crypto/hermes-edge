"""Tests for LiteRT backend selection."""

import subprocess

from hermes.litert_model import LiteRTModel


def _model(path, marker=b"GPU Vulkan CPU"):
    path.write_bytes(b"LITERTLM" + marker)
    return path


def test_auto_backend_prefers_gpu_then_fallbacks(tmp_path):
    model = LiteRTModel(str(_model(tmp_path / "model.litertlm")), backend="auto")
    assert model.load()

    assert model.get_recommended_backend() == "gpu"
    assert model.get_backend_attempts()[:3] == ["gpu", "vulkan", "cpu"]
    assert model.active_backend == "gpu"


def test_auto_backend_attempts_gpu_first_when_model_metadata_is_unknown(tmp_path):
    model = LiteRTModel(str(_model(tmp_path / "model.litertlm", marker=b"")), backend="auto")
    assert model.load()

    assert model.get_backend_attempts() == ["gpu", "vulkan", "metal", "ane", "cpu"]


def test_explicit_backend_does_not_fallback(tmp_path):
    model = LiteRTModel(str(_model(tmp_path / "model.litertlm")), backend="cpu")
    assert model.load()

    assert model.get_backend_attempts() == ["cpu"]
    assert model.active_backend == "cpu"


def test_generate_tries_gpu_before_cpu(tmp_path, monkeypatch):
    model = LiteRTModel(str(_model(tmp_path / "model.litertlm", marker=b"GPU CPU")), backend="auto")
    assert model.load()
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(cmd)
        backend = cmd[cmd.index("--backend") + 1]
        if backend == "gpu":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="gpu unavailable")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok from cpu", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert model.generate("hello") == "ok from cpu"
    assert [cmd[cmd.index("--backend") + 1] for cmd in calls] == ["gpu", "cpu"]
    assert model.active_backend == "cpu"
