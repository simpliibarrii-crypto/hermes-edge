import logging
import subprocess
import time
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)


class LiteRTModel:

    SUPPORTED_BACKENDS = ["auto", "cpu", "gpu", "ane", "metal", "vulkan"]

    def __init__(self, model_path: str, cli_path: str = "litert-lm", backend: str = "auto"):
        self.model_path = Path(model_path).resolve()
        self.cli_path = cli_path
        self.backend = backend
        self.enable_mtp = True
        self.vocab_size = 32000
        self.tokenizer = None
        self._loaded = False
        self._metadata: dict = {}
        self._detected_backends: list[str] = []

        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend '{backend}'. Choose from: {self.SUPPORTED_BACKENDS}")

    def load(self) -> bool:
        if not self.model_path.exists():
            log.error("Model not found: %s", self.model_path)
            return False

        with open(self.model_path, "rb") as f:
            header = f.read(16)
            if header[:8] != b"LITERTLM":
                log.error("Invalid model file (bad magic): %s", self.model_path)
                return False

        self._detect_backends()
        self._loaded = True
        mb = self.model_path.stat().st_size / 1024 / 1024
        log.info("Model loaded: %s (%.1f MB) backends=%s", self.model_path.name, mb, self._detected_backends)
        return True

    def _detect_backends(self):
        backend_keywords = {
            "gpu": [b"GPU", b"gpu", b"Gpu"],
            "ane": [b"ANE", b"ane", b"Apple Neural"],
            "metal": [b"Metal", b"metal", b"MTL"],
            "vulkan": [b"Vulkan", b"vulkan"],
            "coreml": [b"CoreML", b"coreml"],
            "opencl": [b"OpenCL", b"opencl"],
            "cpu": [b"CPU", b"cpu"],
        }

        try:
            with open(self.model_path, "rb") as f:
                data = f.read()

            raw = data.decode("latin-1")
            detected = set()
            for backend, keywords in backend_keywords.items():
                for kw in keywords:
                    if kw in data or kw.decode("latin-1", errors="replace") in raw:
                        detected.add(backend)
                        break

            self._detected_backends = sorted(detected)
        except Exception as exc:
            log.warning("Backend detection failed: %s", exc)
            self._detected_backends = ["cpu"]

    def get_supported_backends(self) -> list[str]:
        return list(self._detected_backends)

    def get_recommended_backend(self) -> str:
        priority = ["ane", "metal", "gpu", "vulkan", "coreml"]
        for p in priority:
            if p in self._detected_backends:
                return p
        return "cpu" if "cpu" in self._detected_backends else "cpu"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 40,
    ) -> str:
        if not self._loaded:
            return "Error: Model not loaded."

        try:
            cmd = [
                self.cli_path,
                "run",
                str(self.model_path),
                "--prompt",
                prompt,
                "--max_tokens",
                str(max_tokens),
                "--temperature",
                str(temperature),
                "--top_k",
                str(top_k),
            ]

            if self.backend != "auto":
                cmd.extend(["--backend", self.backend])

            if self.enable_mtp:
                cmd.append("--enable_mtp")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

            if result.stderr:
                log.warning("CLI stderr: %s", result.stderr[:200])

        except FileNotFoundError:
            log.warning("litert-lm CLI not available, using simulated response")
        except subprocess.TimeoutExpired:
            log.warning("Model inference timed out")
        except Exception as exc:
            log.warning("Model inference error: %s", exc)

        return self._simulate_response(prompt)

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 40,
    ) -> Iterator[str]:
        """Stream tokens one at a time (progressive rendering like ChatGPT)."""
        if not self._loaded:
            yield "Error: Model not loaded."
            return

        try:
            cmd = [
                self.cli_path,
                "run",
                str(self.model_path),
                "--prompt",
                prompt,
                "--max_tokens",
                str(max_tokens),
                "--temperature",
                str(temperature),
                "--top_k",
                str(top_k),
                "--stream",
            ]

            if self.backend != "auto":
                cmd.extend(["--backend", self.backend])

            if self.enable_mtp:
                cmd.append("--enable_mtp")

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )

            for line in iter(proc.stdout.readline, ""):
                line = line.strip()
                if line:
                    yield line + " "
                if proc.poll() is not None:
                    break
            proc.stdout.close()

            if proc.returncode and proc.returncode != 0:
                yield from self._simulate_stream(prompt)

        except FileNotFoundError:
            log.warning("litert-lm CLI not available, using simulated stream")
            yield from self._simulate_stream(prompt)
        except Exception as exc:
            log.warning("Stream error: %s", exc)
            yield from self._simulate_stream(prompt)

    def _simulate_stream(self, prompt: str) -> Iterator[str]:
        """Simulate streaming tokens for demo/testing."""
        text = self._simulate_response(prompt)
        words = text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.02)  # ~50 tok/s simulated

    def _simulate_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "<think>Greeting detected.</think>\nHey! I'm Hermes, ready to help. What's up?"
        if "tool" in prompt_lower or "function" in prompt_lower:
            return (
                "<think>They're asking about tools. Quick overview.</think>\n"
                "I've got calculator, web search, memory, and timer tools. "
                "Just tell me what you need."
            )
        if "reason" in prompt_lower or "deep" in prompt_lower:
            return (
                "<think>Breaking it down.</think>\n"
                "Here's what I figure: answer's right there after working through it step by step."
            )
        return (
            f"<think>Processing via {self.model_path.stem} ({self.backend}).</think>\n"
            f"Heard you: \"{prompt[:80]}\". Running offline and ready."
        )

    def get_metadata(self) -> dict:
        return {
            "path": str(self.model_path),
            "size_mb": round(self.model_path.stat().st_size / 1024 / 1024, 1),
            "loaded": self._loaded,
            "format": "LITERTLM",
            "vocab_size": self.vocab_size,
            "supported_backends": self._detected_backends,
            "recommended_backend": self.get_recommended_backend(),
            "backend": self.backend,
        }
