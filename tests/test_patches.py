"""The whisperx patch is the difference between a run and a stack trace on macOS."""

import subprocess

import pytest

from videocortex.patches import (
    _is_cpu_whisperx,
    _rewrite_uvx_whisperx,
    _whisperx_python,
    whisperx_cpu_compat,
)

CPU_CMD = [
    "uvx", "whisperx", "in.wav", "--model", "large-v3",
    "--device", "cpu", "--compute_type", "float16", "--batch_size", "16",
]
CUDA_CMD = [*CPU_CMD[:6], "cuda", *CPU_CMD[7:]]


def test_detects_only_the_cpu_whisperx_invocation():
    assert _is_cpu_whisperx(CPU_CMD)
    assert not _is_cpu_whisperx(CUDA_CMD)
    assert not _is_cpu_whisperx(["ffmpeg", "-i", "a.mp4"])
    assert not _is_cpu_whisperx("uvx whisperx --device cpu")  # str, not list


def test_rewrites_float16_to_int8_on_cpu(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(list(cmd)))
    with whisperx_cpu_compat():
        subprocess.run(CPU_CMD)
    assert seen[0][seen[0].index("--compute_type") + 1] == "int8"


def test_cpu_uvx_pins_python_and_torch_2_6():
    out = _rewrite_uvx_whisperx(list(CPU_CMD))
    assert out[out.index("--python") + 1] == _whisperx_python()
    assert "torch==2.6.0" in out
    assert "torchaudio==2.6.0" in out
    assert out[out.index("--compute_type") + 1] == "int8"
    assert out[out.index("whisperx") + 1] == "in.wav"


def test_cuda_uvx_pins_python_but_not_cpu_torch():
    out = _rewrite_uvx_whisperx(list(CUDA_CMD))
    assert out[out.index("--python") + 1] == _whisperx_python()
    assert "torch==2.6.0" not in out
    assert out[out.index("--compute_type") + 1] == "float16"


def test_leaves_cuda_invocations_compute_type_alone(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(list(cmd)))
    with whisperx_cpu_compat():
        subprocess.run(CUDA_CMD)
    assert seen[0][seen[0].index("--compute_type") + 1] == "float16"


def test_whisperx_subprocess_gets_weights_only_escape(monkeypatch):
    seen = []

    def fake(cmd, *a, **k):
        seen.append(k.get("env"))

    monkeypatch.setattr(subprocess, "run", fake)
    with whisperx_cpu_compat():
        subprocess.run(CPU_CMD, env={"FOO": "1"})
    assert seen[0]["FOO"] == "1"
    assert seen[0]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_unrelated_subprocess_calls_pass_through_untouched(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: seen.append(list(cmd)))
    with whisperx_cpu_compat():
        subprocess.run(["ffmpeg", "-version"])
    assert seen[0] == ["ffmpeg", "-version"]


def test_mps_text_kwargs_are_eager_float32():
    torch = pytest.importorskip("torch")
    from videocortex.patches import mps_text_from_pretrained_kwargs

    kw = mps_text_from_pretrained_kwargs("mps", {})
    assert kw["attn_implementation"] == "eager"
    assert kw["torch_dtype"] == torch.float32
    assert mps_text_from_pretrained_kwargs("cpu", {}) == {}
    assert mps_text_from_pretrained_kwargs("cuda", {"foo": 1}) == {"foo": 1}


def test_allow_mps_extractors_is_idempotent():
    pytest.importorskip("neuralset")
    from neuralset.extractors.text import HuggingFaceText

    from videocortex.patches import allow_mps_extractors

    allow_mps_extractors()
    allow_mps_extractors()
    HuggingFaceText(model_name="gpt2", device="mps")


def test_original_run_is_restored_even_after_an_exception(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(subprocess, "run", sentinel)
    try:
        with whisperx_cpu_compat():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert subprocess.run is sentinel
