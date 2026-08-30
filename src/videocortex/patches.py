"""Small, surgical fixes for running TRIBE v2 off a CUDA box.

Everything here is a workaround for an upstream assumption that the machine
has an NVIDIA GPU. Each patch names the upstream code it compensates for so
that when Meta fixes it, the corresponding patch can be deleted rather than
quietly rotting.

Written against facebookresearch/tribev2 @ main, March 2026.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import typing as tp
from pathlib import Path
from typing import Literal, get_args

logger = logging.getLogger(__name__)

_MPS_EXTRACTORS_PATCHED = False
_LLAMA_MPS_PATCHED = False

# faster-whisper (which whisperx drives) refuses float16 on CPU. These are the
# compute types it will actually accept there.
_CPU_COMPUTE_TYPE = "int8"

# uvx without a pin picks the newest Python on PATH (conda 3.14 here) and a
# torchaudio that already dropped list_audio_backends (removed in 2.9).
# pyannote.audio still calls it. Pin the same torch range this package uses.
_WHISPERX_TORCH = "torch==2.6.0"
_WHISPERX_TORCHAUDIO = "torchaudio==2.6.0"

# torch 2.6 flipped torch.load to weights_only=True. pyannote VAD checkpoints
# pickle omegaconf.ListConfig, which that unpickler rejects. The env var is
# the supported override when the callsite did not pass weights_only.
_WHISPERX_TORCH_ENV = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}


def _whisperx_python() -> str:
    """Interpreter for the uvx whisperx env. Cap at 3.12; 3.13/3.14 break pyannote."""
    v = sys.version_info
    if (v.major, v.minor) >= (3, 13):
        return "3.12"
    if (v.major, v.minor) < (3, 11):
        return "3.11"
    return f"{v.major}.{v.minor}"


@contextlib.contextmanager
def whisperx_cpu_compat() -> tp.Iterator[None]:
    """Rewrite the whisperx command line at the subprocess boundary.

    ``tribev2.eventstransforms.ExtractWordsFromAudio`` builds a fixed command::

        uvx whisperx ... --device {cuda|cpu} --compute_type float16 ...

    Three upstream assumptions fail off a CUDA box:

    1. Device is ``cuda`` iff ``torch.cuda.is_available()``, else ``cpu``.
    2. Compute type is hard-coded ``float16``. faster-whisper refuses that on CPU.
    3. ``uvx`` with no ``--python`` follows the newest interpreter (today: 3.14)
       and a torchaudio that no longer has ``list_audio_backends``.
    4. torch 2.6 ``weights_only=True`` cannot unpickle pyannote's VAD checkpoint.

    We intercept ``subprocess.run`` so transcript parsing stays upstream's.
    """
    import subprocess

    original_run = subprocess.run

    def patched_run(cmd, *args, **kwargs):
        if _is_uvx_whisperx(cmd):
            cmd = _rewrite_uvx_whisperx(list(cmd))
            env = kwargs.get("env")
            env = dict(env) if env is not None else os.environ.copy()
            for key, value in _WHISPERX_TORCH_ENV.items():
                env.setdefault(key, value)
            kwargs["env"] = env
        return original_run(cmd, *args, **kwargs)

    subprocess.run = patched_run
    try:
        yield
    finally:
        subprocess.run = original_run


def _is_uvx_whisperx(cmd: tp.Any) -> bool:
    if not isinstance(cmd, (list, tuple)) or len(cmd) < 2:
        return False
    prog = Path(str(cmd[0])).name
    return prog in {"uvx", "uv"} and "whisperx" in cmd


def _whisperx_device(cmd: list[str]) -> str | None:
    try:
        return cmd[cmd.index("--device") + 1]
    except (ValueError, IndexError):
        return None


def _rewrite_uvx_whisperx(cmd: list[str]) -> list[str]:
    tool = cmd.index("whisperx")
    prefix, rest = cmd[:tool], cmd[tool:]
    inserts: list[str] = []
    if "--python" not in prefix:
        inserts += ["--python", _whisperx_python()]
    cpu = _whisperx_device(cmd) != "cuda"
    if cpu and "--with" not in prefix:
        inserts += ["--with", _WHISPERX_TORCH, "--with", _WHISPERX_TORCHAUDIO]
        logger.info(
            "whisperx via uvx: pinning python %s, %s, %s",
            _whisperx_python(),
            _WHISPERX_TORCH,
            _WHISPERX_TORCHAUDIO,
        )
    elif inserts:
        logger.info("whisperx via uvx: pinning python %s", _whisperx_python())
    cmd = prefix + inserts + rest
    if cpu and "--compute_type" in cmd:
        i = cmd.index("--compute_type")
        if cmd[i + 1] != _CPU_COMPUTE_TYPE:
            logger.info(
                "whisperx on CPU: rewriting --compute_type %s -> %s",
                cmd[i + 1],
                _CPU_COMPUTE_TYPE,
            )
            cmd[i + 1] = _CPU_COMPUTE_TYPE
    return cmd


def allow_mps_extractors() -> None:
    """Widen neuralset's device Literal so ``config_update`` can say ``mps``.

    ``HuggingFaceMixin.device`` is ``Literal["auto", "cpu", "cuda", "accelerate"]``.
    Metal is none of those, so pydantic kills the load before a weight moves.
    Idempotent: a second call is a no-op.
    """
    global _MPS_EXTRACTORS_PATCHED
    if _MPS_EXTRACTORS_PATCHED:
        return

    import neuralset.extractors.audio  # noqa: F401
    import neuralset.extractors.image  # noqa: F401
    import neuralset.extractors.text  # noqa: F401
    import neuralset.extractors.video  # noqa: F401
    from neuralset.extractors.base import HuggingFaceMixin

    classes: list[type] = []
    stack = [HuggingFaceMixin]
    seen = {HuggingFaceMixin}
    while stack:
        cur = stack.pop()
        classes.append(cur)
        for sub in cur.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)

    for cls in classes:
        field = getattr(cls, "model_fields", {}).get("device")
        if field is None:
            continue
        args = get_args(field.annotation)
        if not args or "mps" in args or "cuda" not in args:
            continue
        field.annotation = Literal[(*args, "mps")]

    # Parent first so children rebuild against the widened field.
    for cls in classes:
        if hasattr(cls, "model_rebuild"):
            cls.model_rebuild(force=True)

    _MPS_EXTRACTORS_PATCHED = True
    logger.info("neuralset extractors: device Literal now includes mps")


def mps_text_from_pretrained_kwargs(device: str, kwargs: dict) -> dict:
    """Llama 3.2 GQA (24 q-heads, 8 kv-heads) aborts in MPSGraph SDPA.

    Eager + float32 is the combination that actually returns on Metal.
    SDPA + float32 still LLVM-errors. Verified against Llama-3.2-3B.
    """
    if device != "mps":
        return kwargs
    import torch

    out = dict(kwargs)
    out.setdefault("attn_implementation", "eager")
    out.setdefault("torch_dtype", torch.float32)
    return out


def llama_mps_eager() -> None:
    """Force HuggingFaceText on Metal onto eager attention and float32.

    Idempotent. neuralset's ``_load_model`` otherwise does
    ``AutoModel.from_pretrained(name)`` then ``.to("mps")``, which lands in
    bf16 SDPA and takes the process down with no Python traceback.
    """
    global _LLAMA_MPS_PATCHED
    if _LLAMA_MPS_PATCHED:
        return
    from neuralset.extractors.text import HuggingFaceText

    original = HuggingFaceText._load_model

    def _load_model(self, **kwargs):
        kwargs = mps_text_from_pretrained_kwargs(getattr(self, "device", ""), kwargs)
        if kwargs.get("attn_implementation") == "eager":
            logger.info(
                "Llama on MPS: attn_implementation=eager torch_dtype=float32 "
                "(SDPA GQA aborts in MPSGraph)"
            )
        return original(self, **kwargs)

    HuggingFaceText._load_model = _load_model  # type: ignore[method-assign]
    _LLAMA_MPS_PATCHED = True


def _is_cpu_whisperx(cmd: tp.Any) -> bool:
    """True only for the exact command shape we mean to rewrite."""
    if not _is_uvx_whisperx(cmd):
        return False
    if "--compute_type" not in cmd:
        return False
    return _whisperx_device(list(cmd)) != "cuda"
