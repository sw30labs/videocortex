"""Device selection.

Upstream's ``TribeModel.from_pretrained(device="auto")`` resolves to
``"cuda" if torch.cuda.is_available() else "cpu"``. On Apple Silicon that
silently means CPU, which is roughly an order of magnitude slower than the
Metal backend sitting right there. This module fixes that, and is written so
the decision logic can be unit-tested without torch installed.
"""

from __future__ import annotations

import os
import platform
import typing as tp

VALID_DEVICES = ("auto", "cuda", "mps", "cpu")

# Ops the TRIBE stack touches that were still missing from the MPS backend as
# of torch 2.6. Falling back to CPU per-op is far cheaper than running the
# whole graph on CPU, but it has to be set before torch is imported.
_MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


class DeviceError(RuntimeError):
    """Raised when an explicitly requested device is not available."""


def select_device(
    preferred: str = "auto",
    *,
    has_cuda: bool = False,
    has_mps: bool = False,
) -> str:
    """Pure decision function: given what's available, pick a device.

    Separated from torch so the priority order is testable in isolation.

    >>> select_device("auto", has_cuda=False, has_mps=True)
    'mps'
    >>> select_device("auto", has_cuda=False, has_mps=False)
    'cpu'
    """
    if preferred not in VALID_DEVICES:
        raise ValueError(
            f"device must be one of {VALID_DEVICES}, got {preferred!r}"
        )

    if preferred == "auto":
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
        return "cpu"

    if preferred == "cuda" and not has_cuda:
        raise DeviceError(
            "CUDA was requested but torch reports no CUDA device. "
            "Use --device auto to fall back."
        )
    if preferred == "mps" and not has_mps:
        raise DeviceError(
            "MPS was requested but torch reports no Metal device. On macOS this "
            "usually means a non-Apple-Silicon machine, a torch build without "
            "MPS, or that you are inside a Linux VM (Metal does not cross that "
            "boundary). Use --device auto to fall back."
        )
    return preferred


def enable_mps_fallback() -> None:
    """Let unimplemented MPS ops silently run on CPU instead of crashing.

    Must happen before ``import torch``, hence the env var rather than a call.
    """
    os.environ.setdefault(_MPS_FALLBACK_ENV, "1")


def probe() -> dict[str, tp.Any]:
    """Report what torch can see. Returns a dict rather than raising."""
    info: dict[str, tp.Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": None,
        "has_cuda": False,
        "has_mps": False,
        "note": None,
    }
    try:
        import torch
    except ImportError:
        info["note"] = "torch is not installed (pip install 'videocortex[predict]')"
        return info

    info["torch"] = torch.__version__
    info["has_cuda"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    info["has_mps"] = bool(mps is not None and mps.is_available())

    if platform.system() == "Darwin" and not info["has_mps"]:
        info["note"] = (
            "macOS detected but MPS is unavailable — check that torch was "
            "installed as an arm64 wheel (python -c 'import platform; "
            "print(platform.machine())' should print arm64, not x86_64)."
        )
    return info


def resolve_device(preferred: str = "auto") -> str:
    """Resolve ``preferred`` against the machine we're actually on."""
    enable_mps_fallback()
    info = probe()
    if info["torch"] is None:
        raise DeviceError(
            "torch is required to run the model. Install with:\n"
            "    pip install 'videocortex[predict]'"
        )
    return select_device(
        preferred, has_cuda=info["has_cuda"], has_mps=info["has_mps"]
    )


def describe_device(device: str) -> str:
    """One human-readable line about what we're about to run on."""
    if device == "cuda":
        try:
            import torch

            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    if device == "mps":
        return f"mps (Apple {platform.machine()} / Metal)"
    return f"cpu ({platform.machine()}) — expect this to be slow"
