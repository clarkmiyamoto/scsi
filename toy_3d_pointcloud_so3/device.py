"""Device selection and per-backend performance configuration (CUDA / MPS / CPU)."""
from __future__ import annotations

import contextlib

import torch


def available_device() -> torch.device:
    """Pick the best available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str = "auto") -> torch.device:
    """Map a CLI string ('auto'|'cuda'|'mps'|'cpu') to a torch.device."""
    return available_device() if name == "auto" else torch.device(name)


def configure_backends(device: torch.device) -> None:
    """Enable fast math paths appropriate to the backend.

    CUDA: TF32 matmul/conv + cuDNN autotuner (big speedups on Ampere+).
    MPS / CPU: nothing global to toggle; speed comes from autocast + on-device data.
    """
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True          # autotune kernels for fixed shapes
        torch.set_float32_matmul_precision("high")     # let fp32 matmuls use TF32


def amp_dtype(device: torch.device) -> torch.dtype:
    """Preferred autocast dtype per backend.

    CUDA: bf16 if the GPU supports it (no loss scaling needed), else fp16.
    MPS:  fp16 (bf16 autocast is not reliably supported on Metal).
    CPU:  fp32 (autocast disabled).
    """
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def autocast(device: torch.device, enabled: bool = True):
    """Mixed-precision context manager, or a no-op on CPU / when disabled."""
    if not enabled or device.type == "cpu":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype(device))


def needs_grad_scaler(device: torch.device, use_amp: bool) -> bool:
    """fp16 underflows gradients -> needs a GradScaler. bf16/fp32 do not."""
    return use_amp and device.type == "cuda" and amp_dtype(device) == torch.float16


def synchronize(device: torch.device) -> None:
    """Block until queued accelerator work finishes (for honest timing)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def describe(device: torch.device) -> str:
    """Human-readable one-liner about the active device."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"cuda ({name}, autocast={amp_dtype(device)})"
    if device.type == "mps":
        return f"mps (Apple Metal, autocast={amp_dtype(device)})"
    return "cpu (fp32)"
