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
    """Enable fast math paths appropriate to the backend (CUDA: TF32 + autotuner)."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def amp_dtype(device: torch.device) -> torch.dtype:
    """Preferred autocast dtype per backend (CUDA bf16/fp16, MPS fp16, CPU fp32)."""
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
        return f"cuda ({torch.cuda.get_device_name(device)}, autocast={amp_dtype(device)})"
    if device.type == "mps":
        return f"mps (Apple Metal, autocast={amp_dtype(device)})"
    return "cpu (fp32)"
