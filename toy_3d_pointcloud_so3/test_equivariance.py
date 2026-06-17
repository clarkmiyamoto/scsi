"""SO(3)-equivariance checks for the Vector Neuron point-cloud models.

Run as a script (asserts + summary, non-zero exit on failure):

    uv run python -m toy_3d_pointcloud_so3.test_equivariance

The ``test_*`` functions are also pytest-compatible. Everything runs on CPU in
float64 so the equivariance residuals are at numerical-precision level, with the
rotation convention fixed to that of the corruption channel: a rotated cloud is
``x @ R.T`` and an equivariant map satisfies ``f(x @ R.T) == f(x) @ R.T``.
"""
from __future__ import annotations

import torch

from .corruption import random_so3
from .model import (
    ConditionalPointCloudVelocity,
    PointCloudVelocity,
    _vec_channels,
)
from .vn import VNLayerNorm, VNLeakyReLU, VNLinear, vn_norm

ATOL = 1e-8
RTOL = 1e-6


def _rot_clouds(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate point clouds: (x @ R.T) per sample. x:(B,N,3), R:(B,3,3)."""
    return torch.einsum("bnd,bed->bne", x, R)


def _rot_vec(V: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate a VN feature on its trailing 3-dim. V:(B,N,C,3), R:(B,3,3)."""
    return torch.einsum("bncd,bed->bnce", V, R)


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    B, N, C = 2, 16, 8
    V = torch.randn(B, N, C, 3, dtype=torch.float64)
    R = random_so3(B).double()  # (B, 3, 3) Haar-uniform
    return V, R


# ── Vector Neuron primitives ──────────────────────────────────────────────────


def test_vn_linear_equivariant():
    V, R = _setup()
    layer = VNLinear(V.size(-2), 5).double()
    lhs = layer(_rot_vec(V, R))
    rhs = _rot_vec(layer(V), R)
    assert torch.allclose(lhs, rhs, atol=ATOL, rtol=RTOL), (lhs - rhs).abs().max()


def test_vn_leaky_relu_equivariant():
    V, R = _setup()
    layer = VNLeakyReLU(V.size(-2)).double()
    lhs = layer(_rot_vec(V, R))
    rhs = _rot_vec(layer(V), R)
    assert torch.allclose(lhs, rhs, atol=ATOL, rtol=RTOL), (lhs - rhs).abs().max()


def test_vn_layernorm_equivariant():
    V, R = _setup()
    layer = VNLayerNorm(V.size(-2)).double()
    lhs = layer(_rot_vec(V, R))
    rhs = _rot_vec(layer(V), R)
    assert torch.allclose(lhs, rhs, atol=ATOL, rtol=RTOL), (lhs - rhs).abs().max()


def test_vn_norm_invariant():
    V, R = _setup()
    lhs = vn_norm(_rot_vec(V, R))
    rhs = vn_norm(V)
    assert torch.allclose(lhs, rhs, atol=ATOL, rtol=RTOL), (lhs - rhs).abs().max()


# ── Unconditional model: the headline equivariance guarantee ───────────────────


def test_unconditional_model_equivariant():
    torch.manual_seed(0)
    B, N = 2, 32
    model = PointCloudVelocity(dim=32, depth=2, heads=4).double().eval()
    x = torch.randn(B, N, 3, dtype=torch.float64)
    t = torch.rand(B, dtype=torch.float64)
    R = random_so3(B).double()
    with torch.no_grad():
        lhs = model(_rot_clouds(x, R), t)      # f(x @ R.T, t)
        rhs = _rot_clouds(model(x, t), R)      # f(x, t) @ R.T
    err = (lhs - rhs).abs().max().item()
    assert torch.allclose(lhs, rhs, atol=1e-7, rtol=1e-5), f"equivariance err={err:.2e}"
    return err


# ── Conditional model: smoke test only (intentionally NOT equivariant) ─────────


def test_conditional_model_smoke():
    torch.manual_seed(0)
    B, N, P = 2, 32, 16
    model = ConditionalPointCloudVelocity(
        dim=32, depth=2, heads=4, image_size=P, patch_size=4
    ).train()
    x = torch.randn(B, N, 3)
    t = torch.rand(B)
    y = torch.randn(B, 1, P, P)
    # Perturb the zero-init readout so gradients are non-trivial.
    with torch.no_grad():
        model.out_proj.weight.add_(0.1 * torch.randn_like(model.out_proj.weight))
    out = model(x, t, y)
    assert out.shape == (B, N, 3), out.shape
    out.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    assert torch.isfinite(out).all(), "non-finite output"


def main() -> int:
    print(f"[equivariance] vec_channels(dim=128, heads=4) = {_vec_channels(128, 4)}")
    checks = [
        ("VNLinear equivariant", test_vn_linear_equivariant),
        ("VNLeakyReLU equivariant", test_vn_leaky_relu_equivariant),
        ("VNLayerNorm equivariant", test_vn_layernorm_equivariant),
        ("vn_norm invariant", test_vn_norm_invariant),
        ("unconditional model equivariant", test_unconditional_model_equivariant),
        ("conditional model smoke", test_conditional_model_smoke),
    ]
    failures = 0
    for name, fn in checks:
        try:
            extra = fn()
            note = f"  (max err {extra:.2e})" if isinstance(extra, float) else ""
            print(f"  PASS  {name}{note}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"[equivariance] {len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
