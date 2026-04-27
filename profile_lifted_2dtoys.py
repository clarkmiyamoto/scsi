#!/usr/bin/env python3
"""
Mirror of lifted_2dtoys.ipynb training step for wall-clock profiling.
Run: python profile_lifted_2dtoys.py --steps 80 --warmup 5
"""
from __future__ import annotations

import argparse
import copy
import time
from contextlib import contextmanager

import torch
import torch.nn as nn


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def timed_region(device: torch.device, accum: dict[str, float], key: str):
    sync_device(device)
    t0 = time.perf_counter()
    yield
    sync_device(device)
    accum[key] = accum.get(key, 0.0) + (time.perf_counter() - t0)


def forward_corruption(x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    return x + noise_std * torch.randn_like(x)


def sample_clean(n: int, kind: str, device: torch.device) -> torch.Tensor:
    if kind == "two_moons":
        n1 = n // 2
        n2 = n - n1
        t1 = torch.rand(n1, device=device) * torch.pi
        t2 = torch.rand(n2, device=device) * torch.pi
        a = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
        b = torch.stack([1 - torch.cos(t2), 1 - torch.sin(t2) - 0.5], dim=1)
        y = torch.cat([a, b], dim=0) + 0.05 * torch.randn(n, 2, device=device)
        return y[torch.randperm(n, device=device)]
    if kind == "checkerboard":
        y = 4 * torch.rand(n, 2, device=device) - 2
        y[:, 1] += ((torch.floor(y[:, 0]) + torch.floor(y[:, 1])) % 2) * 0.5 - 0.25
        return y
    raise ValueError(f"Unknown kind: {kind}")


def sample_corrupted(n: int, kind: str, noise_std: float, device: torch.device) -> torch.Tensor:
    y = sample_clean(n, kind, device)
    return forward_corruption(y, noise_std)


class SimpleMLP(nn.Module):
    def __init__(self, x_dim: int = 2, y_dim: int = 2, hidden: int = 516):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x_t: torch.Tensor, y_cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, y_cond, t], dim=1))


def drift(model_fixed: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return model_fixed(x, y, t)


def flow(
    model_fixed: nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 64,
    t_eps: float = 0.0,
    skip_nan_check: bool = False,
) -> torch.Tensor:
    x = z
    t_final = 1.0 - t_eps
    dt = t_final / n_steps
    for s in range(n_steps):
        t0 = torch.full((z.size(0), 1), s * dt, device=z.device)
        k1 = drift(model_fixed, x, y, t0)
        t_mid = t0 + 0.5 * dt
        k2 = drift(model_fixed, x + 0.5 * dt * k1, y, t_mid)
        t_3q = t0 + 0.75 * dt
        k3 = drift(model_fixed, x + 0.75 * dt * k2, y, t_3q)
        x_new = x + dt * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)
        x = x_new
    if not skip_nan_check and (
        torch.isnan(x_new).any()
        or torch.isinf(x_new).any()
        or torch.isnan(k1).any()
        or torch.isinf(k1).any()
        or torch.isnan(k2).any()
        or torch.isinf(k2).any()
        or torch.isnan(k3).any()
        or torch.isinf(k3).any()
    ):
        print("[Numerical Error @ flow]: nan/inf")
    return x


def one_training_step(
    *,
    device: torch.device,
    model_oracle: nn.Module,
    model_em: nn.Module,
    phi_k: nn.Module,
    opt_oracle: torch.optim.Optimizer,
    opt_em: torch.optim.Optimizer,
    batch_size: int,
    data_kind: str,
    noise_std: float,
    y_fake_ratio: float,
    x0_independent: bool,
    max_grad_norm: float,
    accum: dict[str, float],
    skip_flow_nan_check: bool,
    n_flow_steps: int,
) -> None:
    z = torch.randn(batch_size, 2, device=device)
    t = torch.rand(batch_size, 1, device=device)
    z_prime = torch.randn(batch_size, 2, device=device) if x0_independent else z

    with timed_region(device, accum, "oracle"):
        model_oracle.train()
        opt_oracle.zero_grad()
        x_clean = sample_clean(batch_size, data_kind, device)
        y_oracle = forward_corruption(x_clean, noise_std)
        i_t_or = (1 - t) * z_prime + t * x_clean
        b_target_or = x_clean - z_prime
        b_hat_or = model_oracle(i_t_or, y_oracle, t)
        loss_or = ((b_hat_or - b_target_or) ** 2).mean()
        loss_or.backward()
        torch.nn.utils.clip_grad_norm_(model_oracle.parameters(), max_grad_norm)
        opt_oracle.step()

    with timed_region(device, accum, "flow_e_step"):
        model_em.train()
        opt_em.zero_grad()
        y_real = sample_corrupted(batch_size, data_kind, noise_std, device)
        with torch.no_grad():
            x_em = flow(
                phi_k,
                z,
                y_real,
                n_steps=n_flow_steps,
                skip_nan_check=skip_flow_nan_check,
            )
            y_fake = forward_corruption(x_em, noise_std)

    with timed_region(device, accum, "em_m_step"):
        n_fake = int(batch_size * y_fake_ratio)
        n_fake = max(0, min(n_fake, batch_size))
        if n_fake == 0:
            y_cond = y_real
        elif n_fake == batch_size:
            y_cond = y_fake
        else:
            y_cond = torch.cat((y_fake[:n_fake], y_real[n_fake:]), dim=0)
        i_t_em = (1 - t) * z_prime + t * x_em
        b_target_em = x_em - z_prime
        b_hat_em = model_em(i_t_em, y_cond, t)
        loss_em_val = ((b_hat_em - b_target_em) ** 2).mean()
        loss_em_val.backward()
        torch.nn.utils.clip_grad_norm_(model_em.parameters(), max_grad_norm)
        opt_em.step()


def bench_deepcopy(model: nn.Module, device: torch.device, n: int) -> float:
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(n):
        _ = copy.deepcopy(model)
    sync_device(device)
    return (time.perf_counter() - t0) / n


def run_block(
    *,
    steps: int,
    device: torch.device,
    model_oracle: nn.Module,
    model_em: nn.Module,
    phi_k: nn.Module,
    opt_oracle: torch.optim.Optimizer,
    opt_em: torch.optim.Optimizer,
    batch_size: int,
    data_kind: str,
    noise_std: float,
    y_fake_ratio: float,
    x0_independent: bool,
    max_grad_norm: float,
    skip_nan: bool,
    n_flow_steps: int,
) -> dict[str, float]:
    acc: dict[str, float] = {}
    for _ in range(steps):
        one_training_step(
            device=device,
            model_oracle=model_oracle,
            model_em=model_em,
            phi_k=phi_k,
            opt_oracle=opt_oracle,
            opt_em=opt_em,
            batch_size=batch_size,
            data_kind=data_kind,
            noise_std=noise_std,
            y_fake_ratio=y_fake_ratio,
            x0_independent=x0_independent,
            max_grad_norm=max_grad_norm,
            accum=acc,
            skip_flow_nan_check=skip_nan,
            n_flow_steps=n_flow_steps,
        )
    return acc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=516)
    p.add_argument("--hidden", type=int, default=516)
    p.add_argument("--n-flow-steps", type=int, default=64)
    p.add_argument("--skip-flow-nan-check", action="store_true")
    p.add_argument("--torch-profiler", action="store_true")
    args = p.parse_args()

    device = pick_device()
    print(f"device={device}")

    data_kind = "two_moons"
    noise_std = 0.1
    y_fake_ratio = 0.9
    x0_independent = True
    max_grad_norm = 1.0

    model_em = SimpleMLP(hidden=args.hidden).to(device)
    model_oracle = SimpleMLP(hidden=args.hidden).to(device)
    phi_k = copy.deepcopy(model_em).eval()
    opt_oracle = torch.optim.Adam(model_oracle.parameters(), lr=1e-4)
    opt_em = torch.optim.Adam(model_em.parameters(), lr=1e-4)

    for _ in range(args.warmup):
        _ = run_block(
            steps=1,
            device=device,
            model_oracle=model_oracle,
            model_em=model_em,
            phi_k=phi_k,
            opt_oracle=opt_oracle,
            opt_em=opt_em,
            batch_size=args.batch_size,
            data_kind=data_kind,
            noise_std=noise_std,
            y_fake_ratio=y_fake_ratio,
            x0_independent=x0_independent,
            max_grad_norm=max_grad_norm,
            skip_nan=args.skip_flow_nan_check,
            n_flow_steps=args.n_flow_steps,
        )

    acc = run_block(
        steps=args.steps,
        device=device,
        model_oracle=model_oracle,
        model_em=model_em,
        phi_k=phi_k,
        opt_oracle=opt_oracle,
        opt_em=opt_em,
        batch_size=args.batch_size,
        data_kind=data_kind,
        noise_std=noise_std,
        y_fake_ratio=y_fake_ratio,
        x0_independent=x0_independent,
        max_grad_norm=max_grad_norm,
        skip_nan=args.skip_flow_nan_check,
        n_flow_steps=args.n_flow_steps,
    )
    total = sum(acc.values())
    print(f"\n=== Per-step mean over {args.steps} steps (ms) ===")
    for key in ("oracle", "flow_e_step", "em_m_step"):
        ms = 1000.0 * acc[key] / args.steps
        pct = 100.0 * acc[key] / total if total > 0 else 0.0
        print(f"  {key:14s}  {ms:8.2f} ms/step  ({pct:5.1f}%)")
    print(f"  {'TOTAL':14s}  {1000.0 * total / args.steps:8.2f} ms/step")

    dc_ms = 1000.0 * bench_deepcopy(model_em, device, 20)
    print("\n=== copy.deepcopy(model_em) ===")
    print(f"  {dc_ms:.2f} ms/copy (mean of 20)")

    if args.skip_flow_nan_check:
        acc_with_checks = run_block(
            steps=min(args.steps, 40),
            device=device,
            model_oracle=model_oracle,
            model_em=model_em,
            phi_k=phi_k,
            opt_oracle=opt_oracle,
            opt_em=opt_em,
            batch_size=args.batch_size,
            data_kind=data_kind,
            noise_std=noise_std,
            y_fake_ratio=y_fake_ratio,
            x0_independent=x0_independent,
            max_grad_norm=max_grad_norm,
            skip_nan=False,
            n_flow_steps=args.n_flow_steps,
        )
        total_with_checks = sum(acc_with_checks.values())
        print("\n=== NaN check impact (TOTAL ms/step) ===")
        print(f"  skip_nan=True:  {1000.0 * total / args.steps:.2f}")
        print(f"  skip_nan=False: {1000.0 * total_with_checks / min(args.steps, 40):.2f}")

    for ns in (32, 16):
        for _ in range(3):
            _ = run_block(
                steps=1,
                device=device,
                model_oracle=model_oracle,
                model_em=model_em,
                phi_k=phi_k,
                opt_oracle=opt_oracle,
                opt_em=opt_em,
                batch_size=args.batch_size,
                data_kind=data_kind,
                noise_std=noise_std,
                y_fake_ratio=y_fake_ratio,
                x0_independent=x0_independent,
                max_grad_norm=max_grad_norm,
                skip_nan=True,
                n_flow_steps=ns,
            )
        a = run_block(
            steps=20,
            device=device,
            model_oracle=model_oracle,
            model_em=model_em,
            phi_k=phi_k,
            opt_oracle=opt_oracle,
            opt_em=opt_em,
            batch_size=args.batch_size,
            data_kind=data_kind,
            noise_std=noise_std,
            y_fake_ratio=y_fake_ratio,
            x0_independent=x0_independent,
            max_grad_norm=max_grad_norm,
            skip_nan=True,
            n_flow_steps=ns,
        )
        print(f"\n  n_flow_steps={ns}: TOTAL ~ {1000.0 * sum(a.values()) / 20:.2f} ms/step")

    if args.torch_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
            with torch.profiler.record_function("train_chunk"):
                _ = run_block(
                    steps=5,
                    device=device,
                    model_oracle=model_oracle,
                    model_em=model_em,
                    phi_k=phi_k,
                    opt_oracle=opt_oracle,
                    opt_em=opt_em,
                    batch_size=args.batch_size,
                    data_kind=data_kind,
                    noise_std=noise_std,
                    y_fake_ratio=y_fake_ratio,
                    x0_independent=x0_independent,
                    max_grad_norm=max_grad_norm,
                    skip_nan=True,
                    n_flow_steps=args.n_flow_steps,
                )
        print("\n=== torch.profiler top CPU ops (5 steps) ===")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
