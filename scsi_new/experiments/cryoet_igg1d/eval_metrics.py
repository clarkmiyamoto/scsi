"""
Scalar reconstruction quality for the IgG-1D EM loop, scored against GT the EM loop never sees.

For each held-out pool image, x_hat is sampled from the model (fixed x0, Euler) and compared with
that image's image-frame GT target (data.build_eval_pool). The names follow the supervised cryofm
run's sample/* metrics (cryofm IgG1DImageCond.evaluate_samples), so the two can be read side by
side:

    corr_target        Pearson r(x_hat, target)
    conf_err_deg       circular error of the best-matching conformation: x_hat is correlated with
                       all 100 conformations posed by the image's GT pose, argmax, times 3.6 deg
    conf_acc           fraction with the exact conformation; conf_acc_within_3 within +-3
    pred_conf_std      std of the predicted conformation index (small = collapse)
    fsc0.5_res_A       FSC=0.5 resolution vs the target (A), cryofm's shells and interpolation

A single image cannot tell x from its mirror image through the z = D//2 plane (the projection is
the same), so an unsupervised model is free to learn either hand. corr_target is the as-is value;
corr_target_hand takes the better hand per item, hand_flip_frac is how often that is the mirror,
and the conformation and FSC scores use the better hand. calibrate() scores the GT targets
themselves: the ceilings, for eval_calib/*.

Nothing here touches the global RNG: x0 is fixed in the pool and scoring is deterministic.
"""

import torch

from ode import euler_integration
from rotation import mirror_z, pose_volumes


def _unit(v: torch.Tensor) -> torch.Tensor:
    # (B, ...) -> (B, N) mean-centred, unit-norm rows: a dot product of two of these is Pearson r.
    v = v.flatten(1).float()
    v = v - v.mean(dim=1, keepdim=True)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)


def fsc_resolution(a: torch.Tensor, b: torch.Tensor, apix: float, threshold: float = 0.5) -> torch.Tensor:
    """
    FSC-threshold resolution (A) of each pair of (B, 1, D, D, D) volumes. A torch port of cryofm's
    calc_fsc + fsc_to_resolution (core/utils/metrics.py): D//2 shells up to Nyquist, zero
    frequency skipped, linear interpolation at the crossing, the last shell if it never crosses.
    """
    D = a.size(-1)
    fa, fb = torch.fft.fftn(a[:, 0].double(), dim=(-3, -2, -1)), torch.fft.fftn(b[:, 0].double(), dim=(-3, -2, -1))
    k = torch.fft.fftfreq(D, dtype=torch.float64, device=a.device)
    kz, ky, kx = torch.meshgrid(k, k, k, indexing="ij")
    radii = torch.sqrt(kz**2 + ky**2 + kx**2)
    shell_radii = torch.linspace(0, k.abs().max().item(), D // 2 + 1, dtype=torch.float64, device=a.device)
    labels = torch.searchsorted(shell_radii, radii.flatten(), right=False)   # side="left"
    n_labels = len(shell_radii) + 1

    def shell_sum(v):  # (B, D^3) -> (B, D//2) sums over shells 1 .. D//2
        out = torch.zeros(v.size(0), n_labels, dtype=v.dtype, device=v.device)
        return out.index_add_(1, labels, v)[:, 1:len(shell_radii)]

    num = shell_sum((fa * fb.conj()).real.flatten(1))
    den = torch.sqrt(shell_sum(fa.abs().pow(2).flatten(1)) * shell_sum(fb.abs().pow(2).flatten(1)))
    fsc = torch.where(den > 1e-5, num / den.clamp_min(1e-30), torch.zeros_like(num))
    freq = shell_radii[1:]

    res = []
    for f in fsc:
        below = torch.nonzero(f <= threshold)
        i = below[0, 0].item() if len(below) else len(f) - 1
        fr = freq[i]
        if i > 0 and f[i - 1] > threshold > f[i]:
            frac = (threshold - f[i]) / (f[i - 1] - f[i])
            fr = freq[i] * (1.0 - frac) + freq[i - 1] * frac
        res.append(apix / fr)
    return torch.stack(res).float()


@torch.no_grad()
def conformation_scores(x: torch.Tensor, pool: dict, device: torch.device) -> torch.Tensor:
    """(n, C) Pearson r of each x[k] with every conformation posed by item k's GT pose and shift."""
    gt = pool["gt_volumes"].to(device)[:, None]                               # (C, 1, D, D, D)
    C, u = gt.size(0), _unit(x.to(device))
    scores = []
    for k in range(x.size(0)):
        refs = pose_volumes(gt, pool["rot"][k].expand(C, 3, 3), pool["shift"][k].expand(C, 2))
        scores.append(_unit(refs) @ u[k])
    return torch.stack(scores)


@torch.no_grad()
def score(x: torch.Tensor, pool: dict, device: torch.device, apix: float) -> dict[str, float]:
    x = x.to(device)
    x_gt = pool["x_gt"].to(device)
    label = pool["label"].to(device)
    C = pool["gt_volumes"].size(0)
    hands = torch.stack([x, mirror_z(x)])                                      # (2, n, 1, D, D, D)

    corr = torch.stack([(_unit(h) * _unit(x_gt)).sum(1) for h in hands])      # (2, n)
    best = corr.argmax(0)
    conf = torch.stack([conformation_scores(h, pool, device) for h in hands])  # (2, n, C)
    pred = conf.amax(0).argmax(-1)
    d = (pred - label).abs()
    err = torch.minimum(d, C - d).float()
    x_best = torch.where(best[:, None, None, None, None] == 1, hands[1], hands[0])
    return {
        "corr_target": corr[0].mean().item(),
        "corr_target_hand": corr.amax(0).mean().item(),
        "hand_flip_frac": (best == 1).float().mean().item(),
        "conf_err_deg": (err.mean() * 360.0 / C).item(),
        "conf_acc": (err == 0).float().mean().item(),
        "conf_acc_within_3": (err <= 3).float().mean().item(),
        "pred_conf_std": pred.float().std().item(),
        "fsc0.5_res_A": fsc_resolution(x_best, x_gt, apix).mean().item(),
    }


@torch.no_grad()
def sample(model, pool: dict, n_steps_sampling: int, device: torch.device,
           batch_size: int = 32) -> torch.Tensor:
    """x_hat for every pool item, integrated batch_size at a time (no bigger than the E-step's)."""
    x0, y = pool["x0"], pool["y"]
    return torch.cat([euler_integration(model, x0[i:i + batch_size].to(device),
                                        y[i:i + batch_size].to(device), n_steps_sampling)
                      for i in range(0, x0.size(0), batch_size)])


def evaluate(model, pool: dict, n_steps_sampling: int, device: torch.device, apix: float,
             batch_size: int = 32) -> dict[str, float]:
    return score(sample(model, pool, n_steps_sampling, device, batch_size), pool, device, apix)


def calibrate(pool: dict, device: torch.device, apix: float) -> dict[str, float]:
    """Metric values for the GT targets themselves (corr_target 1, FSC at Nyquist): the ceiling on
    the conformation scores, whose references are posed at this resolution and so slightly
    smoother than the targets (posed at 128 px, then cropped)."""
    return score(pool["x_gt"], pool, device, apix)
