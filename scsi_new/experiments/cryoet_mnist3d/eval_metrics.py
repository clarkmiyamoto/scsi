"""
Scalar reconstruction quality for the EM loop, so runs can be ranked by a number instead of by
eyeballing panels (sbatch/lr_schedule_ablation/).

x_hat lives in whatever frame EM locked onto -- the channel's SO(3) mount is unobserved, so the
model's frame is only defined up to a global rotation, and early in EM not even consistently
across samples. So the metric is orientation-free: for each eval item b,

    corr_own[b]   = max_{R in SO(3)} pearson(R . x_hat[b], x_gt[b])
    corr_other[b] = max_{R in SO(3)} pearson(R . x_hat[b], x_gt[same[b]])
    corr_gap      = mean_b (corr_own[b] - corr_other[b])

where same[b] is the nearest earlier pool item (cyclically) of the SAME digit class -- on a
single-class pool, b-1, the original pairing. corr_own is shape quality. corr_gap is how much
of x_hat is specific to ITS observation: any y-independent output (a single template, the
collapse mode in the long-LR runs) scores gap == 0 exactly, since same[] is a permutation within
each class. Pool items are similar digits, so even a perfect recon has corr_other well above
0 -- calibrate() reports the perfect-recon values (GT vs a randomly rotated copy of itself) to
read both numbers against.

Multi-class pools (build_viz_pool's `label`) add the same scores against a DIFFERENT-class
partner (the nearest earlier item of another class): corr_other_xcls / corr_gap_xcls. The two
gaps separate the collapse modes a mixed pool allows: a per-class template (right digit, not
this instance) has corr_gap ~ 0 but corr_gap_xcls > 0; a single template has both ~ 0.
evaluate() also reports corr_own / corr_gap per class (by_class/<digit>/...), to spot digits
that EM never resolves; with a few items per class those are noisy.

The rotation alignment is applied only inside the metric. Panels still show the raw x_hat.

Search: a fixed Haar-random coarse set (shared across calls, so every EM step and every run is
scored against the same candidates), then local refinement around each item's best candidate at
shrinking angular scales. Every random draw comes from a private generator: the metric never
touches the global RNG, so turning it on does not change the training trajectory.
"""

import math

import torch

from ode import euler_integration
from rotation import quaternion_to_matrix, rotate_3d, sample_uniform_rotation_so3


def _unit(v: torch.Tensor) -> torch.Tensor:
    # (..., N) -> mean-centred, unit-norm rows: a dot product of two of these is Pearson r.
    v = v - v.mean(dim=-1, keepdim=True)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _small_rotations(n: int, max_deg: float, generator: torch.Generator,
                     device: torch.device) -> torch.Tensor:
    # (n, 3, 3): uniform random axis, angle uniform in [0, max_deg].
    axis = torch.randn(n, 3, generator=generator, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half = torch.rand(n, generator=generator, device=device) * math.radians(max_deg) / 2
    q = torch.cat([half.cos()[:, None], half.sin()[:, None] * axis], dim=-1)
    return quaternion_to_matrix(q)


class AlignedCorrelation:
    """
    Callable (x, refs) -> (len(refs), B) tensor of max over SO(3) of pearson(R . x[b], ref[b]).

    Args:
        n_coarse: size of the fixed Haar-random coarse set. 8192 puts the median
            nearest candidate within ~7 degrees of any rotation.
        refine_deg: angular scales of the local refinement rounds, coarsest first.
        n_refine: candidates per refinement round (the current best is always kept).
        chunk: rotated volumes per grid_sample call, which bounds peak memory.
    """

    def __init__(self, device: torch.device, n_coarse: int = 8192,
                 refine_deg: tuple[float, ...] = (8.0, 4.0, 2.0, 1.0), n_refine: int = 64,
                 chunk: int = 2048, seed: int = 0):
        self.device = device
        self.refine_deg = refine_deg
        self.n_refine = n_refine
        self.chunk = chunk
        self.seed = seed
        g = torch.Generator(device=device).manual_seed(seed)
        self.coarse = sample_uniform_rotation_so3(n_coarse, device=device, generator=g)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, refs: list[torch.Tensor]) -> torch.Tensor:
        x = x.to(self.device)
        B, V3 = x.size(0), x[0].numel()
        ref_u = torch.stack([_unit(r.to(self.device).reshape(B, V3)) for r in refs])  # (J, B, V3)
        J = ref_u.size(0)

        # Coarse: every item against every coarse rotation. Each rotated x is scored against all
        # J refs at once, so the refs share one grid_sample pass.
        K = self.coarse.size(0)
        best = torch.full((J, B), -float("inf"), device=self.device)
        best_R = self.coarse[:1].expand(J, B, 3, 3).clone()
        k_step = max(1, self.chunk // B)
        for k0 in range(0, K, k_step):
            Rk = self.coarse[k0:k0 + k_step]                                  # (k, 3, 3)
            k = Rk.size(0)
            xr = rotate_3d(x.repeat_interleave(k, dim=0), Rk.repeat(B, 1, 1))  # (B*k, 1, V, V, V)
            corr = torch.einsum("bkv,jbv->jbk", _unit(xr.reshape(B, k, V3)), ref_u)
            val, idx = corr.max(dim=-1)                                        # (J, B)
            better = val > best
            best = torch.where(better, val, best)
            best_R = torch.where(better[..., None, None], Rk[idx], best_R)

        # Refine: perturb each (ref, item)'s own best rotation at shrinking scales. The generator
        # is reseeded per call so a given x always gets the same candidates.
        g = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        x_rep = x.unsqueeze(0).expand(J, -1, -1, -1, -1, -1)                  # (J, B, 1, V, V, V)
        for deg in self.refine_deg:
            P = torch.cat([torch.eye(3, device=self.device)[None],
                           _small_rotations(self.n_refine, deg, g, self.device)])  # (n+1, 3, 3)
            n = P.size(0)
            cand = best_R[:, :, None] @ P                                      # (J, B, n, 3, 3)
            vals = torch.empty(J, B, n, device=self.device)
            flat_x = x_rep.reshape(J * B, *x.shape[1:])
            flat_R = cand.reshape(J * B, n, 3, 3)
            flat_ref = ref_u.reshape(J * B, V3)
            m_step = max(1, self.chunk // n)
            for m0 in range(0, J * B, m_step):
                xm, Rm = flat_x[m0:m0 + m_step], flat_R[m0:m0 + m_step]
                m = xm.size(0)
                xr = rotate_3d(xm.repeat_interleave(n, dim=0), Rm.reshape(m * n, 3, 3))
                c = (_unit(xr.reshape(m, n, V3)) * flat_ref[m0:m0 + m, None]).sum(-1)
                vals.view(J * B, n)[m0:m0 + m_step] = c
            val, idx = vals.max(dim=-1)                                        # (J, B)
            best = val
            best_R = torch.gather(cand, 2, idx[..., None, None, None].expand(J, B, 1, 3, 3))[:, :, 0]
        return best


def partners(label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    For each pool item b, the nearest earlier item (cyclically) of the same class, and of a
    different class (None when the pool has one class). On a single-class pool the first is
    b-1, i.e. x_gt.roll(1). Raises if a class has one item: its only partner would be itself.
    """
    lab = label.tolist()
    B = len(lab)
    same, xcls = [], []
    for b in range(B):
        prev = [(b - s) % B for s in range(1, B)]
        s = next((j for j in prev if lab[j] == lab[b]), None)
        if s is None:
            raise ValueError(f"eval pool has a single item of class {lab[b]}; raise --eval_n so "
                             f"every class gets at least 2")
        same.append(s)
        xcls.append(next((j for j in prev if lab[j] != lab[b]), None))
    xcls_idx = None if xcls[0] is None else torch.tensor(xcls)
    return torch.tensor(same), xcls_idx


def _score(x: torch.Tensor, pool: dict, align: AlignedCorrelation, device: torch.device,
           by_class: bool) -> dict[str, float]:
    x_gt = pool["x_gt"].to(device)
    label = pool.get("label", torch.zeros(x_gt.size(0), dtype=torch.long))
    same, xcls = partners(label)
    refs = [x_gt, x_gt[same.to(device)]]
    if xcls is not None:
        refs.append(x_gt[xcls.to(device)])
    corr = align(x, refs)
    own, other = corr[0], corr[1]
    out = {"corr_own": own.mean().item(), "corr_other": other.mean().item(),
           "corr_gap": (own - other).mean().item()}
    if xcls is not None:
        out.update({"corr_other_xcls": corr[2].mean().item(),
                    "corr_gap_xcls": (own - corr[2]).mean().item()})
        if by_class:
            label = label.to(device)
            for c in label.unique().tolist():
                m = label == c
                out[f"by_class/{c}/corr_own"] = own[m].mean().item()
                out[f"by_class/{c}/corr_gap"] = (own - other)[m].mean().item()
    return out


@torch.no_grad()
def evaluate(model, pool: dict, n_steps_sampling: int, align: AlignedCorrelation,
             device: torch.device, batch_size: int = 32) -> dict[str, float]:
    """
    Sample x_hat for the fixed eval pool (fixed x0 and y, so the score is deterministic given the
    weights) and score it. Returns {corr_own, corr_other, corr_gap}, each a mean over the pool,
    plus corr_other_xcls / corr_gap_xcls and by_class/<digit>/{corr_own,corr_gap} for a
    multi-class pool. Integrates batch_size items at a time, so a large pool never pushes a
    bigger batch through the network than the E-step does.
    """
    x0, y = pool["x0"], pool["y"]
    x_hat = torch.cat([euler_integration(model, x0[i:i + batch_size].to(device),
                                         y[i:i + batch_size].to(device), n_steps_sampling)
                       for i in range(0, x0.size(0), batch_size)])
    return _score(x_hat, pool, align, device, by_class=True)


@torch.no_grad()
def calibrate(pool: dict, align: AlignedCorrelation, device: torch.device) -> dict[str, float]:
    """
    Metric values for a PERFECT reconstruction in an unknown frame: each GT volume under a
    random rotation, scored like x_hat. corr_own here is the ceiling set by search error and
    trilinear interpolation. corr_gap (and corr_gap_xcls) here is the largest gap a perfect
    recon reaches, given how alike the eval digits are.
    """
    x_gt = pool["x_gt"].to(device)
    g = torch.Generator(device=device).manual_seed(align.seed + 2)
    R = sample_uniform_rotation_so3(x_gt.size(0), device=device, generator=g)
    return _score(rotate_3d(x_gt, R), pool, align, device, by_class=False)
