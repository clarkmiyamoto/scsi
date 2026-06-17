"""Self-Consistent Stochastic Interpolant (SCSI) core.

Minimal port of ``src/interpolant_utils.py::SCSInterpolant`` reduced to the
essential machinery: a deterministic Euler ODE transport and the velocity loss.
Stripped of every optional axis (``gamma_scale=0`` so pure ODE, no score
network, no Heun, no canonicalization, ``n_transports=1``).

Conventions (must match the forward map in ``forward.py``):
  * ``obs`` is the forward-model prior ``z_out`` (Gaussian, the ``t=1`` endpoint).
  * ``cond`` is the tilt-series conditioning the drift network reads.
  * ``transport`` integrates the learned drift backward in time, noise -> clean.
  * ``push_fwd(x0)`` re-corrupts a pseudo-clean estimate back into ``(z, cond)``.
"""
import torch


class SCSInterpolant(torch.nn.Module):

    def __init__(self, push_fwd, n_steps: int = 80, alpha: float = 1.0):
        super().__init__()
        self.push_fwd = push_fwd
        self.n_steps = n_steps
        self.delta_t = 1.0 / n_steps
        self.alpha = alpha

    @torch.no_grad()
    def transport(self, b, x, cond=None):
        """Integrate ``X' = -b(X, t, cond)`` from ``t=1`` (the prior ``x``) down to
        ``t~0`` via explicit Euler, returning the pseudo-clean estimate ``x0``."""
        X = x * 1.0
        for i in range(1, self.n_steps + 1):
            t = torch.ones(x.shape[0], device=x.device) - (i - 1) * self.delta_t
            X = X - b(X, t, cond) * self.delta_t
        return X

    def loss_fn(self, b, obs, cond=None, x0=None, b_fixed=None):
        """Stochastic-interpolant velocity loss.

        If ``x0`` is given (warmup phase) it is used directly as the pseudo-clean
        target. Otherwise (bootstrap phase) ``x0`` is produced by transporting
        ``obs`` through the frozen model ``b_fixed`` (or the live ``b``).

        The target is re-corrupted to ``x1`` via ``push_fwd`` and mixed with the
        raw observation by a per-sample Bernoulli(``alpha``) mask; the network
        then regresses the interpolant velocity ``v = x1 - x0``.
        """
        if x0 is None:
            x0 = self.transport(b_fixed if b_fixed is not None else b, obs, cond)

        x1, cond1 = self.push_fwd(x0, return_latents=True)

        batch = obs.shape[0]
        raw_mask = torch.bernoulli(torch.full((batch,), self.alpha, device=obs.device))
        m = raw_mask.view(batch, *([1] * (obs.ndim - 1)))
        x1 = x1 * m + obs * (1 - m)
        if cond is not None:
            mc = raw_mask.view(batch, *([1] * (cond1.ndim - 1)))
            cond1 = cond1 * mc + cond * (1 - mc)

        t_flat = torch.rand(batch, device=obs.device)
        t = t_flat.view(batch, *([1] * (obs.ndim - 1)))
        It = (1 - t) * x0 + t * x1
        v_true = x1 - x0
        vt = b(It, t_flat, cond1)
        return torch.mean((vt - v_true) ** 2)
