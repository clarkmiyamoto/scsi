from abc import ABC, abstractmethod
from typing import Callable
import math
import torch
from torch import Tensor
from interpolant import Interpolant


class ODE:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def drift(self, t: Tensor, x: Tensor, conditional: Tensor = None) -> Tensor:
        '''
        Drift term of the ODE. dX/dt = drift(t, X)
        '''
        return self.model(x, t, conditional=conditional)

class SimulatorODE(ABC):
    def __init__(self, ode: ODE):
        self.ode = ode

    @abstractmethod
    def forwards(self, t: Tensor, x: Tensor, dt: float) -> Tensor:
        pass
    
    @abstractmethod
    def backwards(self, t: Tensor, x: Tensor, dt: float, conditional: Tensor = None) -> Tensor:
        pass
    
    def solve_forwards(self, x: Tensor, schedule: Tensor) -> Tensor:
        '''
        Solve the ODE forward along *schedule*.

        Args:
            x: Initial condition. Shape (batch, dim).
            schedule: 1-D Tensor of monotonically increasing time points
                      (length N+1).  Use :func:`uniform_schedule` or
                      :func:`exponential_schedule` to generate one.
        Returns:
            x: Solution at the final time point. Shape (batch, dim).
        '''
        schedule = schedule.to(x.device)
        batch_size = x.shape[0]
        for k in range(len(schedule) - 1):
            t_k = schedule[k].item()
            dt = (schedule[k + 1] - schedule[k]).item()
            t_batch = torch.full((batch_size,), t_k, device=x.device)
            x = self.forwards(t_batch, x, dt)
        return x

    def solve_forwards_trajectory(self, x: Tensor, schedule: Tensor) -> list[Tensor]:
        '''Solve forward along *schedule* and return trajectory at each step.'''
        schedule = schedule.to(x.device)
        trajectory = [x.clone()]
        for k in range(len(schedule) - 1):
            t_k = schedule[k].item()
            dt = (schedule[k + 1] - schedule[k]).item()
            t_batch = torch.full((x.shape[0],), t_k, device=x.device)
            x = self.forwards(t_batch, x, dt)
            trajectory.append(x.clone())
        return trajectory

    def solve_backwards(self, x: Tensor, schedule: Tensor, conditional: Tensor = None) -> Tensor:
        '''Integrate the ODE backward along *schedule* (from last to first time point).

        Args:
            x: Initial condition at ``schedule[-1]``. Shape (batch, dim).
            schedule: 1-D Tensor of monotonically increasing time points.
        '''
        schedule = schedule.to(x.device)
        for k in range(len(schedule) - 1, 0, -1):
            t_k = schedule[k].item()
            dt = (schedule[k] - schedule[k - 1]).item()
            t_batch = torch.full((x.shape[0],), t_k, device=x.device)
            x = self.backwards(t_batch, x, dt, conditional=conditional)
        return x

    def solve_backwards_trajectory(self, x: Tensor, schedule: Tensor) -> list[Tensor]:
        '''Solve backward along *schedule* and return trajectory at each step.'''
        schedule = schedule.to(x.device)
        trajectory = [x.clone()]
        for k in range(len(schedule) - 1, 0, -1):
            t_k = schedule[k].item()
            dt = (schedule[k] - schedule[k - 1]).item()
            t_batch = torch.full((x.shape[0],), t_k, device=x.device)
            x = self.backwards(t_batch, x, dt)
            trajectory.append(x.clone())
        return trajectory

class EulerSimulator(SimulatorODE):
    def forwards(self, t: Tensor, x: Tensor, dt: float) -> Tensor:
        return x + dt * self.ode.drift(t, x)

    def backwards(self, t: Tensor, x: Tensor, dt: float, conditional: Tensor = None) -> Tensor:
        # ODE is dX/dt = drift(t,X). Backward step: from (t, x) to (t-dt, x_prev)
        # so x_prev = x - drift(t, x)*dt (explicit Euler in reverse time).
        return x - dt * self.ode.drift(t, x, conditional=conditional)

class SDE(ABC):
    def __init__(self, 
                 drift: torch.nn.Module, 
                 denoiser: torch.nn.Module,
                 interpolant: Interpolant,
                 noise_schedule: Callable,
                 tolerance: float = 1e-7):
        self.drift_model = drift
        self.denoiser_model = denoiser
        self.interpolant = interpolant
        self.noise_schedule = noise_schedule
        self.tolerance = tolerance
    
    def drift_forward(self, t: Tensor, x: Tensor) -> Tensor:
        noise = self.noise_schedule(t).unsqueeze(-1)   # (batch,) -> (batch, 1)
        gamma = self.interpolant.gamma(t).unsqueeze(-1) # (batch,) -> (batch, 1)
        return self.drift_model(x, t) - noise * self.denoiser_model(x, t) / (gamma + self.tolerance)
    
    def drift_backward(self, t: Tensor, x: Tensor) -> Tensor:
        noise = self.noise_schedule(t).unsqueeze(-1)   # (batch,) -> (batch, 1)
        gamma = self.interpolant.gamma(t).unsqueeze(-1) # (batch,) -> (batch, 1)
        return self.drift_model(x, t) + noise * self.denoiser_model(x, t) / (gamma + self.tolerance)
    
    def diffusion(self, t: Tensor, x: Tensor) -> Tensor:
        return torch.sqrt(2 * self.noise_schedule(t)).unsqueeze(-1)  # (batch,) -> (batch, 1)


class SimulatorSDE(ABC):

    def __init__(self, 
                 sde: SDE,):
        self.sde = sde

    @abstractmethod
    def forwards(self, t: Tensor, x: Tensor, dt: float) -> Tensor:
        pass

    @abstractmethod
    def backwards(self, t: Tensor, x: Tensor, dt: float, conditional: Tensor = None) -> Tensor:
        pass

    def solve_forwards(self, x: Tensor, schedule: Tensor) -> Tensor:
        '''
        Solve the ODE forward along *schedule*.

        Args:
            x: Initial condition. Shape (batch, dim).
            schedule: 1-D Tensor of monotonically increasing time points
                      (length N+1).  Use :func:`uniform_schedule` or
                      :func:`exponential_schedule` to generate one.
        Returns:
            x: Solution at the final time point. Shape (batch, dim).
        '''
        schedule = schedule.to(x.device)
        batch_size = x.shape[0]
        for k in range(len(schedule) - 1):
            t_k = schedule[k].item()
            dt = (schedule[k + 1] - schedule[k]).item()
            t_batch = torch.full((batch_size,), t_k, device=x.device)
            x = self.forwards(t_batch, x, dt)
        return x

    def solve_forwards_trajectory(self, x: Tensor, schedule: Tensor) -> list[Tensor]:
        '''Solve forward along *schedule* and return trajectory at each step.'''
        schedule = schedule.to(x.device)
        trajectory = [x.clone()]
        for k in range(len(schedule) - 1):
            t_k = schedule[k].item()
            dt = (schedule[k + 1] - schedule[k]).item()
            t_batch = torch.full((x.shape[0],), t_k, device=x.device)
            x = self.forwards(t_batch, x, dt)
            trajectory.append(x.clone())
        return trajectory

    def solve_backwards(self, x: Tensor, schedule: Tensor, conditional: Tensor = None) -> Tensor:
        '''Integrate the SDE backward along *schedule* (from last to first time point).

        Args:
            x: Initial condition at ``schedule[-1]``. Shape (batch, dim).
            schedule: 1-D Tensor of monotonically increasing time points.
        '''
        schedule = schedule.to(x.device)
        for k in range(len(schedule) - 1, 0, -1):
            t_k = schedule[k].item()
            dt = (schedule[k] - schedule[k - 1]).item()
            t_batch = torch.full((x.shape[0],), t_k, device=x.device)
            x = self.backwards(t_batch, x, dt, conditional=conditional)
        return x


class EulerMaruyamaSimulator(SimulatorSDE):
    '''Euler-Maruyama discretization of the SDE.

    Forward SDE  (data -> observations, t: 0 -> 1):
        dX = drift_forward(t,X) dt + sqrt(2 epsilon) dW

    Reverse SDE  (observations -> data, t: 1 -> 0, Eq. 2 in the paper):
        dX^B = drift_backward(t,X) dt + sqrt(2 epsilon) dW^B

    When discretising the reverse SDE backward in time (from t_k to t_{k-1}
    with positive dt = t_k - t_{k-1}), the Euler-Maruyama step is:

        X_{k-1} = X_k  -  drift_backward * dt  +  sqrt(2 epsilon dt) * z

    The sign follows the same convention as the ODE backward step
    (x - b*dt).
    '''

    def __init__(self, sde: SDE):
        super().__init__(sde)

    def forwards(self, t: Tensor, x: Tensor, dt: float) -> Tensor:
        z = torch.randn_like(x, device=x.device)
        return x + self.sde.drift_forward(t, x) * dt + self.sde.diffusion(t, x) * (dt ** 0.5) * z

    def backwards(self, t: Tensor, x: Tensor, dt: float) -> Tensor:
        z = torch.randn_like(x, device=x.device)
        return x - self.sde.drift_backward(t, x) * dt + self.sde.diffusion(t, x) * (dt ** 0.5) * z


# ---------------------------------------------------------------------------
# Schedule helper
# ---------------------------------------------------------------------------

def _build_schedule(schedule_type: str, num_steps: int, t0: float, tN: float) -> Tensor:
    """Return a time-point schedule tensor from a human-readable type string."""
    if schedule_type == "uniform":
        return uniform_schedule(num_steps, t0, tN)
    elif schedule_type == "exponential":
        return exponential_schedule(num_steps, t0, tN)
    else:
        raise ValueError(
            f"Unknown schedule_type '{schedule_type}'. "
            "Choose 'uniform' or 'exponential'."
        )


# ---------------------------------------------------------------------------
# Time discretization schedulers
# ---------------------------------------------------------------------------

def uniform_schedule(num_steps: int, t0: float = 0.0, tN: float = 1.0) -> Tensor:
    """Generate a uniform (evenly spaced) time schedule.

    Args:
        num_steps: Number of integration steps N.
        t0: Start time.
        tN: End time.

    Returns:
        1-D Tensor of length ``num_steps + 1`` with evenly spaced time
        points from *t0* to *tN* (inclusive).
    """
    return torch.linspace(t0, tN, num_steps + 1)


def exponential_schedule(num_steps: int, t0: float = 1e-3, tN: float = 0.999) -> Tensor:
    """Exponentially decaying time schedule (Liu et al., 2025, Section 5).

    Step sizes are smallest near t=0 and t=1 (where gamma -> 0) and largest
    around the midpoint t=0.5, cancelling the gamma-dependent error terms in
    Theorem 4.3.  The schedule is defined as::

        t_k = (1/2)(1-h)^{M-k}       for k < M
        t_k = 1 - (1/2)(1-h)^{k-M}   for k >= M

    Given *num_steps* total steps and endpoints (t0, tN), the parameters are
    derived as:

        r = (1-h) = (4 * t0 * (1-tN))^{1/N}
        M = round(N * log(2*t0) / log(4*t0*(1-tN)))

    Args:
        num_steps: Total number of integration steps N.
        t0: Start time (> 0, avoids gamma(0)=0 singularity).
        tN: End time (< 1, avoids gamma(1)=0 singularity).

    Returns:
        1-D Tensor of length ``num_steps + 1`` with the exponential schedule
        time points from *t0* to *tN* (inclusive).
    """
    if t0 <= 0 or tN >= 1:
        raise ValueError(f"Require 0 < t0 < tN < 1, got t0={t0}, tN={tN}")

    N = num_steps
    log_arg = 4.0 * t0 * (1.0 - tN)
    r = log_arg ** (1.0 / N)           # r = 1 - h
    log_r = math.log(r)

    # Midpoint index M (number of steps in the first half)
    M = round(math.log(2.0 * t0) / log_r)
    M = max(1, min(N - 1, M))          # clamp to [1, N-1]

    ts = []
    for k in range(N + 1):
        if k < M:
            t_k = 0.5 * r ** (M - k)
        elif k == M:
            t_k = 0.5
        else:
            t_k = 1.0 - 0.5 * r ** (k - M)
        ts.append(t_k)

    # Ensure exact boundary values
    ts[0] = t0
    ts[-1] = tN

    return torch.tensor(ts, dtype=torch.float32)

AVAILABLE_SCHEDULES = ["uniform", "exponential"]

def select_schedule(schedule_type: str) -> Callable:
    if schedule_type == "uniform":
        return uniform_schedule
    elif schedule_type == "exponential":
        return exponential_schedule
    else:
        raise ValueError(f"Unknown schedule_type '{schedule_type}'. Choose from: {AVAILABLE_SCHEDULES}.")