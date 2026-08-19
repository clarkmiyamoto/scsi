import copy

import torch
import wandb
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader, TensorDataset
from si import loss_ConditionalDrift
from ode import euler_integration
from distribution import Distribution
from dataclasses import dataclass, field

"""
Implements SCSI algorithm for solving inverse problems.

Pseudocode
1. E-Step
    - From previous iteration of model, sample posterior p(x|y)
    - Construct dataset of (x_hat, y_hat) pairs for M-step
2. M-Step
    - Update model parameters using the dataset of (x_hat, y_hat) pairs from E-step
"""

@dataclass
class Config_SCSI_EStep:
    num_samples: int = 10_000
    method: str = "euler" # Integration method for ODE (only "euler" is implemented)
    n_steps_sampling: int = 64 # Number of steps for ODE integration during sampling
    batch_size: int = 1024 # Batch size for sampling from the posterior

@dataclass
class Config_SCSI_MStep:
    interpolant_style: str = "gvp" # Style of interpolant for the model ("linear" or "gvp")
    n_steps_train: int = 5_000 # Number of steps for training the model
    batch_size: int = 258 # Batch size for training
    lr: float = 1e-4 # Learning rate
    weight_decay: float = 0.0 # Weight decay for optimizer
    ema: float = 0.999 # Exponential moving average for model parameters

@dataclass
class Config_SCSI:
    num_scsi_steps: int = 40
    estep: Config_SCSI_EStep = field(default_factory=Config_SCSI_EStep)
    mstep: Config_SCSI_MStep = field(default_factory=Config_SCSI_MStep)
    device: str = "cuda"
    seed: int = 42
    lr_eta_min: float = 0.0 # Floor LR for the global cosine schedule spanning warmup + all SCSI steps


def estep(model, 
          base_dist: Distribution, 
          observations: Dataset, 
          corruption_channel, 
          config: Config_SCSI_EStep) -> Dataset:
    """
    E-step of the SCSI algorithm: Integrate the ODE to sample from the posterior distribution
    of the latent variables given the observations.
    """
    model.eval()
    device = next(model.parameters()).device
    dataloader = DataLoader(observations, batch_size=config.batch_size, shuffle=True)
    dataloader_iter = iter(dataloader)

    num_samples = config.num_samples
    batch_size = config.batch_size
    # Ceil, not floor: the iterator-refresh below (StopIteration -> re-iter) means we're never
    # short a batch, so round up rather than silently dropping the remainder batch worth of
    # samples num_samples // batch_size would floor away.
    num_integrations = -(-num_samples // batch_size)

    x1_batches = []
    y_batches = []
    with torch.no_grad():
        for _ in range(num_integrations):
            try:
                (ys,) = next(dataloader_iter)  # observations is a 1-tensor TensorDataset
            except StopIteration:
                dataloader_iter = iter(dataloader)
                (ys,) = next(dataloader_iter)
            ys = ys.to(device)
            x0s = base_dist.sample(ys.size(0))
            x1s = euler_integration(model, x0s, ys, config.n_steps_sampling)

            x1_batches.append(x1s)
            ys_hat = corruption_channel(x1s)
            y_batches.append(ys_hat)

    x1s_all = torch.cat(x1_batches, dim=0)[:config.num_samples]
    ys_all = torch.cat(y_batches, dim=0)[:config.num_samples]

    return TensorDataset(x1s_all, ys_all) # (x_hat, y_hat) pairs for M-step

class EMA:
    """
    Exponential moving average of a model's weights, updated once per optimizer step.
    """

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.ema_model.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for eb, b in zip(self.ema_model.buffers(), model.buffers()):
            eb.copy_(b)


def mstep_lifted(model,
                 base_dist: Distribution,
                 dataset: Dataset,
                 optimizer: Optimizer,
                 config: Config_SCSI_MStep,
                 scheduler: LRScheduler | None = None,
                 ema: EMA | None = None,
                 global_step: list[int] | None = None,
                 log_prefix: str = "train"):
    """
    M-step of the SCSI algorithm: Update the model parameters to maximize the expected log-likelihood
    of the observed data given the latent variables.
    """
    model.train()
    device = next(model.parameters()).device

    batch_size = config.batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_iter = iter(dataloader)

    for _ in range(config.n_steps_train):
        try:
            x_hat, y_hat = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            x_hat, y_hat = next(dataloader_iter)
        x_hat, y_hat = x_hat.to(device), y_hat.to(device)

        optimizer.zero_grad()
        ts = torch.rand(x_hat.size(0), 1, 1, 1, device=device)  # Random time for each sample
        x0s = base_dist.sample(x_hat.size(0))
        loss = loss_ConditionalDrift(model, x0=x0s, x1=x_hat, t=ts, y=y_hat, style=config.interpolant_style)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        if global_step is not None:
            wandb.log({f"{log_prefix}/loss": loss.item(),
                      f"{log_prefix}/grad_norm": grad_norm.item(),
                      f"{log_prefix}/lr": optimizer.param_groups[0]["lr"]}, step=global_step[0])
            global_step[0] += 1



        