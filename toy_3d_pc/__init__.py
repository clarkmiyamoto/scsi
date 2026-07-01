"""Lifted SCSI for CryoET in the point-cloud representation.

A 3D object is a set of N points ``X in R^{N x 3}``. The CryoET forward model ``F``
(:mod:`toy_3d_pc.corruption`) renders a tilt series of K noisy projections under one
unknown global SO(3) pose. SCSI (:mod:`toy_3d_pc.scsi`) recovers a generative prior over
clean clouds from only those projections, via F-dagger bootstrap + warm-start (Algorithm 1)
+ the literal self-consistent EM loop (Algorithm 2). Run as ``python -m toy_3d_pc scsi``.
"""
from .corruption import (
    backproject_tomo,
    forward_channel,
    pseudo_inverse,
    random_rotations,
    rotate_clouds,
    tilt_rotations,
)
from .data import (
    available_shapes,
    make_mixture_sampler,
    mixture_volume_residual,
    sample_dumbbell,
    sample_l_shape,
    sample_perturbed_dataset,
    sample_t_shape,
    sample_torus,
    sample_trefoil,
)
from .device import available_device, resolve_device
from .model import (
    ConditionalModelConfig,
    ConditionalPointCloudVelocity,
    build_conditional_model,
    clone_ema,
    ema_update_outer,
)
from .scsi import (
    load_checkpoint,
    log_bootstrap,
    log_em_step,
    save_checkpoint,
    scsi_train,
)
from .si import interpolant, transport_sample
from .supervised import train_supervised
from .tracking import Tracker
from .warmstart import find_initialization

__all__ = [
    # forward model F + pseudo-inverse F-dagger
    "forward_channel",
    "pseudo_inverse",
    "backproject_tomo",
    "tilt_rotations",
    "random_rotations",
    "rotate_clouds",
    # data
    "sample_torus",
    "sample_dumbbell",
    "sample_trefoil",
    "sample_l_shape",
    "sample_t_shape",
    "make_mixture_sampler",
    "sample_perturbed_dataset",
    "mixture_volume_residual",
    "available_shapes",
    # device
    "available_device",
    "resolve_device",
    # model
    "ConditionalPointCloudVelocity",
    "ConditionalModelConfig",
    "build_conditional_model",
    "clone_ema",
    "ema_update_outer",
    # stochastic interpolant
    "interpolant",
    "transport_sample",
    # algorithms
    "find_initialization",
    "scsi_train",
    "train_supervised",
    "save_checkpoint",
    "load_checkpoint",
    "log_em_step",
    "log_bootstrap",
    # logging
    "Tracker",
]
