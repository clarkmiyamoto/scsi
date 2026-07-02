"""Lifted SCSI for a CryoEM-style 2D->1D channel in the point-cloud representation.

A 2D object (an MNIST-digit silhouette) is a set of N points ``X in R^{N x 2}``. The CryoEM
forward model ``F`` (:mod:`toy_2d_pc.corruption`) renders it (:mod:`toy_2d_pc.renderer`) as a
``(P, P)`` image and projects a tilt series of K noisy 1D sinograms under one unknown global
SO(2) pose. SCSI (:mod:`toy_2d_pc.scsi`) recovers a generative prior over clean clouds from
only those sinograms, via F-dagger bootstrap + warm-start (Algorithm 1) + the literal
self-consistent EM loop (Algorithm 2). Run as ``python -m toy_2d_pc scsi``.
"""
from .canonicalize import (
    chamfer,
    icp_align,
    kabsch,
    reference_canonicalize,
    seed_reference,
    update_reference,
)
from .corruption import (
    backproject_tomo,
    forward_channel,
    pseudo_inverse,
    random_rotations,
    rotate_clouds,
    tilt_angles,
)
from .data import (
    image_to_pointcloud,
    load_mnist_digit_pool,
    make_mnist_sampler,
)
from .device import available_device, resolve_device
from .model import (
    ConditionalModelConfig,
    ConditionalPointCloudVelocity,
    build_conditional_model,
    clone_ema,
    ema_update_outer,
)
from .renderer import disk_splat, gaussian_splat, histogram_splat, render
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
    "tilt_angles",
    "random_rotations",
    "rotate_clouds",
    # rendering G
    "render",
    "gaussian_splat",
    "disk_splat",
    "histogram_splat",
    # data
    "load_mnist_digit_pool",
    "image_to_pointcloud",
    "make_mnist_sampler",
    # canonicalization
    "kabsch",
    "chamfer",
    "icp_align",
    "seed_reference",
    "reference_canonicalize",
    "update_reference",
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
