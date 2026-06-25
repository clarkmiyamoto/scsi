"""Flow matching generative model over 3D point clouds (CUDA / MPS optimized).

Also includes lifted SCSI (``scsi`` subcommand): recover a generative prior over
clean 3D point clouds from only their corrupted CryoET tilt-series projections,
via tomo bootstrap + supervised pretraining + EM.
"""
from .balls import point_cloud_to_balls, save_balls_obj
from .corruption import backproject_tomo, forward_channel
from .data import (
    available_shapes,
    make_mixture_sampler,
    mixture_surface_residual,
    sample_cylinder,
    sample_torus,
    torus_surface_residual,
)
from .device import available_device, resolve_device
from .flow import (
    ModelConfig,
    build_model,
    load_checkpoint,
    sample,
    save_checkpoint,
    train,
)
from .model import ConditionalPointCloudVelocity, PointCloudVelocity
from .prior import (
    BootstrapContext,
    available_bootstraps,
    make_bootstrap,
)
from .scsi import (
    ConditionalModelConfig,
    build_conditional_model,
    scsi_train,
    train_supervised,
)

__all__ = [
    "available_device",
    "resolve_device",
    "ModelConfig",
    "build_model",
    "train",
    "sample",
    "save_checkpoint",
    "load_checkpoint",
    "PointCloudVelocity",
    "point_cloud_to_balls",
    "save_balls_obj",
    # corruption channel F
    "forward_channel",
    "backproject_tomo",
    # shapes / dataset
    "sample_torus",
    "sample_cylinder",
    "torus_surface_residual",
    "mixture_surface_residual",
    "make_mixture_sampler",
    # bootstrap priors pi(0)
    "BootstrapContext",
    "make_bootstrap",
    "available_bootstraps",
    "available_shapes",
    # lifted SCSI
    "ConditionalPointCloudVelocity",
    "ConditionalModelConfig",
    "build_conditional_model",
    "scsi_train",
    "train_supervised",
]
