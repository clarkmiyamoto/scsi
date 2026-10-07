"""
Where things live, per cluster. Nothing outside this file should hardcode a filesystem path.

Two profiles, picked by $SCSI_CLUSTER ("rusty" | "nyu"); unset, we guess "rusty" when /mnt/ceph
exists and "nyu" otherwise. Every location can also be overridden individually by an env var:

    SCSI_REPO_DIR          checkout that generated sbatch jobs `cd` into  (repo_dir())
    SCSI_RUNS_DIR          checkpoints, slurm logs, run outputs      (runs_dir())
    SCSI_DATA_DIR          torchvision download root for (E)MNIST    (torchvision_data_dir())
                           default <profile data_dir>/<experiment name>
    SCSI_IGG_DATA_ROOT     IgG-1D dataset                            (igg_data_root())
    SCSI_CRYOBENCH_ROOT    CryoBench checkout                        (cryobench_root())

The NYU IgG-1D / CryoBench defaults are placeholders -- set the env vars to wherever they are.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "rusty": {
        "repo": "/mnt/home/cmiyamoto/scsi",
        "runs_dir": "/mnt/ceph/users/cmiyamoto/scsi_runs",
        "data_dir": "/mnt/ceph/users/cmiyamoto/scsi_data",
        "igg_data_root": "/mnt/ceph/users/cmiyamoto/IgG-1D",
        "cryobench_root": "/mnt/home/cmiyamoto/CryoBench",
    },
    "nyu": {
        "repo": "/scratch/cm6627/scsi",
        "runs_dir": "/scratch/cm6627/scsi_runs",
        "data_dir": "/scratch/cm6627/scsi_data",
        "igg_data_root": "/scratch/cm6627/IgG-1D",
        "cryobench_root": "/scratch/cm6627/CryoBench",
    },
}


def cluster() -> str:
    name = os.environ.get("SCSI_CLUSTER") or ("rusty" if Path("/mnt/ceph").exists() else "nyu")
    if name not in PROFILES:
        raise ValueError(f"SCSI_CLUSTER={name!r}; expected one of {sorted(PROFILES)}")
    return name


def _get(env_var: str, key: str) -> str:
    return os.environ.get(env_var) or PROFILES[cluster()][key]


def repo_dir() -> str:
    """Checkout the sbatch generator points jobs at (override: $SCSI_REPO_DIR)."""
    return _get("SCSI_REPO_DIR", "repo")


def runs_dir() -> str:
    return _get("SCSI_RUNS_DIR", "runs_dir")


def igg_data_root() -> str:
    return _get("SCSI_IGG_DATA_ROOT", "igg_data_root")


def cryobench_root() -> str:
    return _get("SCSI_CRYOBENCH_ROOT", "cryobench_root")


def torchvision_data_dir(experiment_file: str) -> str:
    """
    $SCSI_DATA_DIR if set, else <profile data_dir>/<experiment dir name>. On Rusty that is where
    experiments/cryoet_mnist3d/data (a symlink committed to git) already pointed; on NYU that
    symlink is broken, which is why the default no longer lives inside the checkout.
    """
    return os.environ.get("SCSI_DATA_DIR") or str(
        Path(PROFILES[cluster()]["data_dir"]) / Path(experiment_file).resolve().parent.name)
