"""Sweep definitions for scsi_new.sbatch_gen. Per-sweep design notes: sbatch/<sweep>/README.md."""
from scsi_new.sbatch_gen import Job, Sweep


def k(n: int) -> str:
    return f"{n // 1000}k"


# Download MNIST once, before the jobs start together and race on data.py's first-use download.
PREFETCH_MNIST = (
    "import torchvision.datasets as D; from scsi_new.paths import torchvision_data_dir as t; "
    "import scsi_new.experiments.cryoet_mnist.data as d; D.MNIST(t(d.__file__), train=True, download=True)"
)

WARMUPS = (5_000, 10_000, 20_000, 40_000)
MSTEPS = (2_000, 5_000, 10_000, 15_000, 20_000)


def _wm_jobs(prefix: str) -> list[Job]:
    return [Job(f"{prefix}_w{k(w)}_m{k(m)}", [f"--warmup_n_steps_train {w}", f"--mstep_n_steps_train {m}",
                                               f"--wandb_run_name w{k(w)}_m{k(m)}"], time="12:00:00")
            for w in WARMUPS for m in MSTEPS]


def _fresh_jobs() -> list[Job]:
    days = {5_000: "1-00:00:00", 10_000: "2-00:00:00", 20_000: "4-00:00:00"}
    out = []
    for lr in ("1e-4", "3e-4", "1e-3"):
        for w in (10_000, 20_000, 40_000):
            for m in (5_000, 10_000, 20_000):
                run = f"lr{lr}_w{k(w)}_m{k(m)}"
                out.append(Job(f"fresh_{run}", [f"--warmup_n_steps_train {w}", f"--mstep_lr {lr}",
                                                f"--mstep_n_steps_train {m}", f"--wandb_run_name {run}"],
                               time=days[m]))
    return out


SWEEPS = {s.name: s for s in [
    Sweep("cryoet_mnist", "warmup_mstep_grid", mem="32G", jobs=_wm_jobs("three"),
          common=["--digit_classes 3", "--n_images_per_class 6000", "--num_scsi_steps 40",
                  "--wandb_project scsi-cryoet-mnist-three-warmup-mstep-grid"],
          prefetch=PREFETCH_MNIST,
          description="digit 3: warmup length x M-step length"),
    Sweep("cryoet_mnist", "multiple_digits", mem="32G", jobs=_wm_jobs("multi"),
          common=["--n_images_per_class 6000", "--num_scsi_steps 40",
                  "--wandb_project scsi-cryoet-mnist-multi-warmup-mstep-grid"],
          prefetch=PREFETCH_MNIST,
          description="all ten digits: warmup length x M-step length"),
    Sweep("cryoet_mnist", "fresh_lr_warmup_mstep_grid", mem="32G", jobs=_fresh_jobs(),
          common=["--student_init fresh", "--digit_classes 3", "--n_images_per_class 6000",
                  "--warmup_lr 3e-4", "--num_scsi_steps 40",
                  "--wandb_project scsi-cryoet-mnist-three-fresh-lr-warmup-mstep"],
          prefetch=PREFETCH_MNIST,
          description="fresh student: M-step lr x warmup x M-step length"),
]}
