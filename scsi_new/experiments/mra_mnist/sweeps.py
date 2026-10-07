"""Sweep definitions for scsi_new.sbatch_gen. Per-sweep design notes: sbatch/<sweep>/README.md."""
from scsi_new.sbatch_gen import Job, Sweep


def k(n: int) -> str:
    return f"{n // 1000}k"


PREFETCH_MNIST = (
    "import torchvision.datasets as D; from scsi_new.paths import torchvision_data_dir as t; "
    "import scsi_new.experiments.mra_mnist.data as d; D.MNIST(t(d.__file__), train=True, download=True)"
)

# w5k has no m2k cell (5k warmup is too short to be worth a 2k M-step).
CELLS = [(w, m) for w in (5_000, 10_000, 20_000, 40_000) for m in (2_000, 5_000, 10_000, 15_000, 20_000)
         if (w, m) != (5_000, 2_000)]

SWEEPS = {s.name: s for s in [
    Sweep("mra_mnist", "warmup_mstep_grid", mem="32G",
          jobs=[Job(f"mra_three_w{k(w)}_m{k(m)}",
                    [f"--warmup_n_steps_train {w}", f"--mstep_n_steps_train {m}",
                     f"--wandb_run_name w{k(w)}_m{k(m)}"], time="12:00:00")
                for w, m in CELLS],
          common=["--digit_classes 3", "--n_images_per_class 6000", "--num_scsi_steps 40",
                  "--wandb_project scsi-mra-mnist-three-warmup-mstep-grid"],
          prefetch=PREFETCH_MNIST,
          description="digit 3, multi-reference alignment: warmup length x M-step length"),
]}
