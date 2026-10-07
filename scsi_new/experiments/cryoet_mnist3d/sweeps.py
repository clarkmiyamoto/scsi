"""Sweep definitions for scsi_new.sbatch_gen. Per-sweep design notes: sbatch/<sweep>/README.md."""
from scsi_new.sbatch_gen import Job, Sweep


def k(n: int) -> str:
    return f"{n // 1000}k"


# EMNIST ('digits') once up front: the jobs start together and would race on data.py's download.
PREFETCH_EMNIST = (
    "import torchvision.datasets as D; from scsi_new.paths import torchvision_data_dir as t; "
    "import scsi_new.experiments.cryoet_mnist3d.data as d; "
    "D.EMNIST(t(d.__file__), split='digits', train=True, download=True)"
)
# Batch sizes profiled at ~77% of a 48 GB L40 on the ~37M-param UNet3D (warmup_mstep_grid README).
BATCH = ["--warmup_batch_size 16", "--mstep_batch_size 16", "--estep_batch_size 32"]
THREE = ["--digit_classes 3", "--n_images_per_class 23000"]


def _warmup_mstep_grid() -> Sweep:
    return Sweep(
        "cryoet_mnist3d", "warmup_mstep_grid", mem="48G",
        jobs=[Job(f"three3d_w{k(w)}_m{k(m)}",
                  [f"--warmup_n_steps_train {w}", f"--mstep_n_steps_train {m}",
                   f"--wandb_run_name w{k(w)}_m{k(m)}"], time="24:00:00")
              for w in (10_000, 20_000, 40_000) for m in (2_000, 5_000, 10_000)],
        common=[*THREE, *BATCH, "--num_scsi_steps 12",
                "--wandb_project scsi-cryoet-mnist3d-three-warmup-mstep-grid"],
        prefetch=PREFETCH_EMNIST,
        description="digit 3, 3D: warmup length x M-step length")


def _fresh_lr_tied_steps_grid() -> Sweep:
    days = {10_000: "2-00:00:00", 25_000: "4-00:00:00", 40_000: "6-00:00:00", 60_000: "7-00:00:00"}
    jobs = []
    for lr in ("1e-4", "3e-4", "1e-3"):
        for n in days:  # warmup and M-step length tied: w{n}_m{n}
            run = f"lr{lr}_w{k(n)}_m{k(n)}"
            jobs.append(Job(f"fresh3d_{run}",
                            [f"--warmup_n_steps_train {n}", f"--mstep_lr {lr}",
                             f"--mstep_n_steps_train {n}", f"--ckpt_dir {{ckpt}}/{run}",
                             f"--wandb_run_name {run}"], time=days[n], chain=1))
    return Sweep(
        "cryoet_mnist3d", "fresh_lr_tied_steps_grid", mem="48G", jobs=jobs,
        common=["--student_init fresh", *THREE, *BATCH, "--warmup_lr 3e-4", "--num_scsi_steps 12",
                "--resume", "--wandb_project scsi-cryoet-mnist3d-three-fresh-lr-steps"],
        prefetch=PREFETCH_EMNIST,
        legacy_ckpt={"rusty": "/mnt/ceph/users/cmiyamoto/scsi_checkpoints/cryoet_mnist3d/fresh_lr_tied_steps_grid"},
        description="fresh student, warmup == M-step length: lr x steps")


def _lr_schedule_ablation() -> Sweep:
    warm = "lrabl_warmup"
    pre = ["--num_scsi_steps 24", "--load_warmup_ckpt {ckpt}/warmup_w20k.pt"]

    def arm(name: str, *extra: str) -> Job:
        return Job(f"lrabl_{name}", [*pre, f"--ckpt_dir {{ckpt}}/{name}", *extra,
                                     f"--wandb_run_name {name}"], time="2-00:00:00", after=warm)

    return Sweep(
        "cryoet_mnist3d", "lr_schedule_ablation", mem="48G",
        jobs=[Job(warm, ["--num_scsi_steps 0", "--lr_horizon_scsi_steps 12",
                         "--save_warmup_ckpt {ckpt}/warmup_w20k.pt", "--wandb_run_name warmup_w20k"],
                  time="12:00:00"),
              arm("cos_h12", "--lr_schedule cosine", "--lr_horizon_scsi_steps 12"),
              arm("cos_h12_seed2", "--lr_schedule cosine", "--lr_horizon_scsi_steps 12", "--em_seed 2"),
              arm("cos_h24", "--lr_schedule cosine", "--lr_horizon_scsi_steps 24"),
              arm("cos_h50", "--lr_schedule cosine", "--lr_horizon_scsi_steps 50"),
              arm("cos_h50_ema", "--lr_schedule cosine", "--lr_horizon_scsi_steps 50", "--sample_with_ema"),
              arm("cos_per_mstep", "--lr_schedule cosine_per_mstep"),
              arm("const_1e-4", "--lr_schedule constant", "--mstep_lr 1e-4"),
              arm("const_3e-5", "--lr_schedule constant", "--mstep_lr 3e-5")],
        common=[*THREE, *BATCH, "--warmup_n_steps_train 20000", "--mstep_n_steps_train 5000",
                "--wandb_project scsi-cryoet-mnist3d-three-lr-schedule"],
        legacy_ckpt={"nyu": "/scratch/cm6627/scsi_lr_ablation/checkpoints"},
        description="LR schedule / EMA arms off one shared digit-3 warm start")


def _tenclass_sweep() -> Sweep:
    warm40 = "ten_warmup_w40k"

    def warmup(w: int, time: str) -> Job:
        return Job(f"ten_warmup_w{k(w)}",
                   [f"--warmup_n_steps_train {w}", "--mstep_n_steps_train 5000",
                    "--lr_horizon_scsi_steps 12", "--num_scsi_steps 0",
                    f"--save_warmup_ckpt {{ckpt}}/warmup_w{k(w)}.pt", f"--wandb_run_name warmup_w{k(w)}"],
                   time=time)

    def arm(name: str, samples: int, m: int, *extra: str) -> Job:
        sched = ["--lr_schedule cosine", "--lr_horizon_scsi_steps 12"]
        return Job(f"ten_{name}",
                   ["--warmup_n_steps_train 40000", f"--estep_num_samples {samples}",
                    f"--mstep_n_steps_train {m}", "--num_scsi_steps 24",
                    *(extra if any(e.startswith("--lr_schedule") for e in extra) else sched + list(extra)),
                    "--load_warmup_ckpt {ckpt}/warmup_w40k.pt", f"--ckpt_dir {{ckpt}}/{name}", "--resume",
                    f"--wandb_run_name {name}"], time="2-00:00:00", after=warm40, chain=1)

    return Sweep(
        "cryoet_mnist3d", "tenclass_sweep", mem="64G",
        jobs=[warmup(40_000, "16:00:00"), warmup(20_000, "12:00:00"), warmup(80_000, "1-00:00:00"),
              arm("s2k_m5k", 2000, 5000), arm("s4k_m5k", 4000, 5000), arm("s8k_m5k", 8000, 5000),
              arm("s4k_m10k", 4000, 10000),
              arm("s4k_m5k_h24", 4000, 5000, "--lr_schedule cosine", "--lr_horizon_scsi_steps 24"),
              arm("s4k_m5k_permstep", 4000, 5000, "--lr_schedule cosine_per_mstep"),
              arm("s4k_m5k_seed2", 4000, 5000, "--em_seed 2")],
        common=["--digit_classes 0 1 2 3 4 5 6 7 8 9", "--n_images_per_class 5000", *BATCH,
                "--eval_n 100", "--viz_n_pool 30", "--viz_n_display 10",
                "--wandb_project scsi-cryoet-mnist3d-ten-sweep"],
        legacy_ckpt={"nyu": "/scratch/cm6627/scsi_tenclass/checkpoints"},
        description="all ten digits: E-step samples / M-step length / schedule arms")


def _supervised(name: str, lift: bool) -> Sweep:
    sfx = "_lift" if lift else ""
    jobs = [Job(f"sup3d_{arch}_{interp}{sfx}",
                [f"--arch {arch}", f"--interpolant_style {interp}",
                 f"--checkpoint_dir {{ckpt}}/{arch}_{interp}{sfx}",
                 f"--wandb_run_name {arch}_{interp}{sfx}"], time="48:00:00")
            for arch in ("dit", "unet") for interp in ("gvp", "linear")]
    return Sweep(
        "cryoet_mnist3d", name, mem="48G", jobs=jobs, entry="main_supervised",
        common=[*(["--lift"] if lift else []), *THREE, "--n_steps_train 300000", "--batch_size 16",
                "--log_every 20000", "--viz_ema", "--viz_n_steps_sampling 16 32 64 128 256 512 1024",
                "--checkpoint_steps " + " ".join(str(s) for s in range(20_000, 300_001, 20_000)),
                f"--wandb_project scsi-cryoet-mnist3d-supervised-arch-interp{sfx.replace('_', '-')}"],
        prefetch=PREFETCH_EMNIST,
        description="supervised upper bound: backbone x interpolant" + (", lifted" if lift else ""))


SWEEPS = {s.name: s for s in [
    _warmup_mstep_grid(), _fresh_lr_tied_steps_grid(), _lr_schedule_ablation(), _tenclass_sweep(),
    _supervised("supervised_arch_interp", lift=False), _supervised("supervised_arch_interp_lift", lift=True),
]}
