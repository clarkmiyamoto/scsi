import re
import subprocess

import pytest

from scsi_new import paths, sbatch_gen

SWEEPS = sbatch_gen.all_sweeps()
IDS = [f"{s.experiment}/{s.name}" for s in SWEEPS]


def test_sweeps_registered():
    assert len(SWEEPS) >= 10


@pytest.mark.parametrize("sweep", SWEEPS, ids=IDS)
@pytest.mark.parametrize("cluster", sorted(sbatch_gen.PROFILES))
def test_generate(sweep, cluster, tmp_path):
    prof = paths.PROFILES[cluster]
    out = sbatch_gen.generate(sweep, cluster, prof["repo"], prof["runs_dir"], tmp_path)
    names = [j.name for j in sweep.jobs]
    assert len(set(names)) == len(names), "duplicate job names"
    for j in sweep.jobs:
        text = (out / f"{j.name}.sbatch").read_text()
        # launched as a module from the repo root, never as a script inside the experiment dir
        assert f"cd {prof['repo']}\n" in text
        assert f"python -m scsi_new.experiments.{sweep.experiment}." in text
        assert "main.py" not in text.replace("GENERATED", "")
        assert "{ckpt}" not in text and "{runs}" not in text
        assert f"export SCSI_CLUSTER={cluster}" in text
        assert (cluster == "rusty") == ("--partition=gpu" in text)
        assert f"/{sweep.experiment}/{sweep.name}/logs/" in text
    submit = (out / "submit_all.sh").read_text()
    for j in sweep.jobs:
        assert f"{j.name}.sbatch" in submit
    for f in out.iterdir():
        if f.suffix in (".sbatch", ".sh"):
            assert subprocess.run(["bash", "-n", str(f)]).returncode == 0, f


def test_warm_start_deps_precede_arms(tmp_path):
    sweep = next(s for s in SWEEPS if s.name == "lr_schedule_ablation")
    submit = (sbatch_gen.generate(sweep, "nyu", "/r", "/runs", tmp_path) / "submit_all.sh").read_text()
    assert submit.index("lrabl_warmup.sbatch)") < submit.index("afterok:${ID[lrabl_warmup]}")


def test_checkpoint_paths_stay_inside_runs_dir(tmp_path):
    for s in SWEEPS:
        out = sbatch_gen.generate(s, "nyu", "/r", "/runs", tmp_path / s.name)
        for f in out.glob("*.sbatch"):
            for p in re.findall(r"--(?:ckpt_dir|load_warmup_ckpt|save_warmup_ckpt|checkpoint_dir) (\S+)", f.read_text()):
                assert p.startswith("/runs/"), (f.name, p)


LEGACY = {"cryoet_mnist3d/fresh_lr_tied_steps_grid": "rusty", "cryoet_mnist3d/lr_schedule_ablation": "nyu",
          "cryoet_mnist3d/tenclass_sweep": "nyu"}


@pytest.mark.parametrize("key,cluster", LEGACY.items())
def test_legacy_ckpt_roots(key, cluster, tmp_path):
    sweep = next(s for s, i in zip(SWEEPS, IDS) if i == key)
    prof = paths.PROFILES[cluster]
    out = sbatch_gen.generate(sweep, cluster, prof["repo"], prof["runs_dir"], tmp_path, legacy=True)
    root = sweep.legacy_ckpt[cluster]
    texts = [f.read_text() for f in out.glob("*.sbatch")]
    assert any(f"{root}/" in t for t in texts)
    assert not any(f"{prof['runs_dir']}/{sweep.experiment}/{sweep.name}/checkpoints" in t for t in texts)
    with pytest.raises(ValueError):
        sbatch_gen.generate(sweep, "nyu" if cluster == "rusty" else "rusty", "/r", "/runs", tmp_path, legacy=True)
