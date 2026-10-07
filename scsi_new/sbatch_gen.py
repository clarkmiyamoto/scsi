"""
Generate sbatch scripts for a sweep, for either cluster, from one definition.

A sweep is a `Sweep` listed in `experiments/<experiment>/sweeps.py` (its `SWEEPS` dict): shared
args plus a list of `Job`s. One command renders every job and a `submit_all.sh` that queues them
with their dependencies:

    uv run python -m scsi_new.sbatch_gen --list
    uv run python -m scsi_new.sbatch_gen cryoet_mnist3d lr_schedule_ablation --cluster nyu
    bash scsi_new/experiments/cryoet_mnist3d/sbatch/lr_schedule_ablation/generated/nyu/submit_all.sh

Output lands in <experiment>/sbatch/<sweep>/generated/<cluster>/ (gitignored: the definition in
sweeps.py is the source of truth). Nothing here submits a job; submit_all.sh does, when you run it.

Cluster profiles (header, module loads, paths) are PROFILES below and scsi_new/paths.py. Jobs run
the experiment as a module from the repo root:
    cd <repo> && uv run python -m scsi_new.experiments.<experiment>.<entry> <args>
Point --repo at a separate checkout (git worktree) to freeze the code a sweep runs while you keep
editing the main one.

Regenerated scripts write checkpoints under <runs>/<experiment>/<sweep>/checkpoints, NOT where
the old SBATCH files did: with --resume they would silently start NEW runs next to an existing one.
To continue an existing chain pass --legacy-ckpt (sweeps with a legacy_ckpt root only).

Placeholders usable in Job args: {ckpt} -> <runs>/<experiment>/<sweep>/checkpoints, {runs} -> <runs>.
"""
import argparse
import importlib
import shlex
import stat
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import paths

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
MAIL_USER = "cm6627@nyu.edu"

# Per-cluster batch-script boilerplate. NYU: no partition / constraint (not pinned down yet;
# the CLAUDE.md template sets neither) and uv comes from ~/.local/bin, caches on scratch.
PROFILES = {
    "rusty": {
        "shebang": "#!/bin/bash -l",
        "sbatch": ["--partition=gpu", "--constraint=rocky9&(h100|a100-80gb)"],
        "setup": ["module load python/3.13.2 uv"],
    },
    "nyu": {
        "shebang": "#!/bin/bash",
        "sbatch": [],
        "setup": ['export PATH="$HOME/.local/bin:$PATH"',
                  "export UV_CACHE_DIR=/scratch/cm6627/.uv_cache"],
    },
}


@dataclass
class Job:
    name: str                    # sbatch job-name and file stem
    args: list[str]              # one "--flag value..." string per CLI flag, after Sweep.common
    time: str
    mem: str | None = None       # default: Sweep.mem
    cpus: int = 8
    gpu: bool = True
    entry: str | None = None     # module under the experiment (default: Sweep.entry)
    after: str | None = None     # name of a job in this sweep that must finish OK first
    chain: int = 0               # extra copies queued with afterany (N_JOBS=chain+1 total, env-overridable);
                                 # needs --resume in args so a continuation picks up latest.pt
    comment: str = ""


@dataclass
class Sweep:
    experiment: str              # directory under scsi_new/experiments/
    name: str                    # sweep name == directory under <experiment>/sbatch/
    mem: str
    jobs: list[Job]
    common: list[str] = field(default_factory=list)   # args prepended to every job
    entry: str = "main"
    description: str = ""
    legacy_ckpt: dict[str, str] = field(default_factory=dict)
                                 # cluster -> checkpoint root the pre-generator SBATCH files used.
                                 # --legacy-ckpt substitutes it for {ckpt}, so regenerated scripts
                                 # continue existing --resume chains instead of starting new runs.
    prefetch: str | None = None  # python -c snippet run once first (dataset download) so jobs
                                 # starting together don't race on it; every other job waits on it


def load_sweeps(experiment: str) -> dict[str, Sweep]:
    mod = importlib.import_module(f"scsi_new.experiments.{experiment}.sweeps")
    return mod.SWEEPS


def all_sweeps() -> list[Sweep]:
    out = []
    for d in sorted(p for p in EXPERIMENTS_DIR.iterdir() if (p / "sweeps.py").exists()):
        out += load_sweeps(d.name).values()
    return out


def _sub(text: str, ckpt: str, runs: str) -> str:
    return text.replace("{ckpt}", ckpt).replace("{runs}", runs)


def ckpt_root(sweep: Sweep, cluster: str, runs: str, legacy: bool = False) -> str:
    if legacy:
        if cluster not in sweep.legacy_ckpt:
            raise ValueError(f"{sweep.name} has no legacy checkpoint root for {cluster}")
        return sweep.legacy_ckpt[cluster]
    return f"{runs}/{sweep.experiment}/{sweep.name}/checkpoints"


def render_job(sweep: Sweep, job: Job, cluster: str, repo: str, runs: str,
               legacy: bool = False) -> str:
    prof = PROFILES[cluster]
    sweep_runs = f"{runs}/{sweep.experiment}/{sweep.name}"
    ckpt = ckpt_root(sweep, cluster, runs, legacy)
    entry = job.entry or sweep.entry
    L = [prof["shebang"], f"#SBATCH --job-name={job.name}", "#SBATCH --nodes=1",
         "#SBATCH --ntasks-per-node=1", f"#SBATCH --cpus-per-task={job.cpus}",
         f"#SBATCH --mem={job.mem or sweep.mem}"]
    if job.gpu:
        L.append("#SBATCH --gres=gpu:1")
    L += [f"#SBATCH --time={job.time}"]
    L += [f"#SBATCH {x}" for x in prof["sbatch"]]
    L += [f"#SBATCH --output={sweep_runs}/logs/%x-%j.out",
          f"#SBATCH --error={sweep_runs}/logs/%x-%j.err",
          "#SBATCH --mail-type=END,FAIL", f"#SBATCH --mail-user={MAIL_USER}", "",
          f"# GENERATED by scsi_new.sbatch_gen from experiments/{sweep.experiment}/sweeps.py "
          f"({sweep.name}) -- edit the definition, not this file."]
    if job.comment:
        L += [f"# {c}" for c in job.comment.strip().splitlines()]
    L += ["", "set -euo pipefail", *prof["setup"],
          f"export SCSI_CLUSTER={cluster}", f"export SCSI_RUNS_DIR={shlex.quote(runs)}",
          f"cd {shlex.quote(repo)}", ""]
    args = [_sub(a, ckpt, runs) for a in [*sweep.common, *job.args]]
    cmd = [f"uv run python -m scsi_new.experiments.{sweep.experiment}.{entry}", *args]
    L.append(" \\\n    ".join(cmd))
    return "\n".join(L) + "\n"


def render_submit_all(sweep: Sweep, cluster: str, runs: str, legacy: bool = False) -> str:
    sweep_runs = f"{runs}/{sweep.experiment}/{sweep.name}"
    names = {j.name for j in sweep.jobs}
    for j in sweep.jobs:
        if j.after and j.after not in names:
            raise ValueError(f"{sweep.name}: {j.name} waits on unknown job {j.after!r}")
    L = ["#!/bin/bash",
         f"# GENERATED -- queue every {sweep.experiment}/{sweep.name} job for {cluster}, dependencies",
         "# included. Optionally pass job names to queue only those (their dependencies must exist).",
         "#   bash submit_all.sh [job_name ...]",
         "set -euo pipefail", 'cd "$(dirname "${BASH_SOURCE[0]}")"',
         f"mkdir -p {shlex.quote(sweep_runs)}/logs {shlex.quote(ckpt_root(sweep, cluster, runs, legacy))}",
         'ONLY=" $* "', 'want() { [ "$ONLY" = "  " ] || [[ "$ONLY" == *" $1 "* ]]; }', "declare -A ID", ""]
    if sweep.prefetch:
        L += ["PREFETCH=$(sbatch --parsable prefetch.sbatch)", 'echo "prefetch.sbatch: $PREFETCH"', ""]
    # Dependencies first: the sweep's jobs in topological order (after-targets before dependents).
    done: set[str] = set()
    order: list[Job] = []
    while len(order) < len(sweep.jobs):
        before = len(order)
        for j in sweep.jobs:
            if j.name not in done and (j.after is None or j.after in done):
                order.append(j)
                done.add(j.name)
        if len(order) == before:
            raise ValueError(f"{sweep.name}: dependency cycle")
    for j in order:
        dep = []
        if j.after:
            dep.append(f"afterok:${{ID[{j.after}]}}")
        elif sweep.prefetch:
            dep.append("afterok:$PREFETCH")
        flag = f' --dependency={",".join(dep)}' if dep else ""
        L += [f"if want {j.name}; then",
              f'    ID[{j.name}]=$(sbatch --parsable{flag} {j.name}.sbatch)',
              f'    echo "{j.name}.sbatch: ${{ID[{j.name}]}}"']
        if j.chain:
            # N_JOBS total copies (default chain + 1): continuations of the same file, each
            # afterany on the one before. --resume continues from latest.pt after a walltime
            # kill, or exits at once if the run already finished.
            L += [f'    CH=${{ID[{j.name}]}}',
                  f'    for _ in $(seq 2 "${{N_JOBS:-{j.chain + 1}}}"); do',
                  f'        CH=$(sbatch --parsable --dependency=afterany:$CH {j.name}.sbatch)',
                  f'        echo "{j.name}.sbatch (continuation): $CH"',
                  '    done']
        L += ["fi"]
    return "\n".join(L) + "\n"


def render_prefetch(sweep: Sweep, cluster: str, repo: str, runs: str) -> str:
    job = Job("prefetch", [], "00:30:00", mem="8G", cpus=2, gpu=False,
              comment="One-off dataset download so the real jobs don't race on it.")
    text = render_job(replace(sweep, common=[], prefetch=None), job, cluster, repo, runs)
    head, _, _ = text.rpartition("uv run python -m")
    return head + f"uv run python -c {shlex.quote(sweep.prefetch)}\n"


def generate(sweep: Sweep, cluster: str, repo: str, runs: str, out: Path | None = None,
             legacy: bool = False) -> Path:
    out = out or (EXPERIMENTS_DIR / sweep.experiment / "sbatch" / sweep.name / "generated" / cluster)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.sbatch"):
        old.unlink()
    files = {f"{j.name}.sbatch": render_job(sweep, j, cluster, repo, runs, legacy) for j in sweep.jobs}
    if sweep.prefetch:
        files["prefetch.sbatch"] = render_prefetch(sweep, cluster, repo, runs)
    files["submit_all.sh"] = render_submit_all(sweep, cluster, runs, legacy)
    for name, text in files.items():
        p = out / name
        p.write_text(text)
        if name.endswith(".sh"):
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("experiment", nargs="?")
    ap.add_argument("sweep", nargs="?", help="default: every sweep of the experiment")
    ap.add_argument("--cluster", choices=sorted(PROFILES), default=None,
                    help="default: $SCSI_CLUSTER or autodetected (scsi_new.paths.cluster)")
    ap.add_argument("--repo", help="checkout the jobs cd into (default: the cluster's repo path)")
    ap.add_argument("--runs-dir", help="logs / checkpoints root (default: the cluster's runs dir)")
    ap.add_argument("--legacy-ckpt", action="store_true",
                    help="use the checkpoint root the old hand-written SBATCH files used, so jobs "
                         "continue existing --resume runs (only sweeps that define one)")
    ap.add_argument("--out", type=Path, help="output dir (single sweep only)")
    ap.add_argument("--list", action="store_true", help="list sweeps and exit")
    a = ap.parse_args(argv)

    if a.list or not a.experiment:
        for s in all_sweeps():
            print(f"{s.experiment:16s} {s.name:28s} {len(s.jobs):3d} jobs  {s.description}")
        return 0
    sweeps = load_sweeps(a.experiment)
    chosen = [sweeps[a.sweep]] if a.sweep else list(sweeps.values())
    if a.out and len(chosen) != 1:
        ap.error("--out needs a single sweep")
    cluster = a.cluster or paths.cluster()
    prof = paths.PROFILES[cluster]
    repo, runs = a.repo or prof["repo"], a.runs_dir or prof["runs_dir"]
    for s in chosen:
        out = generate(s, cluster, repo, runs, a.out, a.legacy_ckpt)
        print(f"{s.experiment}/{s.name}: {len(s.jobs)} jobs -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
