# warmup x mstep grid — digit 3 only

20 SBATCH jobs, one per cell of a 4x5 grid, all reconstructing **only MNIST digit 3**
(`--digit_classes 3`, `--n_images_per_class 6000`) under the cryoET-style tilt-series channel in
`experiments/cryoet_mnist/main.py`. Fixed everywhere except the two swept axes: `--num_scsi_steps 40`.

Swept:

| axis | flag | values |
|---|---|---|
| warmup training length | `--warmup_n_steps_train` | 5,000 / 10,000 / 20,000 / 40,000 |
| SCSI M-step training length (per EM iteration) | `--mstep_n_steps_train` | 2,000 / 5,000 / 10,000 / 15,000 / 20,000 |

File naming: `submit_three_w{warmup}_m{mstep}.SBATCH`, e.g. `submit_three_w20k_m10k.SBATCH` =
`warmup_n_steps_train=20000, mstep_n_steps_train=10000`. wandb run name matches (`w20k_m10k`),
all logged to the `scsi-cryoet-mnist-three-warmup-mstep-grid` wandb project so the whole grid is
filterable in one place.

## Submitting

```bash
bash submit_all.sh
```

This pre-downloads/caches MNIST once before queuing any jobs (see the script's comment for why:
20 jobs launched together would otherwise race on `data.py`'s first-use download). Submitting
the `submit_three_*.SBATCH` files individually with plain `sbatch` works too, just don't fire
more than one of the very first jobs at the exact same moment before `./data/MNIST` exists.

## Known caveats (apply to every cell)

- **`--time=12:00:00` is a flat, requested budget, not a measured one.** Throughput extrapolated
  from a sibling codebase's comparable run (`mnist_cryoem`, ~1.2-1.4 M-step-steps/sec on the same
  GPU class) suggests most cells here need well over 12h to run all `num_scsi_steps=40` EM
  iterations to completion, worse for the higher `mstep_n_steps_train` cells. Expect many jobs to
  hit the walltime limit before finishing.
- **No checkpointing.** `main.py` never calls `torch.save`. A walltime kill loses the model
  entirely — there is nothing to resume from, only a fresh restart. What does survive: wandb logs
  the reconstruction + trajectory panels every EM step (`--viz_every` default 1), so a killed run
  still leaves a partial progression in wandb up to whatever EM step it reached.
- **The two swept axes are not cleanly isolated.** `main.py` builds one `CosineAnnealingLR` with
  `T_max = warmup_n_steps_train + num_scsi_steps * mstep_n_steps_train` spanning the entire run.
  Changing either axis changes the LR schedule the *other* phase trains under (e.g. at
  `warmup=40000, mstep=2000`, warmup consumes a third of the cosine and ends well decayed; at
  `warmup=40000, mstep=20000`, the same 40000 warmup steps run at essentially constant LR).
- **`--warmup_lr` has no effect.** The optimizer is constructed once with `mstep.lr`
  (`main.py:46-48`) and reused for both warmup and the SCSI loop — warmup always trains at the
  M-step learning rate regardless of any `--warmup_lr` flag.
