# Experiment journal — how to write an entry

Every experiment directory has a `JOURNAL.md`, newest entry first. One entry per sweep (or per
single run worth remembering). Copy the block below, fill in **Question → Setup** *before*
submitting, and fill in **Result → Next** when the runs finish. Whoever launches a sweep — you or
Claude — appends the entry in the same change that adds the sweep to `sweeps.py`.

Rules that keep the journal useful:

- **Change from baseline** names the baseline (a run, a wandb id, or an earlier entry) and lists
  only what differs. That is what makes a result interpretable.
- **Result** is numbers and wandb run ids, no adjectives. Write `TBD` until you have them;
  never fill it from memory or guess.
- **Interpretation** is what you conclude and how sure you are. Keep it separate from Result so a
  wrong conclusion can be revised without losing the data.
- **Next** is a concrete follow-up experiment, ideally the one you would launch tomorrow.
- Entries are append-only history. To correct one, add a dated `Update:` line under it.

```markdown
## YYYY-MM-DD — short title

- **Status:** planned | running | done | abandoned
- **Question:** the one thing this answers.
- **Hypothesis:** what you expect, and what would change your mind.
- **Change from baseline:** baseline = <entry / wandb run>. Differences: `--flag value`, ...
- **Setup:** sweep `<experiment>/<sweep>` (`sweeps.py`), cluster, wandb project `...`, job ids.
- **Result:** headline numbers (metric, EM step, wandb run ids). `TBD` until the runs finish.
- **Interpretation:** what it means, confidence, caveats.
- **Next:** the follow-up experiment.
```
