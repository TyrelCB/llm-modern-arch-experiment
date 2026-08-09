# The 10B-budget CPT reheat destroyed capability (aborted at 2.08B)

## What happened

`runs/muon-cpt-8b` resumed the finished 2B Muon run and declared a 10B total
budget so cosine would restart high and decay to the floor at 10B. It did:
Muon LR resumed at 4.59e-3.

The 2B run had **ended** at its floor, Muon 5.0e-4. So the CPT restarted at
**9.2x the learning rate the model had converged to.** That is not a warm
continuation; it is a shock.

## The damage

Benchmarks at 2050M vs the 2B anchor, both with the run-on scoring fix and a
96-token budget:

| | anchor (2B) | CPT 2050M |
|---|---|---|
| overall | 6.61% (332/5024) | **2.03%** (102/5024) |
| algebra | 12/100 | **0/100** |
| arithmetic | 26/300 | **0/300** |

z = -11.29. Not noise.

**44.3% of completions degenerated into repetition loops**:

```
9x + 11 - 9 = 146
9x = 146 - 9
9x = 146 - 9
9x = 146 - 9   ...
```

Basic arithmetic broke outright: `11 + 3 = 16`, `48 + 22 = 100`, where the
anchor scored arithmetic correctly on every item it attempted.

Heldout loss told a milder story (2.5772 vs the corpus-adjusted 2.1435
baseline, a 0.434 regression) and **training loss was recovering the whole
time** (2.84 peak -> 2.28). Loss recovery masked capability destruction. The
benchmarks are what caught it.

## The lever, and a correction

Declaring a *longer* budget makes the reheat **hotter**, not gentler -- the
resume step lands earlier on a longer cosine:

| declared budget | Muon LR at resume |
|---|---|
| 10B | 4.59e-3 (92% of peak) |
| 14B | 4.79e-3 (96%) |
| 24B | 4.93e-3 (99%) |

The lever is the declared **peak** (`--muon-learning-rate`), since the resume
point sits at ~92% of whatever peak is set under a 10B budget:

| `--muon-learning-rate` | resume LR | vs the 5e-4 the model ended at |
|---|---|---|
| 0.005 (aborted run) | 4.59e-3 | 9.2x |
| 0.001 | 9.18e-4 | 1.8x |
| 0.0006 | 5.51e-4 | 1.1x |

## Decision

Relaunch at `--muon-learning-rate 0.001` (~1.8x the converged LR) and check
benchmarks, not just loss, at the first checkpoint. If capability holds there,
the schedule is safe to run out.
