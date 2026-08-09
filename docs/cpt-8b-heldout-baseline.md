# CPT heldout loss: what to compare against

The 8B CPT reports heldout loss against the 8B corpus's heldout split. The
2B run's final 2.0863 was measured on the *2B* corpus's split, so the two are
not directly comparable and the raw difference overstates any regression.

Control: the untouched 2B endpoint checkpoint, scored on both splits.

| heldout set | 2B endpoint loss |
|---|---|
| 2B corpus (where 2.0863 came from) | 2.0863 |
| 8B corpus (what CPT reports) | **2.1435** |

Reproducing 2.0863 exactly confirms the method. The corpus change is worth
**0.057** — the 8B split is slightly harder, as expected from a superset drawn
from the same shards.

**Use 2.1435, not 2.0863, as the CPT baseline.**

At the first checkpoint (2050M) the CPT read 2.5772, so the real regression is
0.434, not the 0.491 the raw comparison suggests. That is genuine degradation
from the LR reheat (Muon 5e-4 -> 4.59e-3 in one step), not a measurement
artifact.

Reproduce with `scratchpad/heldout_control.py`: load the checkpoint once and
call `train.evaluate` against each heldout stream with the run's own settings.
