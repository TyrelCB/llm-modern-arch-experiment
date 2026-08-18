#!/usr/bin/env bash
# Resume the 300M / 20 tpp run at a 30M-token checkpoint interval.
#
# The run stopped at 3,450,011,648 tokens (58.2% of 5,925,345,280) with
# optimizer_step 105,286. The log had reached 3,464,560,640 before the process
# died, so resuming replays ~14.5M tokens -- expected, since latest.pt is only
# rewritten on a checkpoint boundary, and harmless because the token stream is
# a deterministic function of position.
#
# --target-tokens and --planned-total-tokens are unchanged from the original
# launch, per the note in run_300m_20x.sh: both are absolute cumulative counts,
# so keeping them identical is what makes the cosine continue its original
# schedule rather than restart. Changing either would train at the wrong LR --
# see the CPT reheat result for what that costs.
#
# What changes here is ONLY the checkpoint interval: 150M -> 30M, for a 5x
# denser probe3 curve over the back half of training.
#
#   remaining 2,475,333,632 tokens / 30M = ~83 checkpoints at 2.8GB = ~229GB
#
# That would fit (976GB free) but is wasteful, so retention is enabled:
# --keep-last-checkpoints 10 plus --milestone-percents keeps every 10% mark
# forever alongside a rotating window of the 10 most recent. Remaining
# milestones are 4.15B (70%), 4.74B, 5.33B and 5.93B. Steady-state ~75GB.
#
# --milestone-percents is NOT optional here: it defaults to empty, and the
# first version of this script omitted it, so --keep-last-checkpoints 10 ran as
# a pure recency window and silently deleted every earlier milestone (the 10%
# through 60% weights are already gone; their JSON sidecars survive). Protection
# also needed a train.py fix -- the tolerance was one update, which cannot reach
# a 10% mark from a 30M grid that misses it by up to 15M tokens.
#
# CONSEQUENCE: at 30M intervals a rotating window of 10 covers only ~300M
# tokens, so a NON-milestone checkpoint is prunable within ~4h of being written.
# Probe a specific point soon after it lands, or it will be gone.
#
# BATCH SHAPE CHANGE, and it is a deliberate intervention on a live trajectory.
# This resume runs mb 64 x ga 1 where the run began at 16 x 4. The two are
# mathematically identical -- same 32,768 tokens per update, and token-weighted
# accumulation makes the gradient the same -- but 16x4 executes as four passes
# over 8,192-row GEMMs where 64x1 is one over 32,768 rows. bench_batch_shape.py
# measured 1.09x at this shape, peaking at 40.2GB of the 121GB pool ([D024]).
#
# The trainer records this: it diffs the resume checkpoint's settings sidecar
# against the flags it was given and writes the difference into train.jsonl as
# the run_identity record's `interventions` field, so the trajectory carries its
# own history rather than depending on anyone remembering this comment ([D020]).
#
# The 9% is a compiled measurement taken WITH the per-microbatch host syncs that
# D023 has since removed -- some of that gain was four stalls per update rather
# than GEMM shape, so treat 1.09x as an upper bound until it is remeasured.
#
# --profile-every 200 emits one synchronized data/forward/backward/optimizer
# breakdown every ~6.5M tokens. That is where the 64x1 gain gets remeasured on
# the real model rather than the bench, and the first step breakdown this repo
# has from a production run. The other timing numbers stay sync-free.
#
# ~35h remaining at the measured 17.8k tok/s, ~32h if the full 9% lands.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

# Match only a real trainer, not this script and not the shell that launched it.
if pgrep -f '^[^ ]*python[0-9.]* -m modern_lm\.(train|sft)' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

# Resume from the newest checkpoint, which is NOT always latest.pt. A SIGTERM
# can land between the numbered write and the latest.pt write, leaving latest.pt
# one interval stale; resuming from it then silently redoes 30M tokens. Pass a
# path as $1 to override.
RESUME="${1:-}"
if [ -z "$RESUME" ]; then
  NEWEST=$(ls -1 runs/size300m-20x/checkpoint-*.pt 2>/dev/null | sort | tail -1)
  LATEST=runs/size300m-20x/latest.pt
  RESUME=$LATEST
  if [ -n "$NEWEST" ] && [ -f "$LATEST" ]; then
    # Compare the token counts the two files actually record.
    n_tok=$(basename "$NEWEST" .pt | sed 's/checkpoint-0*//')
    l_tok=$($PY - "$LATEST" <<'EOF'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state"]["tokens_seen"])
EOF
)
    [ "$n_tok" -gt "$l_tok" ] && RESUME=$NEWEST
  fi
fi

if [ ! -f "$RESUME" ]; then
  echo "ABORT: resume checkpoint '$RESUME' is missing"; exit 1
fi
echo "resuming from $RESUME"

exec $PY -m modern_lm.train \
  --target-tokens 5925345280 \
  --planned-total-tokens 5925345280 \
  --run-dir runs/size300m-20x \
  --resume "$RESUME" \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 1024 --n-layers 20 --n-heads 16 --n-kv-heads 16 --ffn-dim 3456 \
  --checkpoint-tokens 30000000 \
  --keep-last-checkpoints 10 \
  --milestone-percents 10,20,30,40,50,60,70,80,90,100 \
  --microbatch-size 64 --gradient-accumulation 1 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 5424 \
  --log-every 10 \
  --profile-every 200 \
  --seed 2026
