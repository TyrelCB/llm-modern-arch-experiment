#!/usr/bin/env bash
# Chain: wait out the running 2B job, score its final checkpoint, then run the
# two weight-decay probe arms strictly one at a time.
#
# The arms exist to break a confound: Muon shrinks by `lr * weight_decay`, so
# every earlier run varied regularization and step size together. Arm B holds
# LR at the current best and cuts decay 10x; arm C then drops LR at that fixed
# decay. Compared against the two runs that already exist:
#
#   AdamW control  runs/modern-145m          lr 3e-4                 -> 2.4049
#   Muon A         runs/muon-250m-lr0.005    muon_lr 0.005, wd 0.1   -> 2.3114
#   Muon B (here)  muon_lr 0.005,  muon_wd 0.01   isolates decay
#   Muon C (here)  muon_lr 0.0025, muon_wd 0.01   isolates LR
#
# Sequential on purpose: one GB10 at ~95% util has no room for a second job,
# and these arms differ by margins small enough that contention would swamp
# them. Timing is part of the measurement.
set -uo pipefail

cd /home/tyrel/projects/llm-modern-arch-experiment
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
DATA=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-2b
TRAIN_PID="${1:-162630}"

log() { echo "[$(date '+%F %T')] $*"; }

# --- 1. wait for the in-flight 2B run -------------------------------------
log "waiting for 2B run (pid $TRAIN_PID)"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
log "2B run exited"

# Trust the checkpoint, not the exit code: the job may have been killed. A run
# that stopped short still leaves a valid latest.pt, and silently probing off a
# partial 2B run would be worse than stopping here.
FINAL_TOKENS=$($PY -c "
import json; print(json.load(open('runs/muon-2b-lr0.005/latest.json'))['state']['tokens_seen'])
" 2>/dev/null || echo 0)
log "2B final tokens: $FINAL_TOKENS"

# --- 2. score the final 2B checkpoint -------------------------------------
if [ "$FINAL_TOKENS" -ge 1999000000 ]; then
  log "scoring final 2B checkpoint"
  PYTHONPATH=src $PY -m modern_lm.evaluate_benchmarks \
    --checkpoint runs/muon-2b-lr0.005/latest.pt \
    --output runs/eval-muon2b-final.jsonl \
    --max-new-tokens 32 --device cuda \
    > runs/eval-muon2b-final.log 2>&1
  log "eval done -> runs/eval-muon2b-final.jsonl"
else
  log "2B run stopped at $FINAL_TOKENS (<2B); skipping final eval"
fi

# --- 3. the two probe arms, one at a time ---------------------------------
# checkpoint-tokens 10M matches the existing 250M runs' cadence, so milestone
# curves line up against Muon A and the AdamW control. seed/warmup/schedule are
# pinned to those runs; only the two knobs under test move.
run_arm() {
  local name=$1 muon_lr=$2 muon_wd=$3
  local dir="runs/$name"
  if [ -f "$dir/latest.json" ]; then
    log "SKIP $name (already has a checkpoint)"
    return 0
  fi
  log "START $name (muon_lr=$muon_lr muon_wd=$muon_wd)"
  PYTHONPATH=src $PY -m modern_lm.train \
    --target-tokens 250000000 --planned-total-tokens 250000000 \
    --run-dir "$dir" \
    --train "$DATA/train.bin" --heldout "$DATA/heldout.bin" \
    --checkpoint-tokens 10000000 \
    --optimizer muon \
    --muon-learning-rate "$muon_lr" \
    --muon-weight-decay "$muon_wd" \
    --learning-rate 3e-4 --warmup-updates 2000 --seed 2026 \
    > "runs/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"
  return $rc
}

run_arm muon-250m-lr0.005-wd0.01  0.005  0.01
B_RC=$?

# C is only interpretable against B: it varies LR at the decay B establishes.
# If B died, running C would burn two hours on a comparison with no baseline.
if [ $B_RC -ne 0 ]; then
  log "arm B failed (rc=$B_RC); not starting arm C"
  exit $B_RC
fi

run_arm muon-250m-lr0.0025-wd0.01 0.0025 0.01
log "ALL DONE"
