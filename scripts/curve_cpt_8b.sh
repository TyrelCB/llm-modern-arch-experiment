#!/usr/bin/env bash
# Capability curve over the 8B CPT checkpoints.
#
# --max-new-tokens 96, not the repo's default 32. At 32 the model's correct
# two-step algebra derivations get cut off mid-division ("x = 864/12" with the
# reduction never emitted), so extract_number picks up a trailing intermediate
# and scores them wrong: algebra reads 13/100 when it is really ~31/100. The
# scorer already evaluates fractions by value, so the budget was the only
# binding constraint. CPT is meant to improve multi-step solving, which is
# exactly what a 32-token window hides.
#
# Runs one eval at a time: the GPU is shared with a resident llama-server and
# the trainer, and concurrent evals OOM.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
OUT=runs/curve-cpt
mkdir -p "$OUT"

# The 2B endpoint is the origin of this curve, re-scored at 96 tokens so the
# baseline and the CPT points are measured identically. Without it, part of any
# apparent gain would just be the token-budget fix.
# Poll rather than sweep once: checkpoints land every ~2h across a ~63h run, so
# a single pass would evaluate the anchor and exit long before the run finished.
# Exits once training is gone and every checkpoint has a complete evaluation.
while :; do
  progressed=0
  CKPTS=(runs/muon-2b-lr0.005/checkpoint-002000000000.pt runs/muon-cpt-8b/checkpoint-*.pt)
  for ck in "${CKPTS[@]}"; do
    [ -e "$ck" ] || continue
    tok=$(basename "$ck" .pt | sed 's/checkpoint-0*//')
    out="$OUT/cpt-$tok.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out" 2>/dev/null)" -eq 5024 ]; then continue; fi
    # Don't read a checkpoint mid-write: require the file to stop growing.
    a=$(stat -c %s "$ck" 2>/dev/null || echo 0); sleep 20
    b=$(stat -c %s "$ck" 2>/dev/null || echo 0)
    [ "$a" = "$b" ] && [ "$a" != "0" ] || continue

    # An eval needs ~3GB alongside the trainer and the resident llama-server on
    # this unified-memory box. Defer rather than risk OOM-ing a 63h run; the
    # checkpoint is on disk and will still be here on the next pass.
    avail=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
    if [ "${avail:-0}" -lt 12 ]; then
      echo "[$(date '+%T')] deferring @ $tok (only ${avail}GB available)"
      continue
    fi

    echo "[$(date '+%T')] cpt @ $tok"
    PYTHONPATH=src $PY -m modern_lm.evaluate_benchmarks --checkpoint "$ck" \
      --output "$out" --max-new-tokens 96 --device cuda > /dev/null 2>&1 \
      && progressed=1 \
      || echo "[$(date '+%T')] FAILED @ $tok"
  done

  # Training finished and nothing left to score.
  if ! pgrep -f 'modern_lm.train' > /dev/null 2>&1 && [ "$progressed" -eq 0 ]; then
    break
  fi
  [ "$progressed" -eq 0 ] && sleep 300
done
echo "[$(date '+%T')] CPT CURVE DONE"
