#!/usr/bin/env bash
# Capability curve over the 600M/8B checkpoints.
#
# NOT RUN DURING TRAINING. An eval takes ~80 min and costs ~15-20% throughput
# while it runs, and the early checkpoints it produced were all statistically
# indistinguishable from each other (300M 2.15% vs 600M 2.21%, z = 0.20) -- the
# model is in the flat pre-arithmetic phase the 145M also sat in until ~1.5B
# tokens. Paying a fifth of the GPU for points that cannot be told apart is a
# bad trade. Run this after training finishes, when it scores every retained
# checkpoint at full speed.
#
# 96-token budget with the run-on scoring fix, matching how the 145M baselines
# were re-scored, so numbers are comparable across model sizes:
#   145M @ 2B        6.61%
#   145M @ 2B + SFT  9.26%
#
# Polls: checkpoints land every ~3h across a ~10-day run. One eval at a time --
# the trainer shares this box with a resident llama-server.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
OUT=runs/curve-600m
mkdir -p "$OUT"

while :; do
  progressed=0
  for ck in runs/muon-600m-8b/checkpoint-*.pt; do
    [ -e "$ck" ] || continue
    tok=$(basename "$ck" .pt | sed 's/checkpoint-0*//')

    # Score every 3rd checkpoint (~300M tokens, ~9h apart, ~27 points).
    # An eval takes ~70-90 min against a 3h checkpoint cadence, so scoring all
    # 80 would keep the GPU on evals about half the time and slow the training
    # it is measuring. 27 points over 8B tokens is finer than the 33-point
    # curve the 145M run produced over 2B.
    # EVERY_NTH=3 skips two checkpoints in three, which is what to use if this
    # ever runs alongside training. With the GPU free, score everything.
    every=${EVERY_NTH:-1}
    if [ "$every" -gt 1 ] && [ "$tok" -lt 7900000000 ] \
       && [ $(( (tok / 100000000) % every )) -ne 0 ]; then
      continue
    fi

    out="$OUT/m600-$tok.jsonl"
    if [ -f "$out" ] && [ "$(wc -l < "$out" 2>/dev/null)" -eq 5024 ]; then continue; fi

    # Don't read a checkpoint mid-write.
    a=$(stat -c %s "$ck" 2>/dev/null || echo 0); sleep 20
    b=$(stat -c %s "$ck" 2>/dev/null || echo 0)
    [ "$a" = "$b" ] && [ "$a" != "0" ] || continue

    # Defer rather than risk OOM-ing a 10-day run.
    avail=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
    if [ "${avail:-0}" -lt 14 ]; then
      echo "[$(date '+%F %T')] deferring @ $tok (${avail}GB available)"
      continue
    fi

    # One eval per checkpoint even if a manual run is already scoring it.
    if ! mkdir "$OUT/.lock-$tok" 2>/dev/null; then continue; fi
    echo "[$(date '+%F %T')] eval @ $tok"
    # Keep stderr: a failure discarded to /dev/null is invisible for days on a
    # run this long, and the progress events are the only way to tell a crash
    # from a slow eval.
    err="$OUT/eval-$tok.err"
    # --use-cache is a 4x speedup with bit-identical greedy output
    # (tests/test_kv_cache.py), so the scores stay comparable to every arm
    # scored without it.
    if PYTHONPATH=src $PY -m modern_lm.evaluate_benchmarks --checkpoint "$ck" \
        --output "$out" --max-new-tokens 96 --device cuda --use-cache \
        > /dev/null 2>"$err"; then
      progressed=1
      rm -f "$err"
    else
      echo "[$(date '+%F %T')] FAILED @ $tok -- $(tail -1 "$err" 2>/dev/null | cut -c1-160)"
    fi
    rmdir "$OUT/.lock-$tok" 2>/dev/null
  done

  if ! pgrep -f 'modern_lm.train' > /dev/null 2>&1 && [ "$progressed" -eq 0 ]; then
    break
  fi
  [ "$progressed" -eq 0 ] && sleep 600
done
echo "[$(date '+%F %T')] 600M CURVE DONE"
