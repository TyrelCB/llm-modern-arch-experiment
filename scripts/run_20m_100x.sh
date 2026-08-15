#!/usr/bin/env bash
# 20M-parameter model trained to 100 tokens per parameter on the 8B FineMath corpus.
#
# Third rung of the size ladder, directly comparable to runs/size5m-100x and
# runs/size10m-100x: same recipe, same corpus, same tokens-per-parameter, so the
# only variable is model size.
#
# "20M" counts TRANSFORMER BODY parameters -- blocks plus the final norm, with the
# embedding table and lm_head excluded. Body = 20,078,016 (verified against the
# real ModernLM); stored count is 34,758,080.
#
#   2,007,801,600 tokens = 100 x 20,078,016
#   0.251 epochs of the 8B corpus, so every token is fresh.
#
# Expect ~5.2h at the ~108k tok/s measured for this shape by
# scripts/bench_throughput.py. The 5M and 10M runs both landed within 2% of
# their benchmarked rates.
#
# What this rung is testing: neither 5M nor 10M learned any arithmetic. Loss
# improved (2.81 -> 2.58) and perplexity fell (16.6 -> 13.1), but non-echo
# benchmark accuracy stayed at noise (0.32% -> 0.64%), arithmetic held at 1/300,
# and algebra went 1/100 -> 0/100. So far loss is buying fluency, not
# computation. This run doubles capacity again to see whether that continues.
#
# Warmup is 1838 updates (3% of 61,273), close to the 2000 the 145M/600M runs
# used, so this rung is nearly continuous with the established recipe.
# --planned-total-tokens is set equal to --target-tokens so the cosine decays
# across this run rather than against the 250M default.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

if pgrep -f 'modern_lm.train' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

exec $PY -m modern_lm.train \
  --target-tokens 2007801600 \
  --planned-total-tokens 2007801600 \
  --run-dir runs/size20m-100x \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 448 --n-layers 7 --n-heads 7 --n-kv-heads 7 --ffn-dim 1536 \
  --checkpoint-tokens 80000000 \
  --microbatch-size 16 --gradient-accumulation 4 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 1838 \
  --log-every 10 \
  --seed 2026
