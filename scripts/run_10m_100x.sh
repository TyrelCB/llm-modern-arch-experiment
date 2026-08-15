#!/usr/bin/env bash
# 10M-parameter model trained to 100 tokens per parameter on the 8B FineMath corpus.
#
# Second rung of the size ladder, directly comparable to runs/size5m-100x: same
# recipe, same corpus, same tokens-per-parameter, so the only variable is size.
#
# "10M" counts TRANSFORMER BODY parameters -- blocks plus the final norm, with the
# embedding table and lm_head excluded. Body = 10,184,256 (verified against the
# real ModernLM); stored count is 20,670,016.
#
#   1,018,425,600 tokens = 100 x 10,184,256
#   0.127 epochs of the 8B corpus, so every token is fresh.
#
# Expect ~2h at the ~143k tok/s measured for this shape by
# scripts/bench_throughput.py. The 5M run came in within 1% of its benchmarked
# rate, so that estimate should hold.
#
# What this rung is testing: 5M @ 100x reached 2.81 heldout loss but had NO
# arithmetic capability -- 48 of its 64 benchmark "hits" were echoes of a number
# already in the question. The question here is whether 2x the body params
# starts to buy actual computation, or whether emergence needs substantially
# more capacity.
#
# Warmup is 932 updates (3% of 31,079), not the trainer's default 2000.
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
  --target-tokens 1018425600 \
  --planned-total-tokens 1018425600 \
  --run-dir runs/size10m-100x \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 320 --n-layers 7 --n-heads 5 --n-kv-heads 5 --ffn-dim 1088 \
  --checkpoint-tokens 40000000 \
  --microbatch-size 16 --gradient-accumulation 4 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 932 \
  --log-every 10 \
  --seed 2026
