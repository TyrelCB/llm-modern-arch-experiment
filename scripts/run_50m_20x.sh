#!/usr/bin/env bash
# 50M-parameter model at 20 tokens per parameter on the 8B FineMath corpus.
#
# This rung changes the axis. The 5M/10M/20M runs held tokens-per-parameter at
# 100 and scaled capacity; loss improved cleanly (2.81 -> 2.58 -> 2.41) while
# benchmark accuracy flatlined (0.26% -> 0.52% -> 0.50% non-echo). At 20M the
# model writes a correct algebra procedure with wrong arithmetic inside it:
#
#     9x + 11 = 146
#     9x = 146          <- subtraction skipped
#     x = 14            <- should be 15, and it divided the wrong number
#
# So procedure is learned well before arithmetic, and more tokens per parameter
# was not converting into capability.
#
# The comparison that motivates 20x: the 145M run reached 6.61% at 13.8
# tokens/param, not 100. Its budget was Chinchilla-ish, not token-saturated.
# This run asks whether CAPACITY is the binding constraint rather than tokens --
# 50M body params at a budget in the same regime that actually worked at 145M.
#
# "50M" counts TRANSFORMER BODY parameters -- blocks plus the final norm, with
# embeddings and lm_head excluded. Body = 52,324,672 (verified against the real
# ModernLM); stored count is 71,199,040.
#
#   1,046,493,440 tokens = 20 x 52,324,672
#   0.131 epochs of the 8B corpus, so every token is fresh.
#
# Expect ~4.8h at the ~60.5k tok/s measured for this shape by
# scripts/bench_throughput.py. The three prior runs all landed within 2% of
# their benchmarked rates.
#
# Warmup is 958 updates (3% of 31,936). --planned-total-tokens is set equal to
# --target-tokens so the cosine decays across this run rather than against the
# 250M default.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

if pgrep -f 'modern_lm.train' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

exec $PY -m modern_lm.train \
  --target-tokens 1046493440 \
  --planned-total-tokens 1046493440 \
  --run-dir runs/size50m-20x \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 576 --n-layers 11 --n-heads 9 --n-kv-heads 9 --ffn-dim 1984 \
  --checkpoint-tokens 40000000 \
  --microbatch-size 16 --gradient-accumulation 4 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 958 \
  --log-every 10 \
  --seed 2026
