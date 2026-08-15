#!/usr/bin/env bash
# 100M-parameter model at 20 tokens per parameter on the 8B FineMath corpus.
#
# The 50M/20x rung tested whether CAPACITY was the binding constraint, after the
# 5M/10M/20M runs at 100 tokens-per-parameter improved loss cleanly while
# benchmark accuracy stayed flat. It was not:
#
#     20M @ 100x   loss 2.41    1.83%  (92/5024)
#     50M @  20x   loss 2.297   2.09%  (105/5024)
#
# +13 correct on 5,024 examples is inside binomial noise (~+/-20). Loss moved,
# capability did not -- the same decoupling seen on the tokens axis.
#
# probe3 across all 27 checkpoints of that run localizes the failure. Procedure
# is learned; arithmetic is not. At 800M tokens the model writes:
#
#     9x + 11 = 146
#     9x = 146 - 11      <- correct step, correct order
#     9x = 77            <- 146-11=135, not 77
#     x = 7
#
# and 48 + 22 settles into a stable 58/60/78 -- approximating the tens digit
# without carrying the units. Two-digit carrying never arrives. The one probe
# hit in the whole sweep was 6 + 2 = 8: single-digit, no carry.
#
# So this rung doubles capacity again on the same 20x budget. It asks whether
# 2.09% is a capacity floor that 100M clears, or an architecture/corpus plateau
# that more parameters will not move. A flat result here is the stronger and
# more useful answer: it would close out the capacity axis and point at the data
# (arithmetic is rare in FineMath prose) or at SFT, which is what actually moved
# the 145M model (412 -> 568 corrected).
#
# "100M" counts TRANSFORMER BODY parameters -- blocks plus the final norm, with
# embeddings and lm_head excluded. Body = 97,793,728 (verified against the real
# ModernLM by scripts/bench_throughput.py, which asserts the count and refuses
# to run on mismatch). Stored count is 120,862,400.
#
#   1,955,874,560 tokens = 20 x 97,793,728
#   0.24 epochs of the 8B corpus, so every token is fresh.
#
# Expect ~13-16h. Extrapolated from the 50M run's sustained 61.7k tok/s rather
# than benchmarked: bench_throughput.py has no single-rung flag and sweeping the
# whole ladder to time one shape is not worth the GPU hour.
#
# Warmup is 1,790 updates (3% of 59,688). --planned-total-tokens is set equal to
# --target-tokens so the cosine decays across this run rather than against the
# 250M default -- a mismatch here trains at the cosine floor.
#
# Checkpoints every 50M tokens: 39 of them, ~38GB (1.4TB free). Dense enough for
# a probe3 capability curve comparable to the 50M run's, and it caps rework at
# ~20 minutes if this run dies the way the 50M one did (power outage at 120M).
#
# Resume after an interruption with the SAME --target-tokens and
# --planned-total-tokens, adding --resume runs/size100m-20x/latest.pt. Both are
# absolute cumulative counts and tokens_seen is restored from the checkpoint, so
# the cosine continues rather than restarting.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

if pgrep -f 'modern_lm.train' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

exec $PY -m modern_lm.train \
  --target-tokens 1955874560 \
  --planned-total-tokens 1955874560 \
  --run-dir runs/size100m-20x \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 704 --n-layers 14 --n-heads 11 --n-kv-heads 11 --ffn-dim 2368 \
  --checkpoint-tokens 50000000 \
  --microbatch-size 16 --gradient-accumulation 4 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 1790 \
  --log-every 10 \
  --seed 2026
