#!/usr/bin/env bash
# 300M-parameter model at 20 tokens per parameter on the 8B FineMath corpus.
#
# The capacity axis started paying off at 100M. Holding 20 tpp and doubling
# capacity:
#
#     20M @ 100x   loss 2.41    1.83%   algebra  1/100   arith  3/300
#     50M @  20x   loss 2.297   2.09%   algebra  2/100   arith  3/300
#    100M @  20x   loss 2.170   2.69%   algebra 10/100   arith 15/300
#
# 20M->50M bought loss and no capability. 50M->100M bought both, concentrated
# in exactly the two subsets probe3 had been failing: algebra and arithmetic
# each 5x. Overall (+30/5024) is only ~1.5 sigma on its own, but the two
# targeted subsets moving together is what makes it real.
#
# probe3 across all 40 checkpoints of the 100M run agrees. Hit rate 1/21 over
# the first half of training, 9/19 over the second, with 2/3 appearing three
# times -- never once at 50M. At 1,650M tokens it solves the algebra probe
# cleanly end to end:
#
#     9x + 11 = 146
#     9x = 146 - 11
#     9x = 135          <- the step 50M always got wrong (it wrote 9x = 77)
#     x = 135/9
#     x = 15
#
# and 48 + 22 = 70 lands at three separate late checkpoints. Two-digit carrying,
# which failed at every single 50M checkpoint (50/58/60/64/78), now works.
#
# So this rung asks whether that trend continues to 300M or flattens. Note the
# two prior steps are not the same experiment: 50M->100M doubled capacity, this
# triples it. If the algebra/arithmetic subsets keep climbing, capacity is the
# live axis and the next question is where it saturates.
#
# "300M" counts TRANSFORMER BODY parameters -- blocks plus the final norm, with
# embeddings and lm_head excluded. Body = 296,267,264 (verified against the real
# ModernLM before launch). Stored count is 329,821,696.
#
#   5,925,345,280 tokens = 20 x 296,267,264
#   0.74 epochs of the 8B corpus, so every token is still fresh -- but only
#   just. This is the last rung that fits: 600M at 20 tpp would need 11.9B and
#   would start repeating data, which would confound the capacity result.
#
# Expect ~80h (3.3 days). Projected from the measured 50M (61.7k tok/s) and
# 100M (41.6k tok/s) rates, which give a throughput exponent of 0.632 against
# body params -> ~20.6k tok/s here. Both prior projections landed within 2%.
#
# Warmup is 5,424 updates (3% of 180,827). --planned-total-tokens is set equal
# to --target-tokens so the cosine decays across this run rather than against
# the 250M default -- a mismatch there trains at the cosine floor.
#
# Checkpoints every 150M tokens: 40 of them, ~104GB (1.3TB free). Matches the
# 100M run's 40-point density for a comparable probe3 curve, and caps rework at
# ~2h if this dies the way the 50M run did (power outage at 120M).
#
# Resume with the SAME --target-tokens and --planned-total-tokens, adding
# --resume runs/size300m-20x/latest.pt. Both are absolute cumulative counts and
# tokens_seen is restored, so the cosine continues rather than restarting.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

if pgrep -f 'modern_lm.train' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

exec $PY -m modern_lm.train \
  --target-tokens 5925345280 \
  --planned-total-tokens 5925345280 \
  --run-dir runs/size300m-20x \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 1024 --n-layers 20 --n-heads 16 --n-kv-heads 16 --ffn-dim 3456 \
  --checkpoint-tokens 150000000 \
  --microbatch-size 16 --gradient-accumulation 4 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 5424 \
  --log-every 10 \
  --seed 2026
