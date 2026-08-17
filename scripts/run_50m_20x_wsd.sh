#!/usr/bin/env bash
# 50M @ 20 tpp with a WSD schedule -- the schedule arm of the cosine run in
# scripts/run_50m_20x.sh.
#
# EVERY flag below is identical to that script except --lr-schedule,
# --wsd-decay-fraction and --min-lr-ratio. Same shape, same 1,046,493,440
# tokens, same seed 2026, same corpus, same Muon LR, same warmup. The schedule
# is the only variable, so the comparison is clean:
#
#     cosine   loss 2.297   2.09% (105/5024)   algebra 2/100   arith 3/300
#     wsd      <- this run
#
# Warmup 958 updates (3%), then the peak 3e-4 is HELD for 24,782 updates
# (77%), then a linear decay to zero over the final 6,195 updates (20%, the
# last 203M tokens). Cosine instead starts decaying the moment warmup ends, so
# WSD spends most of the run at a strictly higher LR.
#
# --min-lr-ratio 0.0 makes the tail decay to zero rather than to the cosine
# run's 3e-5 floor. That is the published WSD form and is where most of its
# reported loss advantage comes from; it does mean the floor differs between
# the two arms, which is deliberate -- this compares WSD-as-practiced against
# cosine-as-practiced, not one isolated knob.
#
# Why this rung: at ~4.8h it is the cheapest point on the ladder that already
# has a full cosine baseline (final loss, 5,024-question benchmark, and a
# 27-point probe3 curve), so a schedule effect shows up against real numbers
# rather than against a fresh control.
#
# If WSD wins here it changes how the ladder gets run. A cosine schedule is
# welded to one token budget -- comparing 1B and 2B needs two full runs. WSD's
# stable phase means a single long run can be branched at any checkpoint into a
# short decay, yielding several budget points from one training job.
#
# Body params 52,324,672; stored 71,199,040. 0.131 epochs of the 8B corpus.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

if pgrep -f '^[^ ]*python[0-9.]* -m modern_lm\.(train|sft)' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

exec $PY -m modern_lm.train \
  --target-tokens 1046493440 \
  --planned-total-tokens 1046493440 \
  --run-dir runs/size50m-20x-wsd \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 576 --n-layers 11 --n-heads 9 --n-kv-heads 9 --ffn-dim 1984 \
  --checkpoint-tokens 40000000 \
  --microbatch-size 16 --gradient-accumulation 4 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 958 \
  --lr-schedule wsd \
  --wsd-decay-fraction 0.2 \
  --min-lr-ratio 0.0 \
  --log-every 10 \
  --seed 2026
