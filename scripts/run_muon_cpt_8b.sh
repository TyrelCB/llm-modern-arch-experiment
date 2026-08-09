#!/usr/bin/env bash
# Continued pretraining: the finished 2B Muon run carried on over the 8B-token
# finemath corpus, for a 10B total token budget.
#
# The schedule is the whole point of this script. train.py computes the LR from
# `optimizer_step` against `planned_total_tokens`, and the 2B run ended at step
# 61036 -- exactly its own total_updates -- so cosine had fully decayed to the
# 0.1x floor. Resuming under the old 2B plan would run all 8B tokens at that
# floor (Muon 5e-4) and waste the compute. Declaring the full 10B budget instead
# places step 61036 partway down a longer cosine: Muon resumes at 4.59e-3, near
# its 5e-3 peak, and decays to the floor at 10B. That is the correct CPT shape
# and it needs no change to the scheduler.
#
# --target-tokens is the stopping point (10B cumulative, i.e. 8B new), not the
# amount to add: `tokens_seen` is restored from the checkpoint and the loop runs
# while tokens_seen < target_tokens.
set -u
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b
BASE=runs/muon-2b-lr0.005/checkpoint-002000000000.pt

# The 8B corpus is a superset of the 2.05B one drawn from the same shards under
# the same split seed, so the 2B run's heldout documents are still held out
# here. Loss stays comparable across the join.
exec /home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python -m modern_lm.train \
  --target-tokens 10000000000 \
  --planned-total-tokens 10000000000 \
  --run-dir runs/muon-cpt-8b \
  --resume "$BASE" \
  --train "$D/train.bin" \
  --heldout "$D/heldout.bin" \
  --checkpoint-tokens 250000000 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 2000 \
  --seed 2026
