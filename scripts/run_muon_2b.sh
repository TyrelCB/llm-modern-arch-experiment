#!/usr/bin/env bash
# Muon 0.005 at the 2B-token budget, matched to the AdamW 2B baseline
# (runs/modern-145m-2b): same corpus, seed, schedule, batch, and 50M
# checkpoint cadence. Detached with setsid so it survives disconnects.
set -u
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-2b
exec /home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python -m modern_lm.train \
  --target-tokens 2000000000 \
  --planned-total-tokens 2000000000 \
  --run-dir runs/muon-2b-lr0.005 \
  --train "$D/train.bin" \
  --heldout "$D/heldout.bin" \
  --checkpoint-tokens 50000000 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 2000 \
  --seed 2026
