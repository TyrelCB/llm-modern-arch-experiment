#!/usr/bin/env bash
# A 600M dense model trained from scratch on the 8B-token finemath corpus.
#
# This is the size-and-data experiment: 4.15x the parameters of the 145M runs
# and 4x the tokens. Everything else matches the 145M Muon 2B recipe (Muon
# 0.005, AdamW 3e-4, 2000 warmup, seed 2026) so model size and corpus size are
# the only things that changed.
#
# Shape keeps the 145M aspect ratios: ffn_dim ~3.4x dim, head_dim 64,
# dim/n_layers ~53. 1280/24/20/4352 -> 600,375,552 parameters.
#
# microbatch 8 x accumulation 8 holds the 145M run's 32768 tokens/update while
# halving activation memory, since the model is 4x wider and deeper.
#
# Checkpoint every 100M tokens, not the 145M runs' 250M: this model trains at
# 9,266 tok/s against 35,110, so 250M would be 7.5h between data points on a
# 10-day run -- too slow to catch a bad config early, and too coarse for the
# capability curve. 100M is ~3h apart, 80 checkpoints, ~288GB at 3.6GB each,
# which fits the 1.5TB free. Keep all of them: the curve IS the result, so
# --keep-last-checkpoints would delete the thing we are trying to measure.
set -u
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

exec /home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python -m modern_lm.train \
  --target-tokens 8000000000 \
  --planned-total-tokens 8000000000 \
  --run-dir runs/muon-600m-8b \
  --train "$D/train.bin" \
  --heldout "$D/heldout.bin" \
  --dim 1280 --n-layers 24 --n-heads 20 --n-kv-heads 20 --ffn-dim 4352 \
  --checkpoint-tokens 100000000 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 2000 \
  --microbatch-size 8 \
  --gradient-accumulation 8 \
  --seed 2026
