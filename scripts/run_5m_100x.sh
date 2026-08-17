#!/usr/bin/env bash
# 5M-parameter model trained to 100 tokens per parameter on the 8B FineMath corpus.
#
# "5M" counts TRANSFORMER BODY parameters -- blocks plus the final norm, with the
# embedding table and lm_head excluded. Body = 4,754,816 (verified against the
# real ModernLM). Stored parameter count is 13,143,424, because the untied
# 16,384-token vocab contributes 8,388,608 params that do no matmul work. The
# token budget is set from the body count, not the stored count.
#
#   475,481,600 tokens = 100 x 4,754,816
#   0.059 epochs of the 8B corpus, so every token is fresh.
#
# Throughput: ~197k tok/s measured on synthetic batches (scripts/bench_throughput.py),
# but that measurement excludes the dataloader, and at this size
# PackedTokenStream.batch's per-row Python loop is expected to bind first. Treat
# ~2.5-3.5h as the estimate and check the first logged tokens_per_second.
#
# Warmup is 435 updates (3% of 14,510), NOT the trainer's default 2000. At
# 32,768 tokens/update this run is short enough that the default would spend
# 13.8% of it warming up.
#
# --planned-total-tokens is set equal to --target-tokens so the cosine decays
# across this run. Left unset it defaults to 250M, and the schedule would hit
# the LR floor at 53% and crawl through the rest.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

# Throughput is not the goal here, but a contended GPU still distorts the run
# and risks OOM against a resident llama-server.
if pgrep -f '^[^ ]*python[0-9.]* -m modern_lm\.(train|sft)' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

exec $PY -m modern_lm.train \
  --target-tokens 475481600 \
  --planned-total-tokens 475481600 \
  --run-dir runs/size5m-100x \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 256 --n-layers 5 --n-heads 4 --n-kv-heads 4 --ffn-dim 896 \
  --checkpoint-tokens 20000000 \
  --microbatch-size 16 --gradient-accumulation 4 \
  --optimizer muon \
  --muon-learning-rate 0.005 \
  --learning-rate 3e-4 \
  --warmup-updates 435 \
  --log-every 10 \
  --seed 2026
