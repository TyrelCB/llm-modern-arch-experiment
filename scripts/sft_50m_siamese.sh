#!/usr/bin/env bash
# SFT + benchmark for the 50M SiameseNorm arm (scripts/run_50m_20x_siamese.sh).
#
# This is the ONLY evaluation of the arm. The base checkpoint is deliberately
# not benchmarked: base models at this rung are mostly echo (see
# docs/results-sft.md -- SFT lifts every rung 8-17x), so a base number would
# measure rambling, not capability, and would invite reading noise as signal.
#
# Every hyperparameter matches the WSD arm's SFT exactly, recovered from
# runs/sft-50m-wsd/checkpoint-000100.json: sft-math-words (23,780 examples),
# 1000 updates, lr 5e-5, seed 2027, and the trainer's own defaults for
# microbatch 16 / grad-accum 2 / warmup 100 / wd 0.01 / min-lr-ratio 0.1.
# So the base architecture is the only difference between the two numbers.
#
# The comparison, at the same 32-token greedy budget and the same 5,024
# questions, scored by the same scorer:
#
#     wsd (Pre-LN)      459/5024 = 9.14%   asdiv 11.02  gsm8k 3.49  alg 28  arith 6.67
#     siamese           <- this run
#
# Read it with the ladder's own error bars in mind. Per-arm SFT increments in
# the README run +24 (p=0.30, not significant) to +71 (p=0.0021) at 2B, and
# this rung is smaller and single-seed. A swing under ~30 questions is not a
# result; if it lands there, the honest report is "no detectable effect at 50M
# with Muon", which is a real and publishable outcome for this ladder.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python

BASE=runs/size50m-20x-siamese/latest.pt
if [ ! -f "$BASE" ]; then
  echo "ABORT: $BASE not found -- pretraining has not finished"; exit 1
fi
# Match the python interpreter running the module, not any shell whose command
# line merely contains the string -- a wrapper or a pgrep itself would otherwise
# trip this guard and abort a legitimate launch.
if pgrep -f '^[^ ]*python[0-9.]* -m modern_lm\.(train|sft)' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

$PY -m modern_lm.sft \
  --checkpoint "$BASE" \
  --run-dir runs/sft-50m-siamese \
  --train data/sft-math-words/train.jsonl \
  --heldout data/sft-math-words/heldout.jsonl \
  --target-updates 1000 --planned-total-updates 1000
rc=$?
echo "[$(date '+%T')] sft rc=$rc"
[ $rc -eq 0 ] || exit $rc

# 32 tokens: the budget behind the 459 anchor. Do not raise it here -- a larger
# budget changes the scored quantity and breaks comparability with the WSD arm.
$PY -m modern_lm.evaluate_benchmarks \
  --checkpoint runs/sft-50m-siamese/latest.pt \
  --output runs/eval-sft-50m-siamese.jsonl \
  --max-new-tokens 32 --device cuda
echo "[$(date '+%T')] DONE"
