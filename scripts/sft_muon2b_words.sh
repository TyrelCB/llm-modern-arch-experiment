#!/usr/bin/env bash
# The 568 SFT arm (concise + arithmetic + number words) applied to the Muon 2B
# base, which has never been given it -- the recipe was developed on the AdamW
# 2B model (README: 568/5024 = 11.31%) while the Muon base only ever got the
# default SFT corpus (465/5024 = 9.26%).
#
# Same hyperparameters as the recorded arm so the only difference is the base
# model: seed 2027, 1000 updates, lr 5e-5.
set -u
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python

$PY -m modern_lm.sft \
  --checkpoint runs/muon-2b-lr0.005/checkpoint-002000000000.pt \
  --run-dir runs/muon-2b-sft-words \
  --train data/sft-math-words/train.jsonl \
  --heldout data/sft-math-words/heldout.jsonl \
  --target-updates 1000 --planned-total-updates 1000
rc=$?
echo "[$(date '+%T')] sft rc=$rc"
[ $rc -eq 0 ] || exit $rc

# Score at 96 tokens with the run-on fix, matching the 6.61% base anchor.
$PY -m modern_lm.evaluate_benchmarks \
  --checkpoint runs/muon-2b-sft-words/latest.pt \
  --output runs/curve-cpt/sft-words-96tok.jsonl \
  --max-new-tokens 96 --device cuda
echo "[$(date '+%T')] DONE"
