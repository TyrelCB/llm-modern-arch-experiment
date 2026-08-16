#!/usr/bin/env bash
# 50M @ 20 tpp, WSD schedule, SiameseNorm residual -- the architecture arm of
# scripts/run_50m_20x_wsd.sh.
#
# EVERY flag below is identical to that script except --siamese-norm. Same
# shape, same 1,046,493,440 tokens, same seed 2026, same corpus, same Muon LR,
# same warmup, same WSD decay fraction and zero floor. The residual topology is
# the only variable:
#
#     cosine   loss 2.297   2.09% (105/5024)
#     wsd      loss 2.311   <- the schedule arm, already measured
#     siamese  <- this run
#
# SiameseNorm (arXiv 2602.08064, Alibaba) carries two residual streams over one
# shared block body: Y accumulates unnormalized (Pre-LN's identity gradient
# path) while X is re-normalized after every add (Post-LN's bounded path), with
# the update into X scaled by 1/sqrt(l+1) and a final fusion X_N + LN(Y_N).
# Their sub-block is HybridNorm with a learnable gamma on the attention input;
# both are kept here rather than testing the two-stream residual alone.
#
# gamma initializes to 1.0, which makes the attention input exactly the Pre-LN
# input at step 0, so the model starts at the baseline and learns its way off.
# The 0.02/sqrt(2L) residual-output init is DISABLED on this path: it is a
# Pre-LN depth remedy for the same problem the 1/sqrt(l+1) divisor solves
# structurally, and the paper is explicit about initializing norm scales to 1.0
# without Pre-LN-biased init. Keeping both would confound an init change with
# the architecture change.
#
# WHAT THIS RUN CAN AND CANNOT SETTLE. The paper's effect is size- and
# LR-dependent: 1.3B models, AdamW, and a margin that grows with learning rate
# (their Pre-LN baseline diverges at 1e-3 where SiameseNorm holds). This is 50M
# with Muon at 0.005, which already controls per-layer update scale via
# orthogonalization -- so a null here is evidence about THIS rung and THIS
# optimizer, not a refutation of the paper. Their honest effect size is the
# perplexity column (10.89 -> 10.48, ~0.04 nats), which is SMALLER than the
# Muon loss win that bought no benchmark capability (docs/results-muon.md).
# Single seed, so treat a sub-0.02-nat difference as noise, not a result.
#
# Watch for their stated limitation: they report more pronounced "massive
# activations" than the baseline. This repo has been bitten by healthy-looking
# loss masking dead capability before (the CPT reheat), which is why the arm is
# gated on a post-SFT benchmark rather than on pretrain loss.
#
# Body params ~52.3M plus 4 RMSNorm gains and one gamma vector per layer, and
# one final Y-stream norm -- 1D params only, so the Muon/AdamW split is
# unchanged (all of them route to AdamW like every other norm gain).
#
# THROUGHPUT: measured -7.7% against Pre-LN at this shape (24,397 vs 26,440
# tok/s microbenched). The paper calls its overhead "negligible"; that is true
# of FLOPs and false of wall clock here, because the cost is four extra RMSNorm
# passes over the residual stream per layer plus a second [B,T,D] tensor
# carried through every block -- bandwidth and launch bound, not FLOP bound.
# Budget ~5.3h rather than the WSD arm's 4.9h.
set -uo pipefail
cd /home/tyrel/projects/llm-modern-arch-experiment
export PYTHONPATH=src
PY=/home/tyrel/projects/llm-deepseek-v4-experiment/.venv/bin/python
D=/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b

# Match the python interpreter running the module, not any shell whose command
# line merely contains the string -- a wrapper or a pgrep itself would otherwise
# trip this guard and abort a legitimate launch.
if pgrep -f '^[^ ]*python[0-9.]* -m modern_lm\.(train|sft)' > /dev/null 2>&1; then
  echo "ABORT: a training job is already running"; exit 1
fi

exec $PY -m modern_lm.train \
  --target-tokens 1046493440 \
  --planned-total-tokens 1046493440 \
  --run-dir runs/size50m-20x-siamese \
  --train "$D/train.bin" --heldout "$D/heldout.bin" \
  --dim 576 --n-layers 11 --n-heads 9 --n-kv-heads 9 --ffn-dim 1984 \
  --siamese-norm \
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
