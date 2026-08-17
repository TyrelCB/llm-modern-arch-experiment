# Comparison protocol: modern dense vs DeepSeek-V4 reference

> **Historical Phase I protocol.** This document preserves the rules registered
> for the original comparison. The current project-wide validation contract is in
> [`PROJECT_MEMORY.md`](../PROJECT_MEMORY.md). In particular,
> [`D002`](decisions.md#d002) supersedes routine multi-seed requirements with a
> canonical-trajectory and scale/budget-transfer policy.

Pre-registered before the training run finished. Recording it up front is the
point — this project's sibling repositories have twice seen a metric pass an
exploratory gate and fail on confirmation.

## The question

Does a modern dense architecture (RMSNorm + RoPE + SwiGLU + QK-norm), at
**equal stored parameter count**, reach the DeepSeek-V4 reference's pretraining
quality in less wall-clock time on the same data?

This is a quality/time trade-off question. Throughput alone does not answer it:
a model that is 4x faster per token but learns proportionally less per token has
gained nothing.

## What is held fixed

Everything except the architecture:

| Held fixed | Value |
|---|---|
| Corpus | FineMath-4+ shards 0/16/32/48, packed `train.bin` (260M tokens) |
| Tokenizer | The reference's 16,384-token byte-level BPE, byte-for-byte |
| Held-out set | The reference's `heldout.bin` (5M tokens), same decontaminated split |
| Token order | Same seed (2026), same LCG block permutation — verified by test |
| Sequence length | 512 |
| Tokens per optimizer update | 32,768 (microbatch 16 x accumulation 4) |
| Optimizer | AdamW, wd 0.1, grad clip 1.0 |
| LR schedule | 3e-4, 2,000 warmup updates, cosine to 3e-5, planned against 250M |
| Precision | bf16 autocast, fp32 params/optimizer |
| Budget | Exactly 250,000,000 supervised target tokens |
| Benchmark scoring | The reference's own `extract_number` / `numeric_equal`, imported |
| Decoding | Greedy, `max_new_tokens=32`, `Question: ...\nAnswer:` |

`tests/test_data_and_train.py` asserts the packed-stream block order and the
returned token tensors are identical to the reference's sampler. Without that,
this is not a controlled comparison.

## What differs (the independent variable)

| | DeepSeek-V4 reference | ModernLM |
|---|---|---|
| Normalization | RMSNorm | RMSNorm |
| Position | Partial RoPE + inverse RoPE on output | RoPE (full, standard) |
| Attention | Learned KV compression (CSA/HCA), Lightning Indexer, shared-KV MQA, sink, sliding window | Dense causal GQA (MHA by default) + QK-norm |
| Residual | mHC multi-head hyper-connections, Sinkhorn mixing | Plain pre-norm residual |
| FFN | DeepSeekMoE: 16 routed + 1 shared expert, top-2 | Dense SwiGLU |
| MTP | 1 MTP layer, weight 0.1 | Off (staged) |
| Stored params | 144,669,412 | 144,630,912 (-0.027%) |
| Active params/token | 45,578,980 | 144,630,912 |

**Capacity is matched on stored parameters, per the goal.** Note this makes the
comparison deliberately *unfavorable* to ModernLM on compute: it spends ~3.2x
the active parameters per token. If it still wins on wall clock, the result is
strong; if it wins only on loss-per-token, that must be reported as such.

## Pre-registered gates

Recorded before the 250M-token run completed.

1. **Time-to-quality (primary).** ModernLM wins if it reaches the reference's
   final held-out main loss of **2.55136** in less wall-clock time. The
   reference took ~7.0 h of training for 250M tokens (250M / 9,720 tok/s).
2. **Quality at equal tokens (secondary).** At exactly 250M tokens, compare
   held-out main loss directly. Same tokenizer and held-out set, so this is a
   legitimate comparison — unlike the cross-corpus comparison the sibling repo's
   17f entry warns about.
3. **Benchmark accuracy (secondary).** Overall numeric exact match against the
   reference's pretrained-checkpoint baseline of **56/5,024 (1.115%)**, scored
   by the same code at the same 32-token budget. No SFT on either side.

### Interpretation rules fixed in advance

- Both models are at ~1.7 tokens per stored parameter. This is a *severely
  under-trained* regime; the reference's own documentation says so. Benchmark
  accuracy near 1-2% is expected for both and is **not** a capability claim.
  Small differences there are noise, not architecture wins.
- Held-out loss is the reliable signal at this budget; benchmark exact-match is
  reported for completeness because the goal asked for the exact benchmarks.
- This is a **single seed** (2026) on both sides. Per the sibling repo's
  three-seed rule, that makes any behavioral conclusion exploratory. A loss gap
  of a few percent at one seed is not a settled architecture result.
- Read actual generated completions before trusting the percentages.

## Staged follow-ups (not run here)

MTP and MoE are implemented and flag-gated but **off**, so the first comparison
isolates the dense modern stack. Turning either on changes two variables at once
against the reference and would confound attribution.

| Arm | Flag | Question |
|---|---|---|
| D | `use_mtp=True` | Does MTP help as a training regularizer at this scale? |
| E | `use_moe=True` | Does sparsity pay at ~1.7 tokens/param? |
