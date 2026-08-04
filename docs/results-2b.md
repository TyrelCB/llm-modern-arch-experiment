# Results: modern dense vs DeepSeek-V4 reference

Both models trained on identical tokens in identical order (verified by
test), same tokenizer, same held-out set, same optimizer settings, same
250,000,000-token budget, single seed 2026, no SFT on either side.

## Headline

| | DeepSeek-V4 145M (MoE) | ModernLM 145M (dense) |
|---|---:|---:|
| Stored parameters | 144,669,412 | 144,630,912 |
| Active parameters / token | 45,578,980 | 144,630,912 |
| Final held-out loss | 2.5514 | **2.0416** |
| Held-out perplexity | 12.8245 | **7.7030** |
| Throughput (tok/s) | 8,600 (as run) / 9,720 (best) | **36,659** |
| Wall clock for 250M tokens | 3876.0 min (as run) | **909.3 min** |

## Pre-registered gates

**Gate 2 — quality at equal tokens.** ModernLM final held-out loss 2.0416 vs reference 2.5514: **-0.5097** (-19.98%).

**Gate 1 — time to quality (primary).** ModernLM first reached the reference's *final* loss (2.5514) after **250,019,840 tokens** in **113.6 minutes**. The reference needed 250,000,000 tokens and 3876.0 minutes to get there.

That is a **34.1x** time-to-quality speedup against the reference as actually run, or **30.2x** against its best measured throughput.

**Same-token wall-clock speedup:** 4.26x vs the reference as run, 3.77x vs its best measured throughput.

## Loss trajectory

| Tokens | ModernLM | DeepSeek-V4 | Delta |
|---:|---:|---:|---:|
| 50,003,968 | 3.3329 | 3.8770 | -0.5442 |
| 100,007,936 | 2.8462 | 3.0386 | -0.1924 |
| 150,011,904 | 2.6758 | 2.7514 | -0.0756 |
| 200,015,872 | 2.5923 | 2.6093 | -0.0170 |
| 250,019,840 | 2.5264 | 2.5514 | -0.0250 |
| 300,023,808 | 2.4791 | 2.5514 | -0.0723 |
| 350,027,776 | 2.4381 | 2.5514 | -0.1132 |
| 400,031,744 | 2.4087 | 2.5514 | -0.1427 |
| 450,002,944 | 2.3818 | 2.5514 | -0.1696 |
| 500,006,912 | 2.3573 | 2.5514 | -0.1941 |
| 550,010,880 | 2.3430 | 2.5514 | -0.2083 |
| 600,014,848 | 2.3237 | 2.5514 | -0.2277 |
| 650,018,816 | 2.3063 | 2.5514 | -0.2450 |
| 700,022,784 | 2.2949 | 2.5514 | -0.2564 |
| 750,026,752 | 2.2764 | 2.5514 | -0.2749 |
| 800,030,720 | 2.2601 | 2.5514 | -0.2912 |
| 850,001,920 | 2.2457 | 2.5514 | -0.3057 |
| 900,005,888 | 2.2340 | 2.5514 | -0.3173 |
| 950,009,856 | 2.2194 | 2.5514 | -0.3319 |
| 1,000,013,824 | 2.2021 | 2.5514 | -0.3492 |
| 1,050,017,792 | 2.1929 | 2.5514 | -0.3585 |
| 1,100,021,760 | 2.1809 | 2.5514 | -0.3705 |
| 1,150,025,728 | 2.1659 | 2.5514 | -0.3855 |
| 1,200,029,696 | 2.1579 | 2.5514 | -0.3934 |
| 1,250,000,896 | 2.1476 | 2.5514 | -0.4037 |
| 1,300,004,864 | 2.1345 | 2.5514 | -0.4168 |
| 1,350,008,832 | 2.1255 | 2.5514 | -0.4259 |
| 1,400,012,800 | 2.1161 | 2.5514 | -0.4353 |
| 1,450,016,768 | 2.1042 | 2.5514 | -0.4472 |
| 1,500,020,736 | 2.0958 | 2.5514 | -0.4555 |
| 1,550,024,704 | 2.0872 | 2.5514 | -0.4642 |
| 1,600,028,672 | 2.0812 | 2.5514 | -0.4701 |
| 1,650,032,640 | 2.0725 | 2.5514 | -0.4789 |
| 1,700,003,840 | 2.0649 | 2.5514 | -0.4865 |
| 1,750,007,808 | 2.0607 | 2.5514 | -0.4906 |
| 1,800,011,776 | 2.0554 | 2.5514 | -0.4960 |
| 1,850,015,744 | 2.0509 | 2.5514 | -0.5004 |
| 1,900,019,712 | 2.0483 | 2.5514 | -0.5030 |
| 1,950,023,680 | 2.0454 | 2.5514 | -0.5060 |
| 2,000,000,000 | 2.0416 | 2.5514 | -0.5097 |

The reference column is its `best_heldout_main_loss` at each recorded checkpoint. At intermediate milestones that is a running best rather than a point-in-time evaluation; the 250M endpoint is the clean comparison.

## Benchmarks (no SFT, greedy, 32 new tokens)

| Benchmark | DeepSeek-V4 | ModernLM |
|---|---:|---:|
| asdiv | 22 / 2305 (0.954%) | 68 / 2305 (2.950%) |
| svamp | 15 / 1000 (1.500%) | 19 / 1000 (1.900%) |
| gsm8k | 18 / 1319 (1.365%) | 24 / 1319 (1.820%) |
| algebra | 1 / 100 (1.000%) | 1 / 100 (1.000%) |
| arithmetic | 0 / 300 (0.000%) | 3 / 300 (1.000%) |
| **Overall** | **56 / 5024 (1.115%)** | **115 / 5024 (2.289%)** |

**These numbers are not a capability claim for either model.** Both sit at ~1.7 tokens per stored parameter, far below what matures a small language model, and sampled completions from both are degenerate (repetition loops, answers unrelated to the question, frequently no number at all). Differences of a few tenths of a percent here are noise. Held-out loss is the reliable signal at this budget.

## Scope and limits

- **Single seed (2026) on both sides.** This is an exploratory result, not a settled architecture ranking. A three-seed replication would be needed before treating the loss gap as decision-grade.
- **Capacity-matched, not compute-matched.** ModernLM spends ~3.2x the active parameters per token (144.6M dense vs 45.6M active). The wall-clock win is therefore achieved *despite* a per-token compute disadvantage, but a compute-matched comparison would be a different experiment.
- **MTP and MoE are off** in ModernLM, and the reference used both. Part of any gap may be attributable to those, not to RoPE/SwiGLU/RMSNorm. The staged arms exist to separate that.
- The reference implementation is an explicitly unoptimized clean-room correctness reference (~3% MFU, memory-bound on FP32 elementwise traffic by its own profiling). The throughput gap measures these two implementations, not MoE versus dense in general.

## Scorer artifact: the registered number understates this model

The 250M model's "correct" answers were operand copies. At 2B that is no
longer the whole story -- the model now performs visible arithmetic:

| Prompt | Completion |
|---|---|
| "Ellen has six more balls than Marin. Marin has nine balls..." | "6 + 9 = 15" |
| "Janet has nine oranges and Sharon has seven oranges..." | "9 + 7 = 16" |
| "Calculate 48 + 22." | "48 + 22 = 70" |

All three are correct, and the last has no operand to copy.

**But the registered scorer marks the third one wrong.** The model has no stop
condition after answering, so it continues into a fresh self-generated
"Question: ..." block, and `extract_number` takes the last number in the 32-token
window rather than the answer:

```
Q: Calculate 11 + 3.   gold 14   scored prediction 3
completion: " 11 + 3 = 14\n\nQuestion: Calculate 5 + 2.\nAnswer: 5 + 2 = 7\n\nQuestion: "
```

The answer `14` is right there. Scoring the first line only:

| Slice | As registered | First line only |
|---|---:|---:|
| Arithmetic | 3 / 300 (1.00%) | **14 / 300 (4.67%)** |
| Overall | 115 / 5,024 (2.29%) | **165 / 5,024 (3.28%)** |

**The headline table above keeps the registered numbers**, because changing the
scorer after seeing results is exactly how a comparison stops being honest, and
the 250M and reference runs were scored the same way. This section is recorded
as a measurement limitation, not a restatement of the result. A future run that
wants a clean read should add an `\n\nQuestion:` stop sequence to generation --
decided and registered *before* the run, applied to every arm.

Note the artifact suppresses all three arms, so the relative ranking stands; it
is the absolute level that is understated.
