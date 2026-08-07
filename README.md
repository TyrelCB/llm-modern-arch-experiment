# ModernLM: a capacity-matched dense baseline for the DeepSeek-V4 reference

A modern dense decoder-only Transformer (RMSNorm, RoPE, SwiGLU, GQA, QK-norm)
sized to the **same stored parameter count** as the DeepSeek-V4 145M math model,
trained on the **same tokens in the same order** and scored by the **same
benchmark code**, to measure the quality/time trade-off between the two
architectures.

The companion repository is `../llm-deepseek-v4-experiment`. This one does not
duplicate its corpus, tokenizer, split, SFT data, or scorer — it imports them,
because a re-derived tokenizer or a second copy of a scorer would make the
comparison illegitimate rather than merely inconvenient.

## Results at a glance

Eight arms, all scored by the reference's own scorer at the same greedy
32-token budget. **The evaluation harness was never changed**, so every row is
directly comparable.

| Arm | Pretrain loss | Benchmark | ASDiv | GSM8K | Algebra | Arithmetic |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4 pretrain (250M tok) | 2.5514 | 56 (1.115%) | 0.95% | 1.36% | 1.00% | 0.00% |
| DeepSeek-V4 + SFT | — | 116 (2.309%) | 3.08% | 1.67% | 5.00% | 1.00% |
| ModernLM pretrain (250M tok) | **2.4049** | 95 (1.891%) | 2.17% | 1.67% | 0.00% | 0.00% |
| ModernLM 250M + SFT | — | 163 (3.244%) | 3.69% | 2.73% | 11.00% | 2.00% |
| ModernLM pretrain (2B tok) | **2.0416** | 115 (2.289%) | 2.95% | 1.82% | 1.00% | 1.00% |
| ModernLM 2B + SFT | — | 412 (8.201%) | 9.24% | **4.25%** | 34.00% | 11.67% |
| ModernLM 2B + concise SFT | — | 473 (9.415%) | 10.46% | 3.18% | **37.00%** | 12.33% |
| ModernLM 2B + concise + arithmetic | — | 497 (9.893%) | 10.98% | 3.56% | 35.00% | **14.67%** |
| **ModernLM 2B + concise + arithmetic + number words** | — | **568 (11.306%)** | **13.28%** | 3.49% | 32.00% | 13.00% |

An eighth arm, GRPO, was pre-registered and produced a negative result: at the
registered settings 95.3% of rollout groups carry zero gradient, so the
configuration cannot learn. See [`docs/results-grpo.md`](docs/results-grpo.md).

Detailed write-ups: [`docs/results.md`](docs/results.md) (250M head-to-head),
[`docs/results-2b.md`](docs/results-2b.md) (2B run),
[`docs/results-sft.md`](docs/results-sft.md) (SFT),
[`docs/results-sft-250m.md`](docs/results-sft-250m.md) (decomposition),
[`docs/results-sft-concise.md`](docs/results-sft-concise.md) (concise SFT),
[`docs/results-sft-augmented.md`](docs/results-sft-augmented.md) (arithmetic
augmentation),
[`docs/results-sft-number-words.md`](docs/results-sft-number-words.md)
(spelled-out operands).

The three SFT-data arms are cumulative: **412 → 568, +37.9%**, at no cost in
model size, pretraining, or decode budget — the model and the harness are
untouched, only what the model is shown during SFT changed. Paired McNemar
against the baseline: p = 4.05e-09. Per-arm increments: concise +61
(p = 0.017), arithmetic +24 (p = 0.30, not significant), number words +71
(p = 0.0021).

### 1. At matched capacity, the modern dense stack wins on quality *and* time

Both models hold ~144.6M stored parameters and train on identical tokens in
identical order (test-verified). At 250M tokens:

| | DeepSeek-V4 (MoE) | ModernLM (dense) |
|---|---:|---:|
| Stored / active params | 144,669,412 / 45,578,980 | 144,630,912 / 144,630,912 |
| Held-out loss | 2.5514 | **2.4049** (−5.74%) |
| Perplexity | 12.82 | **11.08** |
| Throughput | 8,600 tok/s | **35,975 tok/s** |
| Wall clock, 250M tokens | ~484 min | **116 min** |

ModernLM reached the reference's *final* loss after 170M tokens in 78.8 minutes
— a **6.1x time-to-quality speedup**. There was no quality/time trade-off to
negotiate: it was faster and better.

Note this is achieved *despite* a per-token compute disadvantage — capacity
matching means the dense model spends 3.2x the active parameters per token. The
reference is also an explicitly unoptimized correctness implementation (~3% MFU
by its own profiling), so the throughput gap measures these two implementations,
not MoE versus dense in general.

### 2. The 145M model was under-trained, not capability-limited

At 250M tokens both models sit at 1.73 tokens/parameter. Training to 2B (13.83
tokens/param, ~1 epoch over a freshly packed 2.05B-token corpus) improved
held-out loss 2.4049 → **2.0416** (perplexity 11.08 → 7.70) in 15.15 h, and the
2B run passed the reference's *final* loss at just 250M tokens.

### 3. SFT teaches how to answer; pretraining decides whether it's right

This is the clearest finding in the project, and it is only visible by reading
completions — the accuracy table alone hides it. Both SFT'd models learn the
output format perfectly and identically. They differ in whether the arithmetic
inside that format is correct:

```
250M + SFT:  "Add the two numbers: 48 + 22 = 69.  Final answer: 69"   (wrong)
2B   + SFT:  "Add the two numbers: 48 + 22 = 70.  Final answer: 70"   (right)
```

Decomposing the 412 result:

| | Change | Multiplier |
|---|---|---:|
| SFT on the 250M base | 95 → 163 | 1.72x |
| SFT on the 2B base | 115 → 412 | **3.58x** |
| More pretraining, no SFT | 95 → 115 | 1.21x |
| More pretraining, with SFT | 163 → 412 | **2.53x** |

The levers **compound**. Longer pretraining alone buys only 1.21x because the
capability is masked; SFT alone buys 1.72x. Together, 4.34x.

### 4. A measurement artifact, found by reading output rather than metrics

The 2B pretrained model answered correctly and then kept generating, running on
into self-generated `Question:` blocks — so the scorer picked up the wrong
number. 34.2% of its completions were affected.

SFT supervises `<eos>` on every response, and run-on drops to **0.0%**. The
artifact closed **as a model improvement, not a scorer change**: the harness was
never touched. Numeric completion rate went 85.0% → 99.6%.

The matched-budget architecture comparison is the 250M+SFT arm (163) against
DeepSeek-V4+SFT (116) — same token budget, same SFT recipe, differing only in
architecture: **1.41x**. The 3.6x headline bundles the 8x token budget and
should not be read as an architecture claim.

### 5. At this scale, every extra reasoning step *costs* accuracy

Finding 4 closed run-on at the pretraining stage. A subtler version of it
survived SFT: the model reached a correct result and then appended one more
step that overwrote it. Scored at 256 tokens, where 95% of completions
terminate on their own, accuracy falls monotonically in lines emitted —
**22.00%** at one line, 6.57% at two, 3.87% at three, ~2% beyond.

The waste this represents is large. Counting the gold answer appearing
*anywhere* in a completion, the 412 model's oracle rate is 17.93% against 8.20%
scored: it computed the right number two to three times more often than it was
credited for. It also explains why a 256-token budget scored *worse* than 32
(358 < 412) — the short budget was accidentally truncating completions before
they could go wrong.

Rewriting the SFT target to "compute, state the answer, stop" — same model,
same scorer, same budget, only the supervision changed — gives **412 → 473**:

| | Baseline SFT | Concise SFT |
|---|---:|---:|
| Emitting 1 reasoning line | 978 | **4,921** |
| Reached `Final answer:` in 32 tokens | 25.1% | **90.4%** |
| Oracle (answer anywhere) | **17.93%** | 14.49% |
| **Scored** | 8.20% | **9.41%** |

**Oracle fell while accuracy rose.** The baseline's extra oracle hits were not
latent capability — they were correct results being buried under a spurious
final step. Chain-of-thought is a capability of scale, and below that scale
it is a liability: GSM8K, the benchmark most dependent on genuine multi-step
decomposition, is the one arm that regressed (56 → 42).

The control rules out "shorter targets are just better". Keeping the last
**two** reasoning lines instead of one — same script, same grounding rule, same
hyperparameters — scores **361**, below even the untouched baseline:

| SFT target | Benchmark |
|---|---:|
| Full chain (1–8 lines) | 412 |
| **1 line** | **473** |
| 2 lines | 361 |

The effect is not monotone in target length. The second line is specifically
where this model starts overwriting a result it had already computed.

### 6. The largest single win came from reading the model's mistakes

After the concise rewrite and an arithmetic-coverage pass, 1,607 errors
remained where the model picked the right operation and computed correctly from
the *wrong operands*. Reading them, one pattern dominated — **the model cannot
read a number word**:

```
"Seven red apples and two green apples ..."  → "7 + 5 = 12"   (gold 9)
"Sandra took six cups, Marcie took two ..."  → "12 + 6 = 18"   (gold 8)
"Brian has four more plums than Paul..."     → "8 + 7 = 15"    (gold 11)
```

The structure is right every time. It knows to add and which quantities to
combine; it just doesn't know "seven" is 7, so it substitutes a plausible digit
and computes flawlessly with the wrong number. Spelled-out questions scored
**8.38%** against **10.32%** for digit-only ones — and only 7.3% of training
questions contained a number word, while ASDiv and SVAMP use them constantly.

4,000 records in three templates, questions spelling their operands and
responses restating them as digits, closed it:

| Evaluation questions | Before | After |
|---|---:|---:|
| Spelled-out numbers (n=1,098) | 8.38% | **11.48%** |
| Digits only (n=3,926) | 10.32% | **11.26%** |

The 1.94-point penalty on spelled-out questions became a 0.22-point advantage,
and digit questions improved too. Benchmark **497 → 568** (p = 0.0021).

Extending the same trick further **stopped working**: three more templates for
"twice", "half" and the tens words moved 568 → 575 (p = 0.78). The targeted
deficit did narrow, but algebra, SVAMP and GSM8K paid for it almost exactly —
shifting yet more corpus mass toward single-step word problems has stopped
being free. That is the diminishing-returns boundary for this approach, and
568 is the arm to use.

Cumulatively the three SFT-data arms take **412 → 568 (+37.9%)** with no change
to the model, the pretraining, the decode budget, or the scorer. For scale, the
8x pretraining increase bought more in absolute terms (163 → 412, +249) — but
it cost 15.15 h of training, where these three arms cost about 13 minutes
total. Reading the model's own errors was by far the cheaper lever, and it was
available the whole time.

## Architecture

| Component | Choice | Replaces |
|---|---|---|
| Normalization | RMSNorm (pre-norm) | LayerNorm |
| Position encoding | RoPE, theta 10,000 | Learned absolute embeddings |
| Feed-forward | SwiGLU, ffn_dim 2,432 | 4x GELU MLP |
| Attention | Causal GQA via SDPA, `n_kv_heads` configurable | MHA |
| Attention stability | QK-norm (RMSNorm on Q and K) | — |
| Biases | None | Linear biases |
| Output head | Untied | — |

`ModernConfig.dense_145m()`: `dim=768`, `n_layers=15`, `n_heads=12`,
`ffn_dim=2432`, `vocab_size=16384`, `max_seq_len=512` → **144,630,912
parameters**, within 0.027% of the reference's 144,669,412.

`n_kv_heads` defaults to `n_heads` (full MHA). GQA's payoff is inference KV
memory, not training throughput, so enabling it would change capacity without a
matching benefit to measure here.

### Staged levers, off by default

MTP (`use_mtp`) and MoE (`use_moe`) are implemented and tested but disabled, so
the comparison isolates the dense modern stack. Enabling either changes two
variables at once against the reference. Measured cost if enabled: MTP 31,479
tok/s (0.85x) and resumable from an existing checkpoint; MoE 28,438 tok/s
(0.77x) and **not** resumable — it replaces every feed-forward, so it needs a
fresh run.

## Layout

```
src/modern_lm/
  config.py               ModernConfig, dense_145m() and tiny() presets
  layers.py               RMSNorm, RoPE, SwiGLU, GQA attention, MoE
  model.py                ModernLM, MTPHead, generate()
  data.py                 PackedTokenStream — port of the reference sampler
  train.py                Resumable pretraining loop
  sft.py                  Supervised fine-tuning
  evaluate_benchmarks.py  Scores checkpoints with the reference's own scorer
  compare.py / report.py  Gate arithmetic and results rendering
scripts/prepare_2b_corpus.py   Packs 2.05B tokens with the reference tokenizer
tests/                    33 tests, CPU-only
docs/                     Protocol, corpus provenance, and per-run results
```

## Usage

```bash
python -m pytest tests/ -q

# Pretrain (schedule planned for the full budget from step 0)
python -m src.modern_lm.train \
  --target-tokens 2000000000 --planned-total-tokens 2000000000 \
  --run-dir runs/modern-145m-2b \
  --microbatch-size 16 --gradient-accumulation 4 \
  --checkpoint-tokens 50000000 --keep-last-checkpoints 3 --device cuda

# SFT: gate at 100 updates, then continue on the same 1000-update schedule
python -m src.modern_lm.sft \
  --checkpoint runs/modern-145m-2b/latest.pt \
  --run-dir runs/modern-145m-2b-sft \
  --target-updates 100 --planned-total-updates 1000

python -m src.modern_lm.evaluate_benchmarks \
  --checkpoint runs/modern-145m-2b-sft/latest.pt \
  --output runs/modern-145m-2b-sft/evaluation.jsonl \
  --max-new-tokens 32 --device cuda
```

### Reproducing the 568 arm

The SFT-data arms differ from the baseline only in the corpus they are given.
Each build is deterministic and each SFT run takes about three minutes on a
GB10. The derived corpora are gitignored (`data/`); these commands regenerate
them byte-identically.

```bash
# 1. Concise targets: one reasoning line, then stop        -> 473
python scripts/prepare_concise_sft.py --keep-lines 1 \
  --output-dir data/sft-math-concise

# 2. Arithmetic coverage for the empty magnitude bucket    -> 497
python scripts/augment_arithmetic_sft.py --count 6000 --seed 2029 \
  --exclude data/sft-math-concise/heldout.jsonl \
  --output data/sft-math-concise-aug/train.jsonl
python scripts/augment_arithmetic_sft.py --count 300 --seed 2030 \
  --base data/sft-math-concise/heldout.jsonl \
  --exclude data/sft-math-concise-aug/train.jsonl \
  --output data/sft-math-concise-aug/heldout.jsonl

# 3. Spelled-out operands                                  -> 568
python scripts/augment_arithmetic_sft.py --number-words --count 4000 --seed 2031 \
  --base data/sft-math-concise-aug/train.jsonl \
  --exclude data/sft-math-concise-aug/heldout.jsonl \
  --output data/sft-math-words/train.jsonl
python scripts/augment_arithmetic_sft.py --number-words --count 200 --seed 2032 \
  --base data/sft-math-concise-aug/heldout.jsonl \
  --exclude data/sft-math-words/train.jsonl \
  --output data/sft-math-words/heldout.jsonl

python -m src.modern_lm.sft \
  --checkpoint runs/modern-145m-2b/latest.pt \
  --run-dir runs/modern-145m-2b-sft-words \
  --train data/sft-math-words/train.jsonl \
  --heldout data/sft-math-words/heldout.jsonl \
  --target-updates 1000 --planned-total-updates 1000
```

The `--exclude` flags are load-bearing, not decoration: the generators sample a
small template space, so without them a question can land in both train and
heldout (or reproduce one that exists only in the other split) even under
different seeds. All three split pairs are verified to have zero overlap.

Both corpora above were checked to rebuild byte-identically from these exact
commands. The twice/half/tens templates that produced the 575 null result are
behind `--extended-words` (seed 2033, count 5000) so this recipe keeps
reproducing the recommended arm.

Training and SFT are resumable; checkpoints carry model, optimizer, and
Python/NumPy/CPU/CUDA RNG state, loaded with `map_location="cpu"` so RNG
ByteTensors stay where `set_rng_state` needs them.

## How the comparison is kept honest

- **Same tokens in the same order.** `PackedTokenStream` is a port of the
  reference sampler; tests assert identical block indices and token tensors.
- **Same tokenizer**, sha256-verified. Retraining BPE would shift every token id
  and make loss incomparable.
- **Same scorer**, imported rather than copied.
- **Same split seed.** The 2B corpus expansion was verified to leak zero
  documents in all three directions, including against the 250M run's held-out
  set (the split hashes document keys, so held-out stays held out).
- **Pre-registered gates** in `docs/comparison-protocol.md`, fixed before
  results landed. When the scorer artifact was found, the registered numbers
  stayed the headline and the artifact was recorded as a limitation rather than
  retroactively "fixed."

## Limitations

- **Single seed per arm.** Every result here is exploratory, not a settled
  ranking. Three seeds would be needed for a decision-grade claim.
- **8.2% is not competence.** The best model still fails ~92% of the suite.
- **Capacity-matched, not compute-matched** — a compute-matched comparison is a
  different experiment.
- MTP and MoE are untested arms; part of the architecture gap may belong to
  them rather than to RoPE/SwiGLU/RMSNorm.

## Data and licensing

No corpus is redistributed here. Training data is FineMath-4+ (ODC-By 1.0) as
packed by the companion repository; benchmarks are GSM8K (MIT), SVAMP (MIT), and
ASDiv (CC BY-NC 4.0). DeepSeek names and configurations belong to their owners.
