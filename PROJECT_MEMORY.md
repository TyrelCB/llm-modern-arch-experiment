# ModernLM project memory

Last reconciled: **2026-08-18**

This is the short, living handoff for the project. It records current truth, not
the full experiment history. Stable decisions live in
[`docs/decisions.md`](docs/decisions.md); the current architecture lives in
[`docs/architecture.md`](docs/architecture.md) and its machine-readable companion
[`docs/architecture.json`](docs/architecture.json).

## Mission

Build an evidence-driven small-language-model optimization testbed that turns new
architecture, training, data, and systems findings into faithful implementations,
then keeps only changes that improve the **capability-cost Pareto frontier** under
controlled, reproducible measurement ([D001](docs/decisions.md#d001)).

The original capacity-matched dense-versus-DeepSeek study is complete historical
Phase I. It remains valuable evidence, but it no longer defines the repository's
scope.

## What “validated” means here

There are separate evidence lanes ([D003](docs/decisions.md#d003)):

- **Semantics-preserving systems changes:** demonstrate forward, loss, gradient,
  optimizer-step, and checkpoint-resume parity before measuring throughput,
  memory, energy, prefill, or decode performance.
- **Approximate numerical changes:** declare tolerances, test short-run divergence,
  and confirm the capability trajectory.
- **Learning changes:** hold the compute contract fixed and judge sustained
  time/tokens-to-capability, not a single loss or endpoint.
- **Evaluation changes:** version the scorer and rescore every compared baseline.

The project uses one canonical deterministic trajectory rather than routinely
funding multiple seeds ([D002](docs/decisions.md#d002)). Strong findings should be
large, persistent across checkpoints, and transfer to another scale or token
budget. Additional seeds are optional escalation for publication or a borderline,
high-impact decision—not a default gate.

## Current best

### Accepted architecture family

`dense-preln-v1` is the current accepted architecture
([D005](docs/decisions.md#d005)):

- Dense, decoder-only, single residual stream.
- Pre-RMSNorm attention and feed-forward sublayers.
- Bias-free Q/K/V/O projections, QK-norm, RoPE, causal PyTorch SDPA.
- Dense SwiGLU feed-forward.
- Final RMSNorm and an untied vocabulary head.
- MTP, MoE, and the local SiameseNorm branch are off.
- KV caching is an opt-in, output-equivalent inference optimization.

The exact graph, operation ordering, shapes, initialization, decision links, and
spin-off checklist are in [`docs/architecture.md`](docs/architecture.md).

### Capability champion — provisional

The highest recorded development-suite result is the 300M-body profile forked at
**5,280,006,144 pretraining tokens**, then trained for 1,000 standard
`sft-math-words` updates with SFT seed 2031 ([D026](docs/decisions.md#d026)):

| Field | Value |
|---|---:|
| Stored parameters | 329,821,696 |
| Transformer-body parameters | 296,267,264 |
| Shape | `D=1024, L=20, H=16, Hkv=16, FFN=3456` |
| Development benchmark | 887 / 5,024 (17.655%) |
| Numeric completion rate | 99.44% |
| SFT supervised tokens | 681,030 |

Artifacts: [pretraining metadata](runs/size300m-20x/checkpoint-005280006144.json),
[SFT metadata](runs/sft-300m-5280M-seed2031/checkpoint-001000.json), and
[evaluation summary](runs/eval-sft-300m-5280M-seed2031-step001000.summary.json).
The [full seed/checkpoint grid](runs/eval-sft-300m-5280M-seed-grid.summary.json)
is the authoritative selection context.

This is a **best observed checkpoint, not a validated recipe**. It was selected
from a 15-cell SFT seed/checkpoint grid on an adaptively used development suite.
At update 1,000, five seeds average 798.8 with sample standard deviation 68.1;
887 is selection-biased and does not establish that 5.28B pretraining tokens beat
4.77B. The parent pretraining run also remains incomplete.

### Working training recipe — provisional

- Pretraining uses bf16 autocast, 512-token sequences, 32,768 target tokens per
  optimizer update, gradient clipping at 1.0, and seed 2026.
- The size ladder uses hybrid Muon (`0.005`) for hidden 2-D matrices and AdamW
  (`3e-4`) for embeddings, norms, the vocabulary head, and other parameters.
  This is operationally useful but not a clean optimizer claim
  ([D012](docs/decisions.md#d012)).
- Cosine decay remains the canonical schedule. The controlled 50M WSD arm lost on
  both loss and post-SFT capability ([D013](docs/decisions.md#d013)).
- `microbatch=64, accumulation=1` is the trainer default: same 32,768 tokens per
  update and test-verified loss/gradient/step equivalence. After sync cleanup it is
  1.095x faster compiled at 50M and 1.098x at 300M
  ([D027](docs/decisions.md#d027)). Peak allocation rises from 5.0→16.0GB and
  14.0→40.2GB respectively. Above ~600M bodies use an explicit smaller shape.
- Sync-free metric collection did not materially change same-shape throughput
  (−0.38% to +0.89%); its value is honest, lower-overhead accounting rather than a
  measured speed claim. New runs quote `training_tokens_per_second`.
- The current SFT baseline is the concise arithmetic/number-word corpus, AdamW at
  `5e-5`, 100 warmup updates, and 1,000 planned updates. Future corpus comparisons
  must match supervised tokens and report wall time
  ([D015](docs/decisions.md#d015)).
- Short-SFT interpretation uses a declared checkpoint grid and targeted seed
  replication. Report the full grid and distribution; label the best checkpoint
  as selection-biased ([D025](docs/decisions.md#d025)).

## Live and recent work

- **Local Siamese/HybridNorm arm:** complete at 50M. Post-SFT score 506 versus 459
  for WSD and 474 for cosine, but training was roughly 9–12% slower and total time
  roughly 10–14% higher. Keep it experimental: it is not an efficiency win and is
  not a faithful implementation of the published algorithm
  ([D018](docs/decisions.md#d018)).
- **300M cosine run:** not currently running at 5,280,006,144 of its planned
  5,925,345,280 tokens (89.1%). The next resume uses 64x1 and
  `--profile-every 200`; the shape change will be recorded in `train.jsonl`
  ([D024](docs/decisions.md#d024), [D027](docs/decisions.md#d027)).
- **5.28B SFT diagnostic:** complete. Seeds 2029/2030/2031 score
  733/666/709 at update 600, 758/786/811 at 800, and 818/801/887 at 1,000.
  Together with seeds 2027/2028, the update-1,000 mean is 798.8 ± 68.1 sample SD.
- **600M/8B run:** stopped at roughly 1.10B tokens; it is not “training now.”
- **WSD at 50M:** rejected for the tested setting: loss 2.311 versus 2.297 for
  cosine, and post-SFT 459 versus 474.
- **Low precision:** the local FP8/NVFP4 paths were slower end to end and are
  deferred until a hardware-compatible fused/custom-kernel implementation exists.

Run-state bullets are snapshots. Check processes and the corresponding JSON/log
before acting.

## Findings worth preserving

- The dense modern stack beat the historical DeepSeek reference at matched stored
  capacity and tokens, though this was capacity-matched rather than compute-matched.
- More pretraining and concise SFT compound; SFT exposes capability that base-model
  completion behavior hides.
- Training loss is not a capability safety signal: continued pretraining recovered
  loss while arithmetic behavior regressed.
- The apparent 5.28B late-cosine capability regression did not replicate across
  SFT trajectories. SFT seed/checkpoint variation can reverse a single endpoint;
  it is not evidence that the base model lost learning capacity.
- GRPO's registered configuration produced zero gradient in 95.3% of rollout
  groups; the bottleneck was reward variation, not trainer mechanics.
- Muon improved the 250M loss trajectory without improving capability and did not
  retain the same loss advantage at 2B. Scope optimizer claims to the recipe.
- WSD did not transfer at 50M. Preserve the negative instead of promoting the paper
  result by default.
- Capability began moving clearly around the 100M-body rung, but the existing
  tokens/body-parameter axis understates the compute-bearing untied vocabulary
  head at small sizes.

Historical result documents preserve the numbers and interpretations available at
their date. The decision ledger is authoritative when policy has changed.

## Measurement debt that blocks strong claims

1. Split the repeatedly inspected benchmark into discovery/dev and sealed
   confirmation sets ([D004](docs/decisions.md#d004)).
2. Add immutable run/evaluation manifests: command, commit and dirty diff, data and
   tokenizer hashes, scorer version, environment, hardware, and intervention log.
3. ~~Remove GPU-to-Python scalar conversions from every microbatch; separate
   training-only, evaluation, checkpoint, and compile time.~~ Done for pretraining
   and validated on 2026-08-18 ([D023](docs/decisions.md#d023),
   [D027](docs/decisions.md#d027)). Terminal checkpoint timing now survives resume.
   Still open: `sft.py` syncs per example, and no run has an MFU number until a
   measured device peak is supplied.
4. Correct final partial-token updates by masking or slicing the unused targets.
5. Match SFT comparisons on supervised tokens and wall time, not examples alone.
6. Replace the body-only scale axis with total parameters, non-embedding
   compute-bearing parameters, and estimated FLOPs.
7. Benchmark the projections the model actually executes. Current FP8/NVFP4 shape
   probes assume fused QKV and gate/up while the model uses separate linears.

## Priority queue

1. **Measurement foundation:** synchronization-free metrics, segment timing, and
   resume interventions landed 2026-08-18 ([D023](docs/decisions.md#d023),
   [D024](docs/decisions.md#d024)) and were independently exercised in
   [D027](docs/decisions.md#d027). Still open: full run manifests (commit, dirty
   diff, data and tokenizer hashes, environment, hardware), sealed evaluation,
   exact partial-final-update token accounting, and the same sync cleanup in `sft.py`.
2. **Semantics-preserving efficiency:** fuse QKV and SwiGLU gate/up with checkpoint
   conversion and parity tests; then test fused linear cross-entropy on actual
   model shapes.
3. **Learning studies:** update-RMS-matched Muon, a faithful SiameseNorm path if its
   cost remains justified, and token-matched SFT composition tests.
4. **Scale transfer:** confirm selected changes at a second size or token budget;
   use an explicit retuning policy rather than assuming one recipe transfers.

## Updating this memory

When current truth changes ([D022](docs/decisions.md#d022)):

1. Append a stable-ID entry to [`docs/decisions.md`](docs/decisions.md).
2. Update this file's champion, live state, findings, or priorities.
3. If model or runtime structure changed, update both
   [`docs/architecture.md`](docs/architecture.md) and
   [`docs/architecture.json`](docs/architecture.json).
4. Update the `Last reconciled`/`as_of` dates and validate links.

Do not erase a failed path. Supersede its decision and retain the evidence.
