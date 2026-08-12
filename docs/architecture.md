# Architecture

A dense, pre-norm, decoder-only Transformer. Every component is the modern
default rather than the 2019 one, and the point of the repository is to measure
what that swap is worth against the DeepSeek-V4 reference at a matched parameter
count. This document covers what is implemented and why; for what it scored, see
[`results.md`](results.md) and the README table.

Source: `src/modern_lm/layers.py` (components), `src/modern_lm/model.py`
(assembly), `src/modern_lm/config.py` (shape and flags).

## The stack

| Component | Choice | Replaces |
|---|---|---|
| Normalization | RMSNorm, pre-norm | LayerNorm, post-norm |
| Position | RoPE, theta 10,000 | Learned absolute embeddings |
| Feed-forward | SwiGLU | 4x GELU MLP |
| Attention | Causal GQA through SDPA | MHA with an explicit mask |
| Attention stability | QK-norm | — |
| Biases | None anywhere | Linear biases |
| Output head | Untied from the embedding | Tied (optional here) |

### RMSNorm

No mean subtraction and no bias — only a learned per-channel scale:

```python
x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

The normalization runs in fp32 and the result is cast back to the input dtype.
That cast is load-bearing: under bf16 autocast, returning fp32 would silently
promote the rest of the block and cost throughput for no accuracy.

### RoPE

Position enters by rotating query/key pairs rather than by adding a position
vector, so attention sees relative offsets. `cos`/`sin` tables are built once at
construction for `max_seq_len` and registered as non-persistent buffers — they
are derived from `rope_theta` and `head_dim`, so writing them into checkpoints
would only bloat the file.

Rotation happens in fp32 before casting back, for the same reason as RMSNorm.

### Attention

`F.scaled_dot_product_attention` with `is_causal=True`. The `[T, T]` score matrix
is never materialized, which is where most of the memory in a naive
implementation goes.

Two ordering details that matter:

1. **QK-norm is applied before RoPE**, not after. Normalizing after rotation
   would partly undo the position encoding, since the rotation changes the
   vector's direction and a subsequent RMS rescale is not rotation-equivariant
   in the way you want.
2. **GQA repeats K/V after the rotation**, so each repeated head carries the
   same positional phase as the head it was expanded from.

`n_kv_heads` defaults to `n_heads`, i.e. full MHA. GQA's payoff is inference KV
memory, not training throughput; enabling it would reduce capacity without a
matching benefit to measure in this comparison.

### SwiGLU

```python
down(silu(gate(x)) * up(x))
```

Three matrices instead of two, so `ffn_dim` is chosen to keep the parameter count
matched rather than following the 4x convention: `2432` at `dim=768` is a 3.17x
ratio, where a 4x GELU MLP would use two matrices of `4*dim`.

### Residual and initialization

Standard pre-norm residual: `x = x + attn(norm(x))`, then
`x = x + ffn(norm(x))`. Normalizing the branch input rather than the sum keeps a
clean identity path from embedding to `final_norm`.

Weights initialize at `std=0.02`, except the two projections that write **into**
the residual stream — `o_proj` and `down_proj` — which are scaled by
`1/sqrt(2 * n_layers)`. Without this the residual variance grows with depth and
deep models need a longer warmup to become stable.

## Shape

`ModernConfig.dense_145m()`: `dim=768`, `n_layers=15`, `n_heads=12`,
`ffn_dim=2432`, `vocab_size=16384`, `max_seq_len=512` → **144,630,912
parameters**, within 0.027% of the reference's 144,669,412.

Where the parameters live:

| | 145M | 600M |
|---|---:|---:|
| embedding | 12,582,912 (8.7%) | 20,971,520 (3.5%) |
| lm_head | 12,582,912 (8.7%) | 20,971,520 (3.5%) |
| blocks | 119,465,088 (82.6%) | 558,432,512 (93.0%) |
| ffn_dim / dim | 3.17 | 3.40 |
| head_dim | 64 | 64 |

Model size is a CLI flag, not a source edit — `--dim`, `--n-layers`,
`--n-heads`, `--n-kv-heads`, `--ffn-dim`. The 600M run keeps `head_dim` at 64 and
roughly the same aspect ratios, so the only deliberate changes are width, depth,
and the token budget.

Config validation rejects the combinations that would fail confusingly later:
`dim` divisible by `n_heads`, `n_heads` divisible by `n_kv_heads`, and an even
`head_dim` (RoPE rotates pairs).

## Staged levers, off by default

Both are implemented and tested but disabled, so the headline comparison
isolates the dense modern stack. Enabling either changes two variables at once
against the reference.

**MTP** (`use_mtp`) predicts token `t+2` from hidden state `t` and the embedding
of `t+1`, through a projection, one extra block, and the shared `lm_head`. Costs
31,479 tok/s (0.85x) and is resumable from an existing checkpoint, since it only
adds parameters.

**MoE** (`use_moe`) replaces the feed-forward with top-k routed experts plus
always-on shared experts, carrying a Switch-Transformer load-balance auxiliary
loss. Costs 28,438 tok/s (0.77x) and is **not** resumable — it replaces every
feed-forward, so it needs a fresh run.

The MoE forward loops over experts and uses `index_add_`, which is correct but
not throughput-optimized; it exists so the staged arm needs no refactor, not to
be fast.

## Generation

`generate()` is greedy and **recomputes the full prefix every step — there is no
KV cache**. That is deliberate: the DeepSeek reference decodes the same way, and
adding a cache here would make evaluation wall-clock incomparable while
smuggling a decoding advantage into a benchmark comparison that is supposed to
isolate architecture.

The practical consequence is that evaluation is O(n²) in generated tokens, which
is why a 5024-question sweep at 96 new tokens takes ~80 minutes on the 600M model
and why `scripts/probe3.py` exists for quick checks.

Once a sequence emits EOS it is pinned to EOS, so a finished row cannot resume
generating and corrupt a batched decode.

## What the architecture did not fix

Worth recording alongside the design, because it bounds what the stack buys. The
modern components produced a real loss win at matched parameters — 2.5514 →
2.4049 at 250M tokens — and a real but small benchmark win. They did not produce
arithmetic competence: the best model still fails ~89% of the suite, and its
accuracy degrades with operand size (50% on two-digit addition, 26% on
three-digit) despite thousands of three-digit examples in the SFT corpus. That
is a capacity and objective limitation, not something RoPE or SwiGLU addresses.
