# ModernLM: a capacity-matched dense baseline for the DeepSeek-V4 reference

A modern dense decoder-only Transformer (RMSNorm, RoPE, SwiGLU, GQA, QK-norm)
sized to the **same stored parameter count** as the DeepSeek-V4 145M math model,
trained on the **same tokens in the same order** and scored by the **same
benchmark code**, to measure the quality/time trade-off between the two
architectures.

The companion repository is `../llm-deepseek-v4-experiment`. This one does not
duplicate its corpus, tokenizer, split, or scorer — it imports them, because a
re-derived tokenizer or a second copy of a scorer would make the comparison
illegitimate rather than merely inconvenient.

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

Default preset `ModernConfig.dense_145m()`: `dim=768`, `n_layers=15`,
`n_heads=12`, `ffn_dim=2432`, `vocab_size=16384`, `max_seq_len=512` →
**144,630,912 parameters**, within 0.027% of the reference's 144,669,412.

`n_kv_heads` defaults to `n_heads` (full MHA). GQA's payoff is inference KV
memory, not training throughput, so enabling it would change capacity without a
matching benefit to measure here.

### Staged levers, off by default

MTP (`use_mtp`) and MoE (`use_moe`) are implemented and tested but disabled, so
the first comparison isolates the dense modern stack. Enabling either changes
two variables at once against the reference. See `docs/comparison-protocol.md`.

## Layout

```
src/modern_lm/
  config.py               ModernConfig, dense_145m() and tiny() presets
  layers.py               RMSNorm, RoPE, SwiGLU, GQA attention, MoE
  model.py                ModernLM, MTPHead, generate()
  data.py                 PackedTokenStream — port of the reference sampler
  train.py                Resumable pretraining loop
  evaluate_benchmarks.py  Scores checkpoints with the reference's own scorer
tests/                    18 tests, CPU-only
docs/comparison-protocol.md   Pre-registered gates and controlled variables
```

## Usage

```bash
python -m pytest tests/ -q

python -m src.modern_lm.train \
  --target-tokens 250000000 \
  --run-dir runs/modern-145m \
  --microbatch-size 16 --gradient-accumulation 4 \
  --device cuda

python -m src.modern_lm.evaluate_benchmarks \
  --checkpoint runs/modern-145m/latest.pt \
  --output runs/modern-145m/evaluation.jsonl \
  --max-new-tokens 32 --device cuda
```

Training is resumable (`--resume runs/modern-145m/latest.pt`); checkpoints carry
model, optimizer, and Python/NumPy/CPU/CUDA RNG state, loaded with
`map_location="cpu"` so RNG ByteTensors stay where `set_rng_state` needs them.

## Results

See `docs/results.md` (written when the run completes).

## Data and licensing

No corpus is redistributed here. Training data is FineMath-4+ (ODC-By 1.0) as
packed by the companion repository; benchmarks are GSM8K (MIT), SVAMP (MIT), and
ASDiv (CC BY-NC 4.0). DeepSeek names and configurations belong to their owners.
