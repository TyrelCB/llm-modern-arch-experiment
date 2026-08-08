"""Does the model give more predicted probability mass to rare tokens?

Frequency-bucketed loss says how well each bucket is predicted when it is the
target. It does not say how much probability mass the model *spends* on rare
tokens. A model can improve tail loss while still being globally over-confident
about frequent tokens.

This measures the output distribution itself, averaged over held-out positions:

- mass_by_bucket: mean predicted probability assigned to each frequency bucket.
  Compared against the empirical next-token distribution, this shows whether a
  model under- or over-allocates to the tail.
- entropy: mean predictive entropy. Higher means less peaked.
- top1_prob: mean probability of the argmax token. Lower means less confident.

Usage:
  PYTHONPATH=src python scripts/rare_token_mass.py \
      --checkpoint runs/muon-probe-lr0.005/checkpoint-000120000000.pt \
      --label muon005
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from modern_lm.config import ModernConfig
from modern_lm.data import default_paths
from modern_lm.model import ModernLM
from token_frequency_loss import BUCKETS, token_frequencies


@torch.no_grad()
def measure(checkpoint: Path, heldout_path: Path, counts: np.ndarray,
            device: torch.device, seq_len: int, batches: int,
            batch_size: int) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ModernLM(ModernConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    del payload
    model.to(device).eval()

    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    rank_t = torch.from_numpy(rank.astype(np.int64)).to(device)

    # Column mask per bucket over the vocabulary.
    masks = {name: ((rank_t >= lo) & (rank_t < hi)).float()
             for name, lo, hi in BUCKETS}

    data = np.memmap(heldout_path, dtype=np.uint16, mode="r")
    usable = (len(data) - 1) // seq_len
    n = min(batches * batch_size, usable)

    mass = {name: 0.0 for name, _, _ in BUCKETS}
    empirical = {name: 0 for name, _, _ in BUCKETS}
    entropy_sum, top1_sum, positions = 0.0, 0.0, 0

    for start in range(0, n, batch_size):
        idx = range(start, min(start + batch_size, n))
        chunks = [np.asarray(data[i * seq_len:(i + 1) * seq_len + 1],
                             dtype=np.int64) for i in idx]
        batch = torch.from_numpy(np.stack(chunks)).to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        output = model(inputs)
        logits = getattr(output, "logits", output)
        probs = F.softmax(logits.reshape(-1, logits.size(-1)).float(), dim=-1)

        logp = torch.log(probs.clamp_min(1e-12))
        entropy_sum += (-(probs * logp).sum(-1)).sum().item()
        top1_sum += probs.max(-1).values.sum().item()
        positions += probs.size(0)

        for name, _, _ in BUCKETS:
            mass[name] += (probs @ masks[name]).sum().item()

        target_rank = rank_t[targets.reshape(-1)]
        for name, lo, hi in BUCKETS:
            empirical[name] += int(((target_rank >= lo)
                                    & (target_rank < hi)).sum().item())

    return {
        "predicted_mass": {k: v / positions for k, v in mass.items()},
        "empirical_share": {k: v / positions for k, v in empirical.items()},
        "entropy": entropy_sum / positions,
        "top1_prob": top1_sum / positions,
        "positions": positions,
    }


def main() -> None:
    paths = default_paths()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--heldout", type=Path, default=paths["heldout"])
    ap.add_argument("--train", type=Path, default=paths["train"])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--freq-sample", type=int, default=100_000_000)
    ap.add_argument("--out", type=Path, default=Path("runs/rare-token-mass.json"))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    counts = token_frequencies(args.train, 16384, args.freq_sample)
    result = measure(args.checkpoint, args.heldout, counts,
                     torch.device(args.device), args.seq_len,
                     args.batches, args.batch_size)
    result["checkpoint"] = str(args.checkpoint)

    existing = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text())
    existing[args.label] = result
    args.out.write_text(json.dumps(existing, indent=2))

    print(f"{args.label}: entropy {result['entropy']:.4f} "
          f"top1 {result['top1_prob']:.4f}")
    for name, _, _ in BUCKETS:
        p = result["predicted_mass"][name]
        e = result["empirical_share"][name]
        print(f"  {name:>10}: predicted {p:.4f} empirical {e:.4f} "
              f"ratio {p/e if e else float('nan'):.3f}")


if __name__ == "__main__":
    main()
