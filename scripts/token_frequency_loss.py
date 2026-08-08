"""Per-token-frequency-bucket held-out loss for matched checkpoints.

Tests whether an optimizer gives more representation to rare tokens, rather than
buying its aggregate loss purely on frequent ones. Token frequencies come from
the training corpus; held-out positions are bucketed by the frequency rank of
the *target* token, and mean NLL is reported per bucket.

Aggregate held-out loss is a frequency-weighted average dominated by common
tokens, so two models with the same aggregate loss can differ substantially on
the tail. That is the quantity of interest here.

Usage:
  PYTHONPATH=src python scripts/token_frequency_loss.py \
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

# Frequency-rank buckets. Bounds are cumulative-probability based rather than
# rank based, so each bucket holds a comparable share of corpus mass.
BUCKETS = [
    ("top-10", 0, 10),
    ("10-100", 10, 100),
    ("100-1k", 100, 1_000),
    ("1k-4k", 1_000, 4_000),
    ("4k-8k", 4_000, 8_000),
    ("8k-16k", 8_000, 16_384),
]


def token_frequencies(train_path: Path, vocab_size: int,
                      sample_tokens: int) -> np.ndarray:
    """Count token occurrences over a prefix of the training stream."""
    data = np.memmap(train_path, dtype=np.uint16, mode="r")
    sample = np.asarray(data[:min(sample_tokens, len(data))])
    counts = np.bincount(sample, minlength=vocab_size).astype(np.int64)
    return counts


@torch.no_grad()
def bucket_losses(checkpoint: Path, heldout_path: Path, counts: np.ndarray,
                  device: torch.device, seq_len: int, batches: int,
                  batch_size: int) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ModernLM(ModernConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    del payload
    model.to(device).eval()

    # rank 0 = most frequent. Unseen tokens sort to the end.
    order = np.argsort(-counts, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    rank_t = torch.from_numpy(rank.astype(np.int64)).to(device)

    data = np.memmap(heldout_path, dtype=np.uint16, mode="r")
    usable = (len(data) - 1) // seq_len
    n = min(batches * batch_size, usable)

    totals = {name: 0.0 for name, _, _ in BUCKETS}
    hits = {name: 0 for name, _, _ in BUCKETS}
    overall_sum, overall_n = 0.0, 0

    for start in range(0, n, batch_size):
        idx = range(start, min(start + batch_size, n))
        chunks = [np.asarray(data[i * seq_len:(i + 1) * seq_len + 1],
                             dtype=np.int64) for i in idx]
        batch = torch.from_numpy(np.stack(chunks)).to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]

        output = model(inputs)
        logits = getattr(output, "logits", output)
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1), reduction="none")

        flat_targets = targets.reshape(-1)
        target_rank = rank_t[flat_targets]
        overall_sum += nll.sum().item()
        overall_n += nll.numel()

        for name, lo, hi in BUCKETS:
            mask = (target_rank >= lo) & (target_rank < hi)
            k = int(mask.sum().item())
            if k:
                totals[name] += nll[mask].sum().item()
                hits[name] += k

    out = {name: (totals[name] / hits[name] if hits[name] else float("nan"))
           for name, _, _ in BUCKETS}
    return {
        "per_bucket_loss": out,
        "per_bucket_positions": hits,
        "overall_loss": overall_sum / overall_n,
        "positions": overall_n,
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
    ap.add_argument("--out", type=Path, default=Path("runs/token-frequency.json"))
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    counts = token_frequencies(args.train, 16384, args.freq_sample)
    result = bucket_losses(args.checkpoint, args.heldout, counts,
                           torch.device(args.device), args.seq_len,
                           args.batches, args.batch_size)
    result["label"] = args.label
    result["checkpoint"] = str(args.checkpoint)

    existing = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text())
    existing[args.label] = result
    args.out.write_text(json.dumps(existing, indent=2))

    print(f"{args.label}: overall {result['overall_loss']:.4f} "
          f"over {result['positions']} positions")
    for name, _, _ in BUCKETS:
        print(f"  {name:>10}: {result['per_bucket_loss'][name]:.4f} "
              f"({result['per_bucket_positions'][name]} positions)")


if __name__ == "__main__":
    main()
