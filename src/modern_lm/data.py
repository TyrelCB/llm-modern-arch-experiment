"""Data access over the DeepSeek-V4 experiment's already-packed FineMath corpus.

Nothing here re-tokenizes or re-packs. The tokenizer (16,384-token byte-level
BPE), the decontaminated document split, and the packed uint16 binaries are
reused byte-for-byte from `llm-deepseek-v4-experiment`, because a different
tokenizer or corpus would make the held-out loss comparison meaningless.

`PackedTokenStream` is a deliberate line-for-line port of the reference's
sampler (same seed, same coprime-multiplier LCG walk), so at a given position
both trainers see the identical sequence of training blocks.

Source corpus: FineMath-4+ (shards 0/16/32/48), documented in the DeepSeek
experiment's `docs/finemath-145m-run.md`. Benchmark data: GSM8K, SVAMP, ASDiv.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch

# The companion repository holding the packed corpus and benchmark artifacts.
DEEPSEEK_REPO = Path("/home/tyrel/projects/llm-deepseek-v4-experiment")
DEFAULT_DATA_DIR = DEEPSEEK_REPO / "data" / "finemath-4plus"
DEFAULT_BENCHMARK_DIR = DEEPSEEK_REPO / "data" / "benchmarks"


def default_paths() -> dict[str, Path]:
    return {
        "train": DEFAULT_DATA_DIR / "train.bin",
        "heldout": DEFAULT_DATA_DIR / "heldout.bin",
        "tokenizer": DEFAULT_DATA_DIR / "tokenizer.json",
        "metadata": DEFAULT_DATA_DIR / "metadata.json",
        "evaluation": DEFAULT_BENCHMARK_DIR / "evaluation.jsonl",
    }


class PackedTokenStream:
    """Port of the reference sampler; identical block order for a given seed."""

    def __init__(self, path: Path, sequence_length: int, seed: int):
        self.tokens = np.memmap(path, mode="r", dtype="<u2")
        self.sequence_length = sequence_length
        self.blocks = (len(self.tokens) - 1) // sequence_length
        self.seed = seed
        self._parameters: dict[int, tuple[int, int]] = {}
        if self.blocks < 1:
            raise ValueError(f"{path} contains no complete sequence")

    def _permutation(self, epoch: int) -> tuple[int, int]:
        if epoch not in self._parameters:
            rng = random.Random(self.seed + epoch)
            multiplier = rng.randrange(1, self.blocks + 1)
            while math.gcd(multiplier, self.blocks) != 1:
                multiplier = rng.randrange(1, self.blocks + 1)
            self._parameters[epoch] = multiplier, rng.randrange(self.blocks)
        return self._parameters[epoch]

    def block_index(self, position: int) -> int:
        epoch, within_epoch = divmod(position, self.blocks)
        multiplier, offset = self._permutation(epoch)
        return (multiplier * within_epoch + offset) % self.blocks

    def batch(self, position: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return [batch_size, sequence_length + 1] int64 tokens.

        Gathers with a single numpy pass per row rather than per-row tensor
        transfers; the reference's per-sample `.to(device)` was measurable
        overhead at microbatch 64.
        """
        rows = np.empty((batch_size, self.sequence_length + 1), dtype=np.int64)
        for offset in range(batch_size):
            start = self.block_index(position + offset) * self.sequence_length
            rows[offset] = self.tokens[start:start + self.sequence_length + 1]
        return torch.from_numpy(rows).to(device, non_blocking=True)
