#!/usr/bin/env python3
"""Convert a checkpoint between separate and fused Q/K/V and gate/up projections.

`--fuse` rewrites q_proj/k_proj/v_proj as the row blocks of one qkv_proj and
gate_proj/up_proj as one gate_up_proj; `--unfuse` reverses it. Both directions
are lossless -- every number is preserved in order -- and both carry the
optimizer state across, so a run resumes with the momentum it had rather than
restarting from zero ([D028](../docs/decisions.md#d028)).

    python3 scripts/convert_projection_fusion.py --fuse \\
        runs/size300m-20x/latest.pt runs/size300m-20x/latest-fused.pt

The written checkpoint records `fuse_projections` in its config, so resuming
needs no extra flag -- but `train.py` builds its model from CLI arguments, so
pass --fuse-projections there to match, or the state dict will not load.

VERIFY, do not assume. --check reloads both checkpoints, runs a batch of random
tokens through each model, and reports the maximum logit difference. It should
be exactly zero on CPU: fusion changes how the matmul is tiled, and on CPU
float32 with these shapes that has produced bit-identical output. A nonzero
value on GPU is expected and small; a large one means the conversion is wrong.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from modern_lm.config import ModernConfig  # noqa: E402
from modern_lm.fusion import convert_checkpoint  # noqa: E402
from modern_lm.model import ModernLM  # noqa: E402


def verify(source: dict, converted: dict, batch: int = 2, length: int = 32) -> float:
    """Max absolute logit difference between the two checkpoints' models."""
    before_config = ModernConfig(**source["config"])
    after_config = ModernConfig(**converted["config"])
    length = min(length, before_config.max_seq_len)

    before = ModernLM(before_config)
    before.load_state_dict(source["model"])
    after = ModernLM(after_config)
    after.load_state_dict(converted["model"])
    before.eval()
    after.eval()

    tokens = torch.randint(0, before_config.vocab_size, (batch, length))
    with torch.no_grad():
        return (before(tokens).logits - after(tokens).logits).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--fuse", action="store_true")
    direction.add_argument("--unfuse", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="load both models and compare logits before writing")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing destination")
    args = parser.parse_args()

    if args.destination.exists() and not args.force:
        raise SystemExit(f"{args.destination} exists; pass --force to overwrite")

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    converted = convert_checkpoint(payload, to_fused=args.fuse)

    if args.check:
        difference = verify(payload, converted)
        print(f"max logit difference: {difference:.3e}")
        if difference > 1e-3:
            raise SystemExit("conversion changed the model's output; refusing to write")

    # Write to a temporary name and rename, so an interrupted write cannot leave
    # a half-written checkpoint where a valid one is expected.
    temporary = args.destination.with_suffix(".tmp")
    torch.save(converted, temporary)
    temporary.replace(args.destination)

    sidecar = args.destination.with_suffix(".json")
    source_sidecar = args.source.with_suffix(".json")
    if source_sidecar.exists():
        import json
        metadata = json.loads(source_sidecar.read_text())
        metadata["config"] = converted["config"]
        sidecar.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"wrote {args.destination} "
          f"(fuse_projections={converted['config']['fuse_projections']})")


if __name__ == "__main__":
    main()
