"""Continuous pretraining that converts an AR checkpoint into a dLM.

Implements the Efficient-DLM recipe (arXiv:2512.14067) on this project's stack:
block-wise attention with clean context, position-dependent masking, and the
MDM objective. See dlm.py for the three components and their equations.

Run:
    python -m modern_lm.convert_dlm \
        --checkpoint runs/sft-50m-cosine/latest.pt \
        --run-dir runs/dlm-50m --target-tokens 200000000

SCALE. The paper converts a 1.5B model with 50B tokens and reports that ~10B is
the point where accuracy recovers. Anything far below that is a mechanism test
-- does the machinery run and does the objective descend -- not a reproduction
of their accuracy or throughput claims. The run's own summary records the
budget so a later reader does not mistake one for the other.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import ModernConfig
from .data import PackedTokenStream
from .model import ModernLM
from . import dlm

DEFAULT_DATA = Path("/home/tyrel/projects/llm-deepseek-v4-experiment/data/finemath-6b")


def evaluate(model, stream, mask, block_size, beta, batch_size, batches, device):
    """Held-out MDM loss at a fixed seed, so the number is comparable run to run."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(batches):
            rows = stream.batch(i * batch_size, batch_size, device)
            ids = rows[:, :-1]
            noisy, m, t = dlm.corrupt(ids, block_size, beta, generator=gen)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(noisy, attn_mask=mask).logits
            total += float(dlm.mdm_loss(logits, ids, m, t))
            n += 1
    model.train()
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--train", type=Path, default=DEFAULT_DATA / "train.bin")
    p.add_argument("--heldout", type=Path, default=DEFAULT_DATA / "heldout.bin")
    p.add_argument("--target-tokens", type=int, default=200_000_000)
    p.add_argument("--block-size", type=int, default=16,
                   help="paper's choice; too small starves denoising context, "
                        "too large over-corrupts and drifts the weights")
    p.add_argument("--mask-lambda", type=float, default=0.1,
                   help="half-life ratio; 0.1 was the paper's best, <=0 = uniform")
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="paper's initial LR; decayed to --min-lr by cosine")
    p.add_argument("--min-lr", type=float, default=3e-6)
    p.add_argument("--microbatch-size", type=int, default=32)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every-tokens", type=int, default=20_000_000)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log = (args.run_dir / "train.jsonl").open("a")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ModernConfig(**ckpt["config"])
    model = ModernLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    seq_len = config.max_seq_len
    if seq_len % args.block_size:
        raise ValueError(
            f"block_size {args.block_size} must divide max_seq_len {seq_len}")

    beta = dlm.beta_from_lambda(args.mask_lambda, args.block_size)
    mask = dlm.block_causal_mask(seq_len, args.block_size, device)

    train = PackedTokenStream(args.train, seq_len, args.seed)
    held = PackedTokenStream(args.heldout, seq_len, args.seed + 1)

    tokens_per_step = args.microbatch_size * seq_len
    total_steps = args.target_tokens // tokens_per_step
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps, eta_min=args.min_lr)

    emit({"event": "start", "checkpoint": str(args.checkpoint),
          "target_tokens": args.target_tokens, "total_steps": total_steps,
          "block_size": args.block_size, "mask_lambda": args.mask_lambda,
          "beta": beta, "learning_rate": args.learning_rate,
          "mask_token_id": dlm.MASK_TOKEN_ID, "seed": args.seed,
          "scale_caveat": "paper reports ~10B tokens for accuracy recovery; "
                          f"this run is {args.target_tokens/1e9:.2f}B",
          "baseline_heldout_mdm": evaluate(
              model, held, mask, args.block_size, beta,
              args.microbatch_size, args.eval_batches, device)})

    started = time.time()
    seen = 0
    next_eval = args.eval_every_tokens
    for step in range(1, total_steps + 1):
        rows = train.batch((step - 1) * args.microbatch_size,
                           args.microbatch_size, device)
        ids = rows[:, :-1]
        noisy, m, t = dlm.corrupt(ids, args.block_size, beta)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(noisy, attn_mask=mask).logits
        loss = dlm.mdm_loss(logits, ids, m, t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += tokens_per_step

        if step % args.log_every == 0:
            emit({"event": "update", "step": step, "tokens_seen": seen,
                  "loss": float(loss), "grad_norm": float(gn),
                  "lr": sched.get_last_lr()[0],
                  "elapsed_seconds": time.time() - started,
                  "tokens_per_second": seen / (time.time() - started)})

        if seen >= next_eval or step == total_steps:
            hl = evaluate(model, held, mask, args.block_size, beta,
                          args.microbatch_size, args.eval_batches, device)
            emit({"event": "evaluation", "step": step, "tokens_seen": seen,
                  "heldout_mdm_loss": hl})
            torch.save({"config": ckpt["config"], "model": model.state_dict(),
                        "dlm": {"block_size": args.block_size,
                                "mask_lambda": args.mask_lambda, "beta": beta,
                                "mask_token_id": dlm.MASK_TOKEN_ID},
                        "tokens_seen": seen},
                       args.run_dir / "latest.pt")
            next_eval += args.eval_every_tokens

    emit({"event": "complete", "tokens_seen": seen,
          "elapsed_seconds": time.time() - started})
    log.close()


if __name__ == "__main__":
    main()
