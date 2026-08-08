"""Resumable pretraining for ModernLM on the shared FineMath corpus.

Hyperparameters mirror the DeepSeek-V4 reference run exactly (sequence length
512, 32,768 target tokens per optimizer update, AdamW wd 0.1, lr 3e-4, 2,000
warmup updates, cosine decay to 0.1x, grad clip 1.0, seed 2026) so the only
intended difference between the two runs is the architecture.

Token accounting matches the reference: the trainer counts *supervised
next-token labels*, not input-array lengths.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ModernConfig
from .data import PackedTokenStream, default_paths
from .model import ModernLM
from .muon import build_optimizer

CHECKPOINT_FORMAT = 1


@dataclass
class TrainSettings:
    sequence_length: int = 512
    microbatch_size: int = 16
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    warmup_updates: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.1
    planned_total_tokens: int = 250_000_000
    checkpoint_tokens: int = 10_000_000
    mtp_weight: float = 0.1
    aux_weight: float = 0.01
    seed: int = 2026
    optimizer: str = "adamw"
    muon_learning_rate: float = 0.02


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_at(update: int, settings: TrainSettings, total_updates: int) -> float:
    """Linear warmup then cosine decay, planned against the full token budget."""
    if update < settings.warmup_updates:
        return settings.learning_rate * (update + 1) / settings.warmup_updates
    progress = (update - settings.warmup_updates) / max(
        1, total_updates - settings.warmup_updates)
    progress = min(1.0, max(0.0, progress))
    floor = settings.learning_rate * settings.min_lr_ratio
    return floor + 0.5 * (settings.learning_rate - floor) * (1 + math.cos(math.pi * progress))


def compute_loss(model: ModernLM, tokens: torch.Tensor, settings: TrainSettings
                 ) -> tuple[torch.Tensor, dict[str, float], int]:
    """Next-token loss over `tokens` [B, S+1]; returns (loss, components, n_targets)."""
    inputs, labels = tokens[:, :-1], tokens[:, 1:]
    # torch.compile's OptimizedModule proxies attribute access to the wrapped
    # module, so `.config` resolves for both compiled and eager models.
    config = model.config
    output = model(inputs,
                   return_aux_loss=config.use_moe,
                   return_mtp_logits=config.use_mtp)
    # bf16 logits straight into cross_entropy: it upcasts internally for the
    # softmax, avoiding a full-precision [B, S, V] materialization.
    main = F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]), labels.reshape(-1))
    total = main
    components = {"main": float(main.detach())}

    if output.aux_loss is not None and settings.aux_weight:
        total = total + settings.aux_weight * output.aux_loss
        components["aux"] = float(output.aux_loss.detach())
    if output.mtp_logits is not None and output.mtp_logits.shape[1] > 0 and settings.mtp_weight:
        mtp_labels = labels[:, 1:]
        mtp = F.cross_entropy(
            output.mtp_logits.reshape(-1, output.mtp_logits.shape[-1]),
            mtp_labels.reshape(-1))
        total = total + settings.mtp_weight * mtp
        components["mtp"] = float(mtp.detach())

    components["total"] = float(total.detach())
    return total, components, labels.numel()


@torch.no_grad()
def evaluate(model: ModernLM, stream: PackedTokenStream, settings: TrainSettings,
             batches: int, device: torch.device, amp: bool) -> dict[str, float]:
    model.eval()
    losses, weights = [], []
    position = 0
    for _ in range(batches):
        tokens = stream.batch(position, settings.microbatch_size, device)
        position += settings.microbatch_size
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            _, components, n_targets = compute_loss(model, tokens, settings)
        losses.append(components["main"] * n_targets)
        weights.append(n_targets)
    model.train()
    main_loss = sum(losses) / sum(weights)
    return {"main_loss": main_loss, "perplexity": math.exp(min(20.0, main_loss))}


def save_checkpoint(path: Path, model, optimizer, config: ModernConfig,
                    settings: TrainSettings, state: dict) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "config": config.to_dict(),
        "settings": asdict(settings),
        "state": state,
        "model": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    meta = {k: payload[k] for k in ("format_version", "config", "settings", "state")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def prune_checkpoints(run_dir: Path, keep_last: int) -> None:
    """Retain only the `keep_last` most recent milestone checkpoints.

    A 2B-token run at a 50M checkpoint interval writes 40 milestones; at ~1.7GiB
    each that is ~70GiB, which does not fit comfortably alongside the corpus.
    The per-milestone JSON metadata is never pruned, so the loss trajectory
    stays fully recoverable after the weights are gone. `latest.pt` is written
    separately and is never a pruning candidate.
    """
    if keep_last <= 0:
        return
    checkpoints = sorted(run_dir.glob("checkpoint-*.pt"))
    for stale in checkpoints[:-keep_last]:
        stale.unlink(missing_ok=True)


def load_checkpoint(path: Path, model, optimizer) -> dict:
    # map_location="cpu" is required: RNG state ByteTensors must stay on CPU for
    # torch.set_rng_state / set_rng_state_all to accept them. load_state_dict
    # moves the model/optimizer tensors onto the device afterwards.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    target.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    rng = payload.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return payload["state"]


def train(config: ModernConfig, settings: TrainSettings, *, target_tokens: int,
          run_dir: Path, device: torch.device, resume: Path | None,
          eval_batches: int, log_every: int, compile_model: bool,
          train_path: Path, heldout_path: Path,
          keep_last_checkpoints: int = 0) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(settings.seed)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model = ModernLM(config).to(device)
    n_params = model.num_params()
    if compile_model:
        model = torch.compile(model)

    if settings.optimizer == "muon":
        optimizer = build_optimizer(
            model, learning_rate=settings.learning_rate,
            muon_learning_rate=settings.muon_learning_rate,
            weight_decay=settings.weight_decay)
    else:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim < 2 else decay).append(param)
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": settings.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=settings.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    train_stream = PackedTokenStream(train_path, settings.sequence_length, settings.seed)
    heldout_stream = PackedTokenStream(heldout_path, settings.sequence_length, settings.seed + 1)

    tokens_per_update = (settings.microbatch_size * settings.gradient_accumulation
                         * settings.sequence_length)
    total_updates = max(1, settings.planned_total_tokens // tokens_per_update)

    state = {"micro_step": 0, "optimizer_step": 0, "tokens_seen": 0,
             "elapsed_seconds": 0.0, "next_checkpoint_tokens": settings.checkpoint_tokens}
    if resume is not None and resume.exists():
        state = load_checkpoint(resume, model, optimizer)
        print(json.dumps({"event": "resumed", "state": state}), flush=True)
    else:
        initial = evaluate(model, heldout_stream, settings, eval_batches, device, amp)
        state["initial_evaluation"] = initial
        print(json.dumps({"event": "initial_evaluation", **initial}), flush=True)

    log_path = run_dir / "train.jsonl"
    log_handle = log_path.open("a")
    started = time.perf_counter()
    base_elapsed = state.get("elapsed_seconds", 0.0)

    model.train()
    while state["tokens_seen"] < target_tokens:
        lr = learning_rate_at(state["optimizer_step"], settings, total_updates)
        # `lr` follows the AdamW base; a group carrying `lr_scale` (Muon) keeps
        # the same warmup/cosine shape rescaled to its own base magnitude.
        for group in optimizer.param_groups:
            scale = group.get("lr_scale")
            group["lr"] = lr if scale is None else lr * (scale / settings.learning_rate)

        optimizer.zero_grad(set_to_none=True)
        update_tokens = 0
        update_components: dict[str, float] = {}
        remaining = target_tokens - state["tokens_seen"]

        # Token-weighted accumulation so a partial final update stays exact.
        planned = []
        for _ in range(settings.gradient_accumulation):
            if remaining <= 0:
                break
            batch_targets = settings.microbatch_size * settings.sequence_length
            planned.append(min(batch_targets, remaining))
            remaining -= planned[-1]
        if not planned:
            break
        planned_total = sum(planned)

        for micro_targets in planned:
            tokens = train_stream.batch(state["micro_step"], settings.microbatch_size, device)
            state["micro_step"] += settings.microbatch_size
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, components, n_targets = compute_loss(model, tokens, settings)
            scale = micro_targets / planned_total
            (loss * scale).backward()
            update_tokens += micro_targets
            for key, value in components.items():
                update_components[key] = update_components.get(key, 0.0) + value * scale

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
        optimizer.step()
        state["optimizer_step"] += 1
        state["tokens_seen"] += update_tokens
        state["elapsed_seconds"] = base_elapsed + (time.perf_counter() - started)

        if state["optimizer_step"] % log_every == 0:
            record = {"event": "update", "optimizer_step": state["optimizer_step"],
                      "tokens_seen": state["tokens_seen"], "lr": lr,
                      "grad_norm": float(grad_norm),
                      "elapsed_seconds": state["elapsed_seconds"],
                      "tokens_per_second": state["tokens_seen"] / max(1e-9, state["elapsed_seconds"]),
                      **{f"loss_{k}": v for k, v in update_components.items()}}
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record), flush=True)

        crossed = state["tokens_seen"] >= state["next_checkpoint_tokens"]
        if crossed or state["tokens_seen"] >= target_tokens:
            metrics = evaluate(model, heldout_stream, settings, eval_batches, device, amp)
            state["heldout"] = metrics
            record = {"event": "evaluation", "tokens_seen": state["tokens_seen"],
                      "elapsed_seconds": state["elapsed_seconds"],
                      "parameters": n_params, **metrics}
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            print(json.dumps(record), flush=True)
            while state["next_checkpoint_tokens"] <= state["tokens_seen"]:
                state["next_checkpoint_tokens"] += settings.checkpoint_tokens
            save_checkpoint(run_dir / f"checkpoint-{state['tokens_seen']:012d}.pt",
                            model, optimizer, config, settings, state)
            save_checkpoint(run_dir / "latest.pt", model, optimizer, config, settings, state)
            prune_checkpoints(run_dir, keep_last_checkpoints)

    log_handle.close()
    print(json.dumps({"event": "complete", "tokens_seen": state["tokens_seen"],
                      "elapsed_seconds": state["elapsed_seconds"],
                      "heldout": state.get("heldout")}), flush=True)


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/modern-145m"))
    parser.add_argument("--train", type=Path, default=paths["train"])
    parser.add_argument("--heldout", type=Path, default=paths["heldout"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-updates", type=int, default=2000)
    parser.add_argument("--planned-total-tokens", type=int, default=250_000_000)
    parser.add_argument("--checkpoint-tokens", type=int, default=10_000_000,
                        help="evaluate and checkpoint on each crossed multiple")
    parser.add_argument("--keep-last-checkpoints", type=int, default=0,
                        help="if >0, retain only the N most recent milestone "
                             "checkpoints (latest.pt is always kept)")
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--muon-learning-rate", type=float, default=0.02,
                        help="peak LR for the Muon group; --learning-rate still "
                             "drives the AdamW group and the schedule shape")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    settings = TrainSettings(
        microbatch_size=args.microbatch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_updates=args.warmup_updates,
        planned_total_tokens=args.planned_total_tokens,
        checkpoint_tokens=args.checkpoint_tokens,
        optimizer=args.optimizer,
        muon_learning_rate=args.muon_learning_rate,
        seed=args.seed)
    train(ModernConfig.dense_145m(), settings,
          target_tokens=args.target_tokens, run_dir=args.run_dir,
          device=torch.device(args.device), resume=args.resume,
          eval_batches=args.eval_batches, log_every=args.log_every,
          compile_model=not args.no_compile,
          train_path=args.train, heldout_path=args.heldout,
          keep_last_checkpoints=args.keep_last_checkpoints)


if __name__ == "__main__":
    main()
