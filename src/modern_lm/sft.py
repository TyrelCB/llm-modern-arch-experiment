"""Supervised fine-tuning for ModernLM on the shared math/instruction corpus.

The SFT corpus, its deterministic holdout split, and its decontamination against
all 5,023 evaluation questions are reused from the DeepSeek-V4 experiment rather
than regenerated -- the same reasoning that governs the pretraining corpus and
the benchmark scorer. Regenerating would risk a different holdout split and
break comparability with the reference's own recorded SFT baseline.

The record encoding, collation, and length-bucketed batch ordering are imported
from `deepseek_v4.sft` for the same reason: a second copy of the prompt format
or the label-masking rule would silently diverge, and the prompt string here
must stay identical to the one the evaluation harness uses.

Settings default to the reference's *recorded* run (microbatch 16 x accum 2),
not its CLI defaults (4 x 8), which differ and would produce a different batch
composition at the same examples-per-update.

Why this stage matters for this project: the 2B pretrained model answers
correctly and then runs on, because packed pretraining text almost never
teaches "stop after answering" (only 0.1% of its completions terminate early).
SFT supervises `<eos>` on every response, so a large part of the expected gain
is the model learning to stop -- a genuine improvement, not a scorer change.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .config import ModernConfig
from .data import DEEPSEEK_REPO, default_paths
from .low_precision import (
    SUPPORTED_PRECISIONS,
    canonical_model_state_dict,
    configure_low_precision,
    load_canonical_model_state_dict,
    low_precision_autocast,
)
from .model import ModernLM
from .train import seed_everything

# Reuse the reference's data-side SFT code. It has no model dependency; see the
# module docstring for why it is imported rather than copied. pyarrow is stubbed
# for the same reason as in evaluate_benchmarks: it is only needed by the
# `prepare` path, which this module does not use.
sys.path.insert(0, str(DEEPSEEK_REPO / "src"))
try:
    from deepseek_v4.sft import (  # noqa: E402
        EncodedSFTDataset, SFTRecord, _batch_order, collate_sft, encode_sft_record,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on interpreter
    if exc.name != "pyarrow":
        raise
    import types

    stub = types.ModuleType("pyarrow.parquet")
    stub.read_table = None
    sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
    sys.modules["pyarrow.parquet"] = stub
    from deepseek_v4.sft import (  # noqa: E402
        EncodedSFTDataset, SFTRecord, _batch_order, collate_sft, encode_sft_record,
    )

SFT_DATA_DIR = DEEPSEEK_REPO / "data" / "sft-math"
CHECKPOINT_FORMAT = 1


@dataclass
class SFTSettings:
    max_sequence_length: int = 512
    microbatch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 5e-5
    warmup_updates: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    planned_total_updates: int = 1000
    checkpoint_updates: int = 100
    min_lr_ratio: float = 0.1
    precision: str = "bf16"
    seed: int = 2027


def learning_rate_for_update(update: int, settings: SFTSettings) -> float:
    """Linear warmup then cosine decay, planned against the full update budget.

    Matches `deepseek_v4.pretrain.learning_rate_for_update` so a 100-update gate
    followed by a resume to 1000 traces the identical schedule.
    """
    base = settings.learning_rate
    if update <= settings.warmup_updates:
        return base * update / max(settings.warmup_updates, 1)
    progress = ((update - settings.warmup_updates)
                / max(settings.planned_total_updates - settings.warmup_updates, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return base * (settings.min_lr_ratio + (1.0 - settings.min_lr_ratio) * cosine)


@torch.no_grad()
def evaluate_sft(model: ModernLM, dataset, *, batch_size: int, batches: int,
                 pad_id: int, device: torch.device, amp: bool) -> dict[str, float]:
    """Supervised-token-weighted held-out loss over a deterministic prefix.

    Sorting by length and taking a prefix (rather than sampling) keeps the
    measurement identical across checkpoints, which is what makes the trajectory
    comparable update to update.
    """
    was_training = model.training
    model.eval()
    numerator = 0.0
    supervised_tokens = 0
    order = sorted(range(len(dataset)), key=lambda index: dataset[index][0].numel())
    for start in range(0, min(len(order), batches * batch_size), batch_size):
        examples = [dataset[index] for index in order[start:start + batch_size]]
        inputs, labels = collate_sft(examples, pad_id)
        inputs, labels = inputs.to(device), labels.to(device)
        count = int((labels != -100).sum())
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            with low_precision_autocast(model, is_first_microbatch=(start == 0)):
                logits = model(inputs).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels.reshape(-1), ignore_index=-100)
        numerator += float(loss) * count
        supervised_tokens += count
    model.train(was_training)
    mean = numerator / max(supervised_tokens, 1)
    return {"loss": mean, "perplexity": math.exp(min(mean, 20.0)),
            "supervised_tokens": float(supervised_tokens)}


def save_sft_checkpoint(path: Path, model, optimizer, config: ModernConfig,
                        settings: SFTSettings, state: dict, evaluation: dict) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "stage": "sft",
        "config": config.to_dict(),
        "settings": asdict(settings),
        "state": state,
        "evaluation": evaluation,
        "model": canonical_model_state_dict(model),
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
    meta = {k: payload[k] for k in
            ("format_version", "stage", "config", "settings", "state", "evaluation")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def train_sft(*, checkpoint: Path, resume: Path | None, run_dir: Path,
              train_path: Path, heldout_path: Path, tokenizer_path: Path,
              settings: SFTSettings, target_updates: int, eval_batches: int,
              log_every: int, device: torch.device) -> None:
    if not 0 < target_updates <= settings.planned_total_updates:
        raise ValueError("target_updates must be within the planned schedule")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(settings.seed)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("tokenizer is missing <pad>")

    train_data = EncodedSFTDataset(train_path, tokenizer, settings.max_sequence_length)
    heldout_data = EncodedSFTDataset(heldout_path, tokenizer, settings.max_sequence_length)

    # map_location="cpu": the payload carries RNG-state ByteTensors, which
    # torch.set_rng_state / set_rng_state_all require to stay on CPU.
    source = resume if resume is not None else checkpoint
    payload = torch.load(source, map_location="cpu", weights_only=False)
    config = ModernConfig(**payload["config"])
    if settings.max_sequence_length > config.max_seq_len:
        raise ValueError("SFT max sequence length exceeds model max_seq_len")
    model = ModernLM(config)
    load_canonical_model_state_dict(model, payload["model"])
    model.to(device)
    precision = configure_low_precision(model, settings.precision, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    state = {"optimizer_step": 0, "examples_seen": 0, "supervised_tokens_seen": 0,
             "epoch": 0, "batch_in_epoch": 0, "elapsed_seconds": 0.0}

    if resume is not None:
        if payload.get("stage") != "sft":
            raise ValueError("resume checkpoint is not from an SFT run")
        previous_settings = dict(payload.get("settings") or {})
        previous_settings.setdefault("precision", "bf16")
        if previous_settings != asdict(settings):
            raise ValueError("resume checkpoint SFT settings differ")
        optimizer.load_state_dict(payload["optimizer"])
        state.update(payload["state"])
        rng = payload.get("rng") or {}
        if rng.get("python"):
            random.setstate(rng["python"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    del payload

    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    evaluation = evaluate_sft(model, heldout_data, batch_size=settings.microbatch_size,
                              batches=eval_batches, pad_id=pad_id, device=device, amp=amp)
    log_path = run_dir / "train.jsonl"
    handle = log_path.open("a")
    start_record = {"event": "start", "state": dict(state), "evaluation": evaluation,
                    "target_updates": target_updates, "settings": asdict(settings),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "train_examples": len(train_data), "heldout_examples": len(heldout_data),
                    "device": str(device), "precision": precision.to_dict()}
    handle.write(json.dumps(start_record) + "\n")
    handle.flush()
    print(json.dumps(start_record), flush=True)

    started = time.perf_counter()
    base_elapsed = state.get("elapsed_seconds", 0.0)
    model.train()
    batches_for_epoch = _batch_order(train_data, settings.microbatch_size,
                                     settings.seed, int(state["epoch"]))

    while int(state["optimizer_step"]) < target_updates:
        optimizer.zero_grad(set_to_none=True)
        selected: list[list[int]] = []
        for _ in range(settings.gradient_accumulation):
            if int(state["batch_in_epoch"]) >= len(batches_for_epoch):
                state["epoch"] = int(state["epoch"]) + 1
                state["batch_in_epoch"] = 0
                batches_for_epoch = _batch_order(train_data, settings.microbatch_size,
                                                 settings.seed, int(state["epoch"]))
            selected.append(batches_for_epoch[int(state["batch_in_epoch"])])
            state["batch_in_epoch"] = int(state["batch_in_epoch"]) + 1

        selected_examples = [[train_data[i] for i in indices] for indices in selected]
        # Token-weighted accumulation: cross_entropy returns a per-token mean over
        # non-ignored positions, so scaling each microbatch by its share of the
        # update's supervised tokens makes the update exact for variable-length
        # responses (a plain 1/accum average would over-weight short responses).
        update_tokens = sum(int((labels != -100).sum())
                            for examples in selected_examples for _, labels in examples)
        main_numerator = 0.0
        examples_in_update = 0

        for microbatch_index, examples in enumerate(selected_examples):
            inputs, labels = collate_sft(examples, pad_id)
            inputs, labels = inputs.to(device), labels.to(device)
            count = int((labels != -100).sum())
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                with low_precision_autocast(
                        model, is_first_microbatch=(microbatch_index == 0)):
                    logits = model(inputs).logits
                main = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       labels.reshape(-1), ignore_index=-100)
                scaled = main * count / max(update_tokens, 1)
            if not torch.isfinite(scaled):
                raise FloatingPointError("non-finite SFT loss")
            scaled.backward()
            main_numerator += float(main.detach()) * count
            examples_in_update += inputs.shape[0]

        lr = learning_rate_for_update(int(state["optimizer_step"]) + 1, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip,
                                                   error_if_nonfinite=True)
        optimizer.step()

        state["optimizer_step"] = int(state["optimizer_step"]) + 1
        state["examples_seen"] = int(state["examples_seen"]) + examples_in_update
        state["supervised_tokens_seen"] = int(state["supervised_tokens_seen"]) + update_tokens
        state["elapsed_seconds"] = base_elapsed + (time.perf_counter() - started)

        step = int(state["optimizer_step"])
        if step == 1 or step % log_every == 0:
            record = {"event": "update", "optimizer_step": step,
                      "loss_main": main_numerator / max(update_tokens, 1),
                      "lr": lr, "grad_norm": float(grad_norm),
                      "examples_seen": state["examples_seen"],
                      "supervised_tokens_seen": state["supervised_tokens_seen"],
                      "epoch": state["epoch"],
                      "elapsed_seconds": state["elapsed_seconds"]}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)

        if step % settings.checkpoint_updates == 0 or step >= target_updates:
            evaluation = evaluate_sft(model, heldout_data,
                                      batch_size=settings.microbatch_size,
                                      batches=eval_batches, pad_id=pad_id,
                                      device=device, amp=amp)
            record = {"event": "evaluation", "optimizer_step": step, **evaluation,
                      "elapsed_seconds": state["elapsed_seconds"]}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)
            save_sft_checkpoint(run_dir / f"checkpoint-{step:06d}.pt", model, optimizer,
                                config, settings, dict(state), evaluation)
            save_sft_checkpoint(run_dir / "latest.pt", model, optimizer,
                                config, settings, dict(state), evaluation)

    handle.close()
    print(json.dumps({"event": "complete", "optimizer_step": state["optimizer_step"],
                      "evaluation": evaluation,
                      "elapsed_seconds": state["elapsed_seconds"]}), flush=True)


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/modern-145m-2b-sft"))
    parser.add_argument("--train", type=Path, default=SFT_DATA_DIR / "train.jsonl")
    parser.add_argument("--heldout", type=Path, default=SFT_DATA_DIR / "heldout.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=paths["tokenizer"])
    parser.add_argument("--target-updates", type=int, default=100)
    parser.add_argument("--planned-total-updates", type=int, default=1000)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--checkpoint-updates", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--precision", choices=SUPPORTED_PRECISIONS, default="bf16",
                        help="projection GEMM precision; fp8/nvfp4 require the "
                             "Transformer Engine optional dependency")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    settings = SFTSettings(
        microbatch_size=args.microbatch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_updates=args.warmup_updates,
        planned_total_updates=args.planned_total_updates,
        checkpoint_updates=args.checkpoint_updates,
        precision=args.precision,
        seed=args.seed)
    train_sft(checkpoint=args.checkpoint, resume=args.resume, run_dir=args.run_dir,
              train_path=args.train, heldout_path=args.heldout,
              tokenizer_path=args.tokenizer, settings=settings,
              target_updates=args.target_updates, eval_batches=args.eval_batches,
              log_every=args.log_every, device=torch.device(args.device))


if __name__ == "__main__":
    main()
