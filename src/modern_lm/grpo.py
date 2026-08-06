"""GRPO on a verifiable numeric reward, starting from the SFT checkpoint.

Protocol: `docs/grpo-protocol.md`, pre-registered before this file was written.

The reward is a rule, not a learned model: a completion scores 1.0 iff the
reference's own `numeric_equal(extract_number(completion), gold)` says the
answer is right. Both functions are imported from the DeepSeek experiment, the
same way `evaluate_benchmarks` imports them -- a second copy of the scorer would
silently diverge and break comparability with all six prior arms.

Two things here are deliberately *not* reused from elsewhere in this package:

1. `ModernLM.generate` is greedy (`argmax`) and wrapped in `@torch.no_grad`, so
   it can produce neither the sampled completions GRPO needs nor the per-token
   log-probabilities the policy gradient is computed from. `rollout` below is a
   separate sampling decoder. Evaluation still uses `generate` unchanged, so the
   benchmark numbers stay comparable.
2. The prompt string is rebuilt to byte-identical form with
   `deepseek_v4.sft.encode_sft_record` (`f"Question: {q}\\nAnswer:"`) and the
   response is sampled with a leading space, because that is what SFT taught and
   what the evaluation harness feeds. A different prompt here would train the
   policy on a distribution the benchmark never presents.

Why GRPO rather than PPO: the group-relative advantage removes the value head,
which is one fewer network that can diverge unnoticed. The sibling `llm-lessons`
repo shipped a PPO run whose reward *fell* (first-five 1.330 -> last-five 0.941)
while its prose claimed it rose. Gate 4 in the protocol exists to catch exactly
that, so `reward_mean` is logged every update against a fixed pre-run baseline.
"""
from __future__ import annotations

import argparse
import json
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
from .model import ModernLM
from .train import seed_everything

# Same import-with-pyarrow-stub dance as evaluate_benchmarks/sft: the reference
# modules pull pyarrow at import time for their *preparation* paths, which this
# module does not use and which is not installed outside the reference venv.
sys.path.insert(0, str(DEEPSEEK_REPO / "src"))


def _import_reference():
    try:
        from deepseek_v4.evaluation import extract_number, numeric_equal
        from deepseek_v4.sft import SFTRecord
        return extract_number, numeric_equal, SFTRecord
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on interpreter
        if exc.name != "pyarrow":
            raise
        import types

        stub = types.ModuleType("pyarrow.parquet")
        stub.read_table = None
        sys.modules.setdefault("pyarrow", types.ModuleType("pyarrow"))
        sys.modules["pyarrow.parquet"] = stub
        from deepseek_v4.evaluation import extract_number, numeric_equal
        from deepseek_v4.sft import SFTRecord
        return extract_number, numeric_equal, SFTRecord


extract_number, numeric_equal, SFTRecord = _import_reference()

SFT_DATA_DIR = DEEPSEEK_REPO / "data" / "sft-math"
CHECKPOINT_FORMAT = 1


@dataclass
class GRPOSettings:
    """Registered in docs/grpo-protocol.md before the first run."""
    group_size: int = 8
    prompts_per_update: int = 32
    rollout_microbatch: int = 64      # sequences per forward during sampling
    max_new_tokens: int = 64
    temperature: float = 1.0
    learning_rate: float = 1e-6       # ~50x below SFT; RL destabilizes easily here
    kl_coefficient: float = 0.04
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    planned_total_updates: int = 300
    checkpoint_updates: int = 50
    max_prompt_tokens: int = 320
    seed: int = 2028


def load_prompts(path: Path) -> list[SFTRecord]:
    """GRPO prompts come from the SFT *train* split only.

    The held-out 878 and all 5,024 benchmark questions stay untouched; the
    corpus metadata records `skipped_evaluation_overlap: 13`, i.e. it was
    already decontaminated against the evaluation set when it was built.
    """
    records = []
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            records.append(SFTRecord(source=item["source"], identifier=item["identifier"],
                                     question=item["question"], response=item["response"],
                                     answer=item["answer"]))
    return records


def score_completion(completion: str, gold: str) -> float:
    """Terminal reward: 1.0 for a correct number, else 0.0.

    No format bonus and no length shaping, per protocol -- SFT already reached
    99.6% numeric completion, so a format term would inflate reward without
    moving accuracy and would make gate 4 easy to pass for the wrong reason.
    """
    return 1.0 if numeric_equal(extract_number(completion), gold) else 0.0


@torch.no_grad()
def rollout(model: ModernLM, prompt_ids: list[list[int]], *, group_size: int,
            max_new_tokens: int, temperature: float, eos_id: int, pad_id: int,
            device: torch.device, amp: bool, microbatch: int,
            generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample `group_size` completions per prompt.

    Returns (sequences, completion_mask, prompt_lengths). Left-padding keeps every
    prompt in a batch ending at the same index, so the sampled continuation always
    starts at a single shared offset and no per-row bookkeeping is needed.

    Like `ModernLM.generate`, this recomputes the full prefix each step -- the
    model has no KV cache. That is the dominant cost of this trainer.
    """
    expanded: list[list[int]] = []
    for ids in prompt_ids:
        expanded.extend([list(ids)] * group_size)

    width = max(len(ids) for ids in expanded)
    sequences_out = []
    masks_out = []

    for start in range(0, len(expanded), microbatch):
        chunk = expanded[start:start + microbatch]
        padded = [[pad_id] * (width - len(ids)) + list(ids) for ids in chunk]
        tokens = torch.tensor(padded, dtype=torch.long, device=device)
        rows = tokens.shape[0]
        finished = torch.zeros(rows, dtype=torch.bool, device=device)
        completion_mask = torch.zeros((rows, max_new_tokens), dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            context = tokens[:, -model.config.max_seq_len:]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(context).logits[:, -1, :].float()
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
            # A finished row emits pad, and its mask stays False, so padding after
            # <eos> contributes no gradient and no KL.
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
            completion_mask[:, step] = ~finished
            finished = finished | (nxt == eos_id)
            tokens = torch.cat([tokens, nxt[:, None]], dim=1)
            if bool(finished.all()):
                completion_mask[:, step + 1:] = False
                break

        # Pad the sampled region back out to a uniform width across microbatches.
        produced = tokens.shape[1] - width
        if produced < max_new_tokens:
            filler = torch.full((rows, max_new_tokens - produced), pad_id,
                                dtype=torch.long, device=device)
            tokens = torch.cat([tokens, filler], dim=1)
        sequences_out.append(tokens)
        masks_out.append(completion_mask)

    sequences = torch.cat(sequences_out, dim=0)
    completion_mask = torch.cat(masks_out, dim=0)
    prompt_lengths = torch.full((sequences.shape[0],), width, dtype=torch.long, device=device)
    return sequences, completion_mask, prompt_lengths


def completion_logprobs(model: ModernLM, sequences: torch.Tensor, prompt_width: int,
                        *, amp: bool, device: torch.device) -> torch.Tensor:
    """Per-token log p(token | prefix) over the sampled region only.

    logits[:, i] predicts sequences[:, i+1], so the distribution for the first
    sampled token sits at index prompt_width - 1.
    """
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        logits = model(sequences).logits
    logits = logits[:, prompt_width - 1:-1, :].float()
    targets = sequences[:, prompt_width:]
    return torch.log_softmax(logits, dim=-1).gather(-1, targets[..., None]).squeeze(-1)


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Group-relative advantage: (r - mean) / (std + eps), per prompt.

    A group whose completions are all right or all wrong has zero variance and
    therefore zero advantage -- it contributes no gradient, which is correct:
    there is nothing to prefer within it.
    """
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + 1e-4)
    advantages = torch.where(std > 1e-6, advantages, torch.zeros_like(advantages))
    return advantages.reshape(-1)


def save_grpo_checkpoint(path: Path, model, optimizer, config: ModernConfig,
                         settings: GRPOSettings, state: dict, evaluation: dict) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "stage": "grpo",
        "config": config.to_dict(),
        "settings": asdict(settings),
        "state": state,
        "evaluation": evaluation,
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
    meta = {k: payload[k] for k in
            ("format_version", "stage", "config", "settings", "state", "evaluation")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def train_grpo(*, checkpoint: Path, run_dir: Path, prompts_path: Path,
               tokenizer_path: Path, settings: GRPOSettings, target_updates: int,
               log_every: int, device: torch.device) -> None:
    if not 0 < target_updates <= settings.planned_total_updates:
        raise ValueError("target_updates must be within the planned schedule")
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(settings.seed)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    if eos_id is None or pad_id is None:
        raise ValueError("tokenizer is missing <eos> or <pad>")

    records = load_prompts(prompts_path)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("stage") != "sft":
        raise ValueError("GRPO must start from an SFT checkpoint")
    config = ModernConfig(**payload["config"])
    model = ModernLM(config)
    model.load_state_dict(payload["model"])
    model.to(device)

    # Frozen reference for the KL anchor: the SFT policy this run starts from.
    reference = ModernLM(config)
    reference.load_state_dict(payload["model"])
    reference.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    del payload

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate,
                                  weight_decay=settings.weight_decay)
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Sampling draws from a dedicated generator so rollout randomness is
    # reproducible independently of parameter-init/dropout streams.
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed)
    order_rng = random.Random(settings.seed)

    state = {"optimizer_step": 0, "prompts_seen": 0, "completions_seen": 0,
             "elapsed_seconds": 0.0}
    log_path = run_dir / "train.jsonl"
    handle = log_path.open("a")

    def encode_prompt(record: SFTRecord) -> list[int]:
        text = f"Question: {record.question}\nAnswer:"
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        return ids[-settings.max_prompt_tokens:]

    def sample_and_score(batch: list[SFTRecord]) -> dict:
        prompt_ids = [encode_prompt(r) for r in batch]
        sequences, completion_mask, lengths = rollout(
            model, prompt_ids, group_size=settings.group_size,
            max_new_tokens=settings.max_new_tokens, temperature=settings.temperature,
            eos_id=eos_id, pad_id=pad_id, device=device, amp=amp,
            microbatch=settings.rollout_microbatch, generator=generator)
        width = int(lengths[0])
        rewards = []
        texts = []
        for row in range(sequences.shape[0]):
            span = sequences[row, width:][completion_mask[row]]
            text = tokenizer.decode(span.tolist())
            gold = batch[row // settings.group_size].answer
            texts.append(text)
            rewards.append(score_completion(text, gold))
        return {"sequences": sequences, "mask": completion_mask, "width": width,
                "rewards": torch.tensor(rewards, dtype=torch.float32, device=device),
                "texts": texts}

    # Gate 4 needs a fixed origin: the SFT policy's pass rate on this prompt
    # pool, measured before any update, so "reward rose" is not relative to a
    # drifting baseline.
    model.eval()
    baseline_batch = [records[i] for i in order_rng.sample(range(len(records)),
                                                           settings.prompts_per_update)]
    baseline = sample_and_score(baseline_batch)
    baseline_reward = float(baseline["rewards"].mean())
    start_record = {"event": "start", "settings": asdict(settings),
                    "baseline_reward": baseline_reward,
                    "target_updates": target_updates,
                    "prompt_pool": len(records),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "device": str(device), "precision": "bf16-autocast" if amp else "fp32"}
    handle.write(json.dumps(start_record) + "\n")
    handle.flush()
    print(json.dumps(start_record), flush=True)

    started = time.perf_counter()
    reward_history: list[float] = []
    model.train()

    while int(state["optimizer_step"]) < target_updates:
        batch = [records[i] for i in order_rng.sample(range(len(records)),
                                                      settings.prompts_per_update)]
        model.eval()
        sampled = sample_and_score(batch)
        model.train()

        sequences = sampled["sequences"]
        mask = sampled["mask"].float()
        width = sampled["width"]
        rewards = sampled["rewards"]
        advantages = group_advantages(rewards, settings.group_size)

        optimizer.zero_grad(set_to_none=True)
        total_tokens = mask.sum().clamp(min=1.0)
        policy_total = 0.0
        kl_total = 0.0

        # Chunked over rollout_microbatch so the backward pass fits the same
        # memory envelope the sampling pass already proved viable.
        for start in range(0, sequences.shape[0], settings.rollout_microbatch):
            stop = start + settings.rollout_microbatch
            chunk = sequences[start:stop]
            chunk_mask = mask[start:stop]
            chunk_adv = advantages[start:stop]

            logprobs = completion_logprobs(model, chunk, width, amp=amp, device=device)
            with torch.no_grad():
                ref_logprobs = completion_logprobs(reference, chunk, width,
                                                   amp=amp, device=device)

            # Single gradient step per rollout batch, so the importance ratio is
            # identically 1 and the surrogate reduces to -A * logprob. Written
            # explicitly rather than as a PPO ratio, because a ratio that is
            # always 1 is exactly the bug that made the sibling repo's PPO run
            # meaningless for several revisions.
            policy = -(chunk_adv[:, None] * logprobs) * chunk_mask

            # k3 estimator: unbiased, non-negative, lower variance than (ref-lp).
            diff = ref_logprobs - logprobs
            kl = (torch.exp(diff) - diff - 1.0) * chunk_mask

            loss = (policy + settings.kl_coefficient * kl).sum() / total_tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite GRPO loss")
            loss.backward()
            policy_total += float(policy.sum().detach())
            kl_total += float(kl.sum().detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip,
                                                   error_if_nonfinite=True)
        optimizer.step()

        state["optimizer_step"] = int(state["optimizer_step"]) + 1
        state["prompts_seen"] = int(state["prompts_seen"]) + len(batch)
        state["completions_seen"] = int(state["completions_seen"]) + sequences.shape[0]
        state["elapsed_seconds"] = time.perf_counter() - started

        reward_mean = float(rewards.mean())
        reward_history.append(reward_mean)
        step = int(state["optimizer_step"])

        if step == 1 or step % log_every == 0:
            denominator = float(total_tokens)
            spurious = sum(1 for t in sampled["texts"] if "Question:" in t)
            record = {"event": "update", "optimizer_step": step,
                      "reward_mean": reward_mean,
                      "reward_baseline": baseline_reward,
                      "policy_loss": policy_total / denominator,
                      "kl": kl_total / denominator,
                      "grad_norm": float(grad_norm),
                      "completion_tokens": denominator,
                      "zero_variance_groups": int(
                          (advantages.view(-1, settings.group_size).abs().sum(dim=1) == 0).sum()),
                      "spurious_question_rate": spurious / len(sampled["texts"]),
                      "prompts_seen": state["prompts_seen"],
                      "elapsed_seconds": state["elapsed_seconds"]}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)

        if step % settings.checkpoint_updates == 0 or step >= target_updates:
            evaluation = {"reward_mean": reward_mean,
                          "reward_baseline": baseline_reward,
                          "recent_reward_mean": float(np.mean(reward_history[-5:]))}
            save_grpo_checkpoint(run_dir / f"checkpoint-{step:06d}.pt", model, optimizer,
                                 config, settings, dict(state), evaluation)
            save_grpo_checkpoint(run_dir / "latest.pt", model, optimizer,
                                 config, settings, dict(state), evaluation)

    # Gate 4, evaluated in-run rather than left to the write-up.
    first5 = float(np.mean(reward_history[:5]))
    last5 = float(np.mean(reward_history[-5:]))
    complete = {"event": "complete", "optimizer_step": state["optimizer_step"],
                "reward_first5": first5, "reward_last5": last5,
                "reward_baseline": baseline_reward,
                "gate4_reward_rose": bool(last5 > first5),
                "elapsed_seconds": state["elapsed_seconds"]}
    handle.write(json.dumps(complete) + "\n")
    handle.close()
    print(json.dumps(complete), flush=True)


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/modern-145m-2b-grpo"))
    parser.add_argument("--prompts", type=Path, default=SFT_DATA_DIR / "train.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=paths["tokenizer"])
    parser.add_argument("--target-updates", type=int, default=300)
    parser.add_argument("--planned-total-updates", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-update", type=int, default=32)
    parser.add_argument("--rollout-microbatch", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coefficient", type=float, default=0.04)
    parser.add_argument("--checkpoint-updates", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    settings = GRPOSettings(
        group_size=args.group_size, prompts_per_update=args.prompts_per_update,
        rollout_microbatch=args.rollout_microbatch, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, learning_rate=args.learning_rate,
        kl_coefficient=args.kl_coefficient,
        planned_total_updates=args.planned_total_updates,
        checkpoint_updates=args.checkpoint_updates, seed=args.seed)

    train_grpo(checkpoint=args.checkpoint, run_dir=args.run_dir,
               prompts_path=args.prompts, tokenizer_path=args.tokenizer,
               settings=settings, target_updates=args.target_updates,
               log_every=args.log_every, device=torch.device(args.device))


if __name__ == "__main__":
    main()
