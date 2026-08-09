"""How far have the weights moved from the 2B checkpoint, relative to their own norm?

Loss recovers after a reheat even when the run has overwritten the model, so loss
cannot tell a warm continuation from a restart-in-disguise. Relative weight drift
can: the aborted lr=0.005 run moved 114.9% in 50M tokens, i.e. it replaced the
model, which is why algebra and arithmetic went to zero while loss looked fine.
"""
import sys
import torch

BASE = "/home/tyrel/projects/llm-modern-arch-experiment/runs/muon-2b-lr0.005/checkpoint-002000000000.pt"


def drift(base_sd, other_sd):
    total, count = 0.0, 0
    for key, value in base_sd.items():
        if key not in other_sd or not value.dtype.is_floating_point:
            continue
        base_norm = value.float().norm().item()
        if base_norm > 0:
            total += (other_sd[key].float() - value.float()).norm().item() / base_norm
            count += 1
    return total / count if count else float("nan")


base = torch.load(BASE, map_location="cpu", weights_only=False)["model"]
for path in sys.argv[1:]:
    other = torch.load(path, map_location="cpu", weights_only=False)["model"]
    print(f"{path.split('/')[-2]:38s} {drift(base, other) * 100:8.2f}%")
