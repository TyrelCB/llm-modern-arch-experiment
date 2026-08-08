"""Muon optimizer (orthogonalized momentum) and the AdamW pairing it needs.

Muon applies only to 2D hidden-layer weight matrices. Embeddings, the LM head,
1D norm gains, and MoE routers stay on AdamW: Muon's update assumes a dense
linear map between hidden spaces, which vocab-indexed and gating matrices are
not. `split_muon_params` encodes that boundary.

The two optimizers are driven through `CombinedOptimizer` so the existing
trainer -- which assumes a single object with .param_groups/.step/.state_dict --
needs no restructuring.
"""
from __future__ import annotations

import torch


@torch.compile(fullgraph=True, dynamic=False)
def zeropower_via_newtonschulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of G via a quintic Newton-Schulz iteration.

    The coefficients drive the singular values into roughly [0.7, 1.3] rather
    than exactly 1; only the direction matters for the update, so exact
    convergence would be wasted work. Runs in bf16 -- the iteration is
    self-correcting, so the reduced precision does not accumulate.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    # Work on the short side: the Gram matrix X @ X.mT is then min(rows, cols)
    # square, which for vocab- or FFN-shaped matrices is a large saving.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.mT if transposed else X


class Muon(torch.optim.Optimizer):
    """Momentum SGD whose update is orthogonalized before being applied."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            momentum, lr = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1.0 - momentum)
                update = p.grad.lerp(buf, momentum) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz(update, group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                # Shape-aware scale so a single LR transfers across matrices of
                # differing aspect ratio (the orthogonal factor has unit-scale
                # singular values regardless of size).
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(update.to(p.dtype), alpha=-lr * scale)
        return loss


def split_muon_params(model) -> tuple[list, list, list]:
    """Partition parameters into (muon_2d, adamw_decay, adamw_no_decay).

    Deduplicates by tensor identity so tied embedding/LM-head weights are not
    handed to two optimizers.
    """
    # torch.compile wraps the model, prefixing every parameter name with
    # "_orig_mod."; match against the underlying module so the name-based rules
    # below behave identically compiled and uncompiled.
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    muon, decay, no_decay = [], [], []
    seen: set[int] = set()
    for name, param in target.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        hidden_matrix = (
            param.ndim == 2
            and name.startswith("blocks.")
            and "router" not in name
        )
        if hidden_matrix:
            muon.append(param)
        elif param.ndim < 2:
            no_decay.append(param)
        else:
            decay.append(param)
    return muon, decay, no_decay


class CombinedOptimizer:
    """Presents two optimizers as one, preserving each group's own LR scale.

    `param_groups` is the concatenation of the children's groups, each tagged
    with `lr_scale` -- the trainer multiplies its scheduled LR by that factor,
    so Muon (lr ~2e-2) and AdamW (lr ~3e-4) follow the same warmup/cosine shape
    at their own magnitudes.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def state_dict(self) -> dict:
        return {"combined": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, payload: dict) -> None:
        if "combined" not in payload:
            raise ValueError(
                "checkpoint optimizer state is not from a combined (Muon) run")
        states = payload["combined"]
        if len(states) != len(self.optimizers):
            raise ValueError("combined optimizer arity differs from checkpoint")
        for opt, state in zip(self.optimizers, states):
            opt.load_state_dict(state)


def build_optimizer(model, *, learning_rate: float, muon_learning_rate: float,
                    weight_decay: float, momentum: float = 0.95,
                    ns_steps: int = 5) -> CombinedOptimizer:
    """Muon on hidden 2D matrices, AdamW on everything else."""
    muon_params, decay, no_decay = split_muon_params(model)
    if not muon_params:
        raise ValueError(
            "Muon parameter group is empty -- the hidden-matrix name rule matched "
            "nothing. Check that split_muon_params sees unwrapped parameter names.")
    muon = Muon(muon_params, lr=muon_learning_rate, momentum=momentum,
                ns_steps=ns_steps, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    # The trainer scales every group by its own base LR ratio.
    for group in muon.param_groups:
        group["lr_scale"] = muon_learning_rate
    for group in adamw.param_groups:
        group["lr_scale"] = learning_rate
    return CombinedOptimizer([muon, adamw])
