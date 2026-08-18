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


def _row_spans(p: torch.Tensor, row_blocks: tuple[int, ...] | None):
    """(start, stop) row ranges to treat as independent matrices.

    Without `row_blocks` this yields the whole tensor, so an unfused parameter
    takes exactly the path it always did -- same shape into Newton-Schulz, so
    the same `dynamic=False` specialization, no extra Dynamo recompiles. A fused
    parameter yields the shapes of the matrices it replaced, which are the same
    shapes the unfused model compiled for.
    """
    if not row_blocks:
        yield 0, p.size(-2)
        return
    if sum(row_blocks) != p.size(-2):
        raise ValueError(
            f"row_blocks {row_blocks} do not sum to {p.size(-2)} rows")
    start = 0
    for rows in row_blocks:
        yield start, start + rows
        start += rows


def fused_row_blocks(model) -> dict[int, tuple[int, ...]]:
    """Map id(weight) -> row blocks, for every module that declares them.

    Read from the MODULE rather than the parameter: `Module._apply` (what `.to`,
    `.cuda` and `.half` go through) can replace the Parameter object, which would
    take a custom attribute set on the parameter with it, silently and without
    error -- the fused model would then train with a different optimizer than the
    unfused one and nothing would say so.
    """
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    blocks = {}
    for module in target.modules():
        rows = getattr(module, "muon_row_blocks", None)
        weight = getattr(module, "weight", None)
        if rows is not None and weight is not None:
            blocks[id(weight)] = tuple(rows)
    return blocks


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
            # Row spans to orthogonalize independently. A fused projection is
            # several matrices stored in one tensor, and orthogonalization is
            # not separable: Newton-Schulz on a stacked [3*dim, dim] matrix does
            # not produce the three factors it would produce on the parts, and
            # the aspect-ratio scale would jump from 1 to sqrt(3) as well. Fusing
            # without this would silently change the optimizer, which is the one
            # thing a semantics-preserving systems change may not do (D025).
            row_blocks = group.get("row_blocks")
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                # Momentum and decay are elementwise, so they need no knowledge
                # of the block structure.
                buf.lerp_(p.grad, 1.0 - momentum)
                update = p.grad.lerp(buf, momentum) if group["nesterov"] else buf
                if group["weight_decay"]:
                    p.mul_(1.0 - lr * group["weight_decay"])
                for start, stop in _row_spans(p, row_blocks):
                    block = update[start:stop]
                    orthogonal = zeropower_via_newtonschulz(block, group["ns_steps"])
                    # Shape-aware scale so a single LR transfers across matrices
                    # of differing aspect ratio (the orthogonal factor has
                    # unit-scale singular values regardless of size).
                    scale = max(1.0, block.size(-2) / block.size(-1)) ** 0.5
                    p[start:stop].add_(orthogonal.to(p.dtype), alpha=-lr * scale)
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


def split_adamw_params(model) -> tuple[list, list]:
    """Partition parameters into (decay, no_decay) for an AdamW-only run.

    A separate rule from `split_muon_params`: with no Muon group, every 2-D
    matrix decays and every 1-D one does not, so the ordering differs from the
    hybrid run's. Both orderings matter beyond convenience -- an optimizer state
    dict indexes its entries by position, so anything that reconstructs one
    (`modern_lm.fusion`) has to walk the parameters in exactly this order.
    """
    target = model._orig_mod if hasattr(model, "_orig_mod") else model
    decay, no_decay = [], []
    seen: set[int] = set()
    for _, param in target.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (no_decay if param.ndim < 2 else decay).append(param)
    return decay, no_decay


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
                    weight_decay: float, muon_weight_decay: float | None = None,
                    momentum: float = 0.95,
                    ns_steps: int = 5) -> CombinedOptimizer:
    """Muon on hidden 2D matrices, AdamW on everything else.

    `muon_weight_decay` defaults to `weight_decay`, which is what every run
    before it existed did. Because Muon shrinks by `lr * weight_decay`, sharing
    the value across a ~17x learning-rate gap couples the two knobs: changing
    Muon's LR silently changes its regularization by the same factor. Pass it
    explicitly to vary one at a time.
    """
    muon_params, decay, no_decay = split_muon_params(model)
    if not muon_params:
        raise ValueError(
            "Muon parameter group is empty -- the hidden-matrix name rule matched "
            "nothing. Check that split_muon_params sees unwrapped parameter names.")
    if muon_weight_decay is None:
        muon_weight_decay = weight_decay
    # One Muon group per block signature. Params keep their original relative
    # order inside the groups, and the plain group comes first, so an unfused
    # model produces exactly the single group it always did -- including the
    # parameter ordering that a checkpoint's optimizer state is indexed by.
    signatures: dict[tuple[int, ...], list] = {}
    blocks = fused_row_blocks(model)
    for param in muon_params:
        signatures.setdefault(blocks.get(id(param), ()), []).append(param)
    muon = Muon([{"params": params, "row_blocks": signature}
                 for signature, params in sorted(signatures.items(), key=lambda i: len(i[0]))],
                lr=muon_learning_rate, momentum=momentum,
                ns_steps=ns_steps, weight_decay=muon_weight_decay)
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
