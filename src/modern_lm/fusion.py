"""Convert checkpoints between separate and fused projections.

`fuse_projections` changes which tensors exist, not what they mean:
`q_proj`, `k_proj` and `v_proj` become the row blocks of one `qkv_proj`, and
`gate_proj`/`up_proj` become one `gate_up_proj`. Every number is preserved, in
order, so conversion is lossless in both directions and a converted checkpoint
resumes into an identical model ([D028](../../docs/decisions.md#d028)).

Optimizer state converts with it. Momentum buffers and Adam moments are
elementwise, so they concatenate along the same rows as the weights, and a
resumed run keeps the momentum it had rather than restarting from zero — which
is what "existing checkpoints remain usable" has to mean for a run that is
already 3.4B tokens deep.

The one thing not preserved is the ordering of parameters inside the optimizer's
state dict, because fusion changes how many parameters there are. Rather than
re-deriving that ordering here and hoping it keeps matching `build_optimizer`,
this module builds both optimizers on meta tensors and reads the ordering off
them. If the grouping rule changes, this follows it automatically.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from .config import ModernConfig
from .model import ModernLM
from .muon import build_optimizer, split_adamw_params

# fused name -> (source names, attribute holding each block's row count)
ATTENTION_FUSION = ("qkv_proj", ("q_proj", "k_proj", "v_proj"))
FFN_FUSION = ("gate_up_proj", ("gate_proj", "up_proj"))
FUSIONS = (ATTENTION_FUSION, FFN_FUSION)


def _row_blocks(config: ModernConfig) -> dict[str, tuple[int, ...]]:
    """Rows each source matrix contributes to its fused matrix."""
    head_dim = config.head_dim
    return {
        "qkv_proj": (config.n_heads * head_dim,
                     config.n_kv_heads * head_dim,
                     config.n_kv_heads * head_dim),
        "gate_up_proj": (config.ffn_dim, config.ffn_dim),
    }


def _fused_groups(keys) -> dict[str, tuple[str, list[str]]]:
    """Map every fusable key to (fused key, the ordered source keys beside it).

    Keyed by prefix so this works for any module path -- blocks, MoE experts,
    shared experts -- without knowing the model's layout.
    """
    present = set(keys)
    groups: dict[str, tuple[str, list[str]]] = {}
    for fused_name, source_names in FUSIONS:
        for key in present:
            suffix = f".{source_names[0]}.weight"
            if not key.endswith(suffix):
                continue
            prefix = key[: -len(suffix)]
            sources = [f"{prefix}.{name}.weight" for name in source_names]
            if all(source in present for source in sources):
                fused = f"{prefix}.{fused_name}.weight"
                for source in sources:
                    groups[source] = (fused, sources)
    return groups


def fuse_state_dict(state: dict, config: ModernConfig) -> dict:
    """Separate projections -> fused. Row order is q, k, v and gate, up."""
    groups = _fused_groups(state)
    converted: dict = {}
    for key, value in state.items():
        if key not in groups:
            converted[key] = value
            continue
        fused, sources = groups[key]
        if fused not in converted:
            converted[fused] = torch.cat([state[source] for source in sources], dim=0)
    return converted


def unfuse_state_dict(state: dict, config: ModernConfig) -> dict:
    """Fused projections -> separate, splitting on the configured row counts."""
    blocks = _row_blocks(config)
    converted: dict = {}
    for key, value in state.items():
        matched = None
        for fused_name, source_names in FUSIONS:
            if key.endswith(f".{fused_name}.weight"):
                matched = (fused_name, source_names)
                break
        if matched is None:
            converted[key] = value
            continue
        fused_name, source_names = matched
        prefix = key[: -len(f".{fused_name}.weight")]
        rows = blocks[fused_name]
        if value.shape[0] != sum(rows):
            raise ValueError(
                f"{key} has {value.shape[0]} rows; the config implies {sum(rows)}. "
                "The checkpoint and the config describe different models.")
        for name, part in zip(source_names, value.split(rows, dim=0)):
            converted[f"{prefix}.{name}.weight"] = part.clone()
    return converted


def _ordered_names(config: ModernConfig) -> list[list[str]]:
    """Parameter names per child optimizer, in the order state indices use.

    Built by constructing the real optimizer over a meta-device model: no memory
    is allocated for a 300M-parameter conversion, and the ordering cannot drift
    away from `build_optimizer` because it IS `build_optimizer`.
    """
    with torch.device("meta"):
        model = ModernLM(config)
    optimizer = build_optimizer(model, learning_rate=1.0, muon_learning_rate=1.0,
                                weight_decay=0.0)
    by_id = {id(param): name for name, param in model.named_parameters()}
    return [[by_id[id(param)] for group in child.param_groups for param in group["params"]]
            for child in optimizer.optimizers]


def _adamw_names(config: ModernConfig) -> list[str]:
    """Parameter names in the order an AdamW-only run's state dict indexes them."""
    with torch.device("meta"):
        model = ModernLM(config)
    by_id = {id(param): name for name, param in model.named_parameters()}
    decay, no_decay = split_adamw_params(model)
    return [by_id[id(param)] for param in decay + no_decay]


def _convert_state(child_state: dict, old_names: list[str], new_names: list[str],
                   state_dict_skeleton: dict, config: ModernConfig, *,
                   to_fused: bool) -> dict:
    """Re-key one optimizer's `state` by name, converting the fused entries."""
    by_name = {old_names[index]: entry
               for index, entry in child_state.get("state", {}).items()}
    if not by_name:
        return state_dict_skeleton                      # nothing stepped yet

    converted: dict = {}
    for name in new_names:
        if name in by_name:
            converted[name] = by_name[name]
            continue
        # A name only the target has: build it from its counterparts.
        converted[name] = _convert_entry(name, by_name, config, to_fused=to_fused)

    skeleton = dict(state_dict_skeleton)
    skeleton["state"] = {index: converted[name] for index, name in enumerate(new_names)
                         if converted[name] is not None}
    return skeleton


def _convert_entry(name: str, by_name: dict, config: ModernConfig, *, to_fused: bool):
    """Concatenate or split one parameter's optimizer state."""
    blocks = _row_blocks(config)
    if to_fused:
        for fused_name, source_names in FUSIONS:
            if not name.endswith(f".{fused_name}.weight"):
                continue
            prefix = name[: -len(f".{fused_name}.weight")]
            parts = [by_name.get(f"{prefix}.{source}.weight") for source in source_names]
            if any(part is None for part in parts):
                return None
            return _merge(parts)
        return None

    for fused_name, source_names in FUSIONS:
        for position, source in enumerate(source_names):
            if not name.endswith(f".{source}.weight"):
                continue
            prefix = name[: -len(f".{source}.weight")]
            entry = by_name.get(f"{prefix}.{fused_name}.weight")
            if entry is None:
                continue
            return _slice(entry, blocks[fused_name], position)
    return None


def _merge(parts: list[dict]) -> dict:
    """Concatenate matching tensor entries; require scalars to already agree.

    Shape decides: a per-element buffer (momentum, exp_avg, exp_avg_sq) has the
    parameter's rows and concatenates. A 0-dim entry (Adam's step counter) is a
    property of the update history, identical across the matrices that were
    stepped together — if it is not, the states came from different runs and
    merging them would fabricate a checkpoint that never existed.
    """
    merged = {}
    for key in parts[0]:
        values = [part[key] for part in parts]
        first = values[0]
        if torch.is_tensor(first) and first.ndim >= 1:
            merged[key] = torch.cat(values, dim=0)
        else:
            for other in values[1:]:
                if not _same_scalar(first, other):
                    raise ValueError(
                        f"optimizer state '{key}' differs across the matrices being "
                        f"fused ({first} vs {other}); they are not from one run")
            merged[key] = first
    return merged


def _slice(entry: dict, rows: tuple[int, ...], position: int) -> dict:
    sliced = {}
    start = sum(rows[:position])
    for key, value in entry.items():
        if torch.is_tensor(value) and value.ndim >= 1:
            sliced[key] = value[start:start + rows[position]].clone()
        else:
            sliced[key] = value
    return sliced


def _same_scalar(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return bool(torch.equal(a, b))
    return a == b


def convert_checkpoint(payload: dict, *, to_fused: bool) -> dict:
    """Convert a whole checkpoint payload, model and optimizer state together.

    Returns a new payload; the input is not modified. `config.fuse_projections`
    in the result reflects the direction converted, so checkpoint consumers can
    reconstruct the layout. The current training CLI still builds its model from
    arguments, so a fused resume must also pass `--fuse-projections`.
    """
    source_config = ModernConfig(**payload["config"])
    if source_config.fuse_projections == to_fused:
        raise ValueError(
            f"checkpoint already has fuse_projections={to_fused}; nothing to convert")
    target_config = replace(source_config, fuse_projections=to_fused)

    converted = dict(payload)
    converted["config"] = target_config.to_dict()
    converted["model"] = (fuse_state_dict(payload["model"], source_config) if to_fused
                          else unfuse_state_dict(payload["model"], source_config))

    optimizer_state = payload.get("optimizer")
    if optimizer_state is not None:
        converted["optimizer"] = _convert_optimizer(
            optimizer_state, source_config, target_config, to_fused=to_fused)
    return converted


def _convert_optimizer(state: dict, source_config: ModernConfig,
                       target_config: ModernConfig, *, to_fused: bool) -> dict:
    old_names = _ordered_names(source_config)
    new_names = _ordered_names(target_config)
    skeletons = _fresh_state_dicts(target_config)

    if "combined" in state:
        if len(state["combined"]) != len(old_names):
            raise ValueError("combined optimizer arity differs from the model's")
        return {"combined": [
            _convert_state(child, old, new, skeleton, source_config, to_fused=to_fused)
            for child, old, new, skeleton
            in zip(state["combined"], old_names, new_names, skeletons)]}

    # A plain AdamW run indexes ALL parameters in one space, decay group first --
    # a different order from the hybrid run's, which is why it gets its own
    # lookup rather than a flattened version of the Muon one.
    old_flat = _adamw_names(source_config)
    new_flat = _adamw_names(target_config)
    raise_if_mismatched(state, old_flat)
    skeleton = _fresh_adamw_state_dict(target_config)
    return _convert_state(state, old_flat, new_flat, skeleton,
                          source_config, to_fused=to_fused)


def _fresh_adamw_state_dict(config: ModernConfig) -> dict:
    with torch.device("meta"):
        model = ModernLM(config)
    decay, no_decay = split_adamw_params(model)
    return torch.optim.AdamW([{"params": decay}, {"params": no_decay}]).state_dict()


def raise_if_mismatched(state: dict, names: list[str]) -> None:
    indices = state.get("state", {})
    if indices and max(indices) >= len(names):
        raise ValueError(
            f"optimizer state references parameter {max(indices)} but the model has "
            f"{len(names)}; the checkpoint and config do not describe the same run")


def _fresh_state_dicts(config: ModernConfig) -> list[dict]:
    """Empty optimizer state dicts with the target's groups and indices."""
    with torch.device("meta"):
        model = ModernLM(config)
    optimizer = build_optimizer(model, learning_rate=1.0, muon_learning_rate=1.0,
                                weight_decay=0.0)
    return [child.state_dict() for child in optimizer.optimizers]
