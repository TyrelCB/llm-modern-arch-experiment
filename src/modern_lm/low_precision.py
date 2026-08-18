"""Transformer Engine FP8/NVFP4 integration without checkpoint lock-in.

The model keeps FP32 master parameters and the embedding, norms, router, and
vocabulary head at the normal BF16-autocast precision.  Only the GEMM-heavy
block projections are replaced by Transformer Engine Linear modules.  Their
quantized workspaces are runtime caches: checkpoints contain the canonical
high-precision weights and can be loaded by either a BF16 or low-precision
model.

Transformer Engine is an optional dependency and is imported only when a low
precision mode is requested.  CPU-only development and the default BF16 path
therefore do not require NVIDIA packages.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
from dataclasses import asdict, dataclass
import glob
import importlib.util
from importlib import metadata
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


SUPPORTED_PRECISIONS = ("bf16", "fp8", "nvfp4")
_RUNTIME_ATTRIBUTE = "_modern_lm_low_precision_runtime"
_TE_LINEAR_ATTRIBUTE = "_modern_lm_transformer_engine_linear"
_FIRST_MICROBATCH: ContextVar[bool | None] = ContextVar(
    "modern_lm_first_low_precision_microbatch", default=None)


@dataclass(frozen=True)
class PrecisionReport:
    """Serializable description of the numerical path used by a run."""

    precision: str
    backend: str
    backend_version: str | None
    recipe: str
    compute_capability: str | None
    converted_linears: int
    skipped_linears: tuple[str, ...] = ()
    disabled_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PrecisionRuntime:
    te: Any
    recipe: Any
    report: PrecisionReport
    converted_names: tuple[str, ...]


def validate_precision(precision: str) -> str:
    precision = precision.lower()
    if precision not in SUPPORTED_PRECISIONS:
        choices = ", ".join(SUPPORTED_PRECISIONS)
        raise ValueError(f"precision must be one of: {choices}")
    return precision


def _unwrap(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _load_transformer_engine() -> tuple[Any, Any, str]:
    # PyTorch's CUDA-13 wheels install cuDNN/NCCL under the ``nvidia`` Python
    # namespace rather than the system loader path. TE's core links both by
    # SONAME. Preloading the wheel libraries makes the optional backend work
    # without requiring every invocation to hand-maintain LD_LIBRARY_PATH.
    library_paths: list[str] = []
    for package, env_name in (("cudnn", "CUDNN_HOME"), ("nccl", "NCCL_HOME")):
        spec = importlib.util.find_spec(f"nvidia.{package}")
        if spec is None or not spec.submodule_search_locations:
            continue
        root = Path(next(iter(spec.submodule_search_locations)))
        os.environ.setdefault(env_name, str(root))
        library_paths.extend(glob.glob(str(root / "lib" / "*.so*")))

    remaining = list(dict.fromkeys(library_paths))
    # Some cuDNN component libraries depend on peers in the same directory.
    # Retrying lets their dependencies load first without baking in a fragile
    # filename order.
    for _ in range(len(remaining) + 1):
        progress = False
        for path in remaining[:]:
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            remaining.remove(path)
            progress = True
        if not remaining or not progress:
            break

    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            "FP8/NVFP4 requires Transformer Engine. Install the project's "
            "low-precision extra (python -m pip install -e '.[low-precision]') "
            "and ensure the pip cuDNN library directory is visible; see "
            "docs/low-precision.md."
        ) from error
    try:
        version = metadata.version("transformer-engine")
    except metadata.PackageNotFoundError:  # source/editable TE install
        version = getattr(te, "__version__", "unknown")
    return te, recipe, version


def _recipe_for(precision: str, recipe_module: Any,
                capability: tuple[int, int]) -> tuple[Any, tuple[str, ...]]:
    if precision == "fp8":
        # DelayedScaling produced zero gradients on the CUDA 13 / GB10 stack.
        # Current scaling is stateless, gives finite gradients, and makes
        # checkpoint interchange straightforward.
        return recipe_module.Float8CurrentScaling(), ()

    major, minor = capability
    if major < 10:
        raise RuntimeError(
            f"NVFP4 requires a Blackwell-class GPU; found sm_{major}{minor}")

    options: dict[str, bool] = {}
    disabled: list[str] = []
    # TE 2.18's stochastic FP4 conversion is explicitly compiled only for
    # sm_100 and sm_103.  GB10 is sm_121: its deterministic FP4 conversion,
    # 2-D weight scaling, and RHT kernels are supported and GPU-validated, but
    # requesting stochastic rounding triggers a device-side assertion.
    if capability not in ((10, 0), (10, 3)):
        options["disable_stochastic_rounding"] = True
        disabled.append("stochastic_rounding")

    # Workstation sm_120 exposes less opt-in shared memory than the RHT kernel
    # requests.  GB10/sm_121 does not share this limitation in the measured
    # projection shapes, so it keeps RHT enabled.
    if capability == (12, 0):
        options["disable_rht"] = True
        disabled.append("random_hadamard_transform")

    return recipe_module.NVFP4BlockScaling(**options), tuple(disabled)


def _make_te_linear_class(te: Any) -> type[nn.Module]:
    class ModernTELinear(te.Linear):
        """TE Linear that honors this project's BF16 autocast boundary.

        Embedding and FP32 residual operations can hand a projection an FP32
        tensor even inside torch.autocast.  PyTorch's native Linear casts that
        tensor internally; TE Linear expects the caller to do so, and NVFP4's
        RHT accepts BF16 rather than FP32.  This cast restores the same boundary
        the old Linear had while retaining FP32 master weights.
        """

        _modern_lm_transformer_engine_linear = True

        @torch.compiler.disable
        def forward(self, inp: torch.Tensor, *args: Any, **kwargs: Any):
            if inp.is_cuda and torch.is_autocast_enabled("cuda"):
                autocast_dtype = torch.get_autocast_dtype("cuda")
                if inp.dtype != autocast_dtype:
                    inp = inp.to(autocast_dtype)
            # TE can skip gradient accumulation on the first microbatch and
            # reuse its quantized weight on later accumulation microbatches.
            # None retains TE's conservative behavior for evaluation and callers
            # that do not declare an update boundary.
            kwargs.setdefault("is_first_microbatch", _FIRST_MICROBATCH.get())
            return super().forward(inp, *args, **kwargs)

    return ModernTELinear


def _eligible_projection(path: str, leaf_name: str) -> bool:
    """Select hidden block projections, not sensitive or output linears."""

    if leaf_name in ("lm_head", "router"):
        return False
    return path.startswith("blocks.") or path.startswith("mtp.")


def _replace_linears(model: nn.Module, te: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    linear_type = _make_te_linear_class(te)
    converted: list[str] = []
    skipped: list[str] = []

    # TE construction initializes temporary weights before the canonical ones
    # are copied over.  Preserve RNG streams so merely choosing the numerical
    # backend cannot shift data/dropout/stochastic-rounding sequences.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        def visit(parent: nn.Module, prefix: str = "") -> None:
            for name, child in list(parent.named_children()):
                path = f"{prefix}.{name}" if prefix else name
                if type(child) is nn.Linear and _eligible_projection(path, name):
                    if child.in_features % 16 or child.out_features % 16:
                        skipped.append(
                            f"{path} ({child.in_features}x{child.out_features}: not 16-aligned)")
                        continue
                    replacement = linear_type(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        params_dtype=child.weight.dtype,
                        device=child.weight.device,
                        name=path,
                    )
                    with torch.no_grad():
                        replacement.weight.copy_(child.weight)
                        replacement.weight.requires_grad_(child.weight.requires_grad)
                        if child.bias is not None:
                            replacement.bias.copy_(child.bias)
                            replacement.bias.requires_grad_(child.bias.requires_grad)
                    if hasattr(child, "muon_row_blocks"):
                        replacement.muon_row_blocks = child.muon_row_blocks
                    replacement.train(child.training)
                    setattr(parent, name, replacement)
                    converted.append(path)
                else:
                    visit(child, path)

        visit(model)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return tuple(converted), tuple(skipped)


def configure_low_precision(model: nn.Module, precision: str,
                            device: torch.device) -> PrecisionReport:
    """Configure a model for BF16, FP8, or NVFP4 training.

    Must run after moving the model to its CUDA device and before constructing
    the optimizer or calling ``torch.compile``.
    """

    precision = validate_precision(precision)
    target = _unwrap(model)
    if hasattr(target, _RUNTIME_ATTRIBUTE):
        runtime = getattr(target, _RUNTIME_ATTRIBUTE)
        if runtime.report.precision != precision:
            raise RuntimeError("model precision was already configured")
        return runtime.report

    if precision == "bf16":
        report = PrecisionReport(
            precision="bf16",
            backend="torch",
            backend_version=torch.__version__,
            recipe="bf16_autocast",
            compute_capability=None,
            converted_linears=0,
        )
        setattr(target, _RUNTIME_ATTRIBUTE,
                _PrecisionRuntime(None, None, report, ()))
        return report

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"{precision} training requires a CUDA device")

    te, recipe_module, version = _load_transformer_engine()
    capability = torch.cuda.get_device_capability(device)
    recipe, disabled = _recipe_for(precision, recipe_module, capability)
    converted, skipped = _replace_linears(target, te)
    if not converted:
        raise RuntimeError("no 16-aligned transformer projections were available to convert")

    report = PrecisionReport(
        precision=precision,
        backend="transformer_engine",
        backend_version=version,
        recipe=type(recipe).__name__,
        compute_capability=f"sm_{capability[0]}{capability[1]}",
        converted_linears=len(converted),
        skipped_linears=skipped,
        disabled_features=disabled,
    )
    setattr(target, _RUNTIME_ATTRIBUTE,
            _PrecisionRuntime(te, recipe, report, converted))
    return report


def precision_report(model: nn.Module) -> PrecisionReport | None:
    runtime = getattr(_unwrap(model), _RUNTIME_ATTRIBUTE, None)
    return None if runtime is None else runtime.report


@contextmanager
def low_precision_autocast(model: nn.Module,
                           is_first_microbatch: bool | None = None):
    """Context for TE projections and their accumulation boundary."""

    runtime = getattr(_unwrap(model), _RUNTIME_ATTRIBUTE, None)
    if runtime is None or runtime.te is None:
        yield
        return
    token = _FIRST_MICROBATCH.set(is_first_microbatch)
    try:
        with runtime.te.autocast(enabled=True, recipe=runtime.recipe):
            yield
    finally:
        _FIRST_MICROBATCH.reset(token)


def canonical_model_state_dict(model: nn.Module) -> OrderedDict[str, Any]:
    """Return a checkpoint state dict without TE's disposable cache entries."""

    target = _unwrap(model)
    state = target.state_dict()
    te_prefixes = {
        name for name, module in target.named_modules()
        if getattr(module, _TE_LINEAR_ATTRIBUTE, False)
    }
    return OrderedDict(
        (key, value) for key, value in state.items()
        if not (key.endswith("._extra_state") and key[:-len("._extra_state")] in te_prefixes)
    )


def _is_disposable_te_extra_state(target: nn.Module, key: str) -> bool:
    """Whether ``key`` can only be a TE cache for a hidden projection."""

    suffix = "._extra_state"
    if not key.endswith(suffix):
        return False
    path = key[:-len(suffix)]
    modules = dict(target.named_modules())
    module = modules.get(path)
    if module is None:
        return False
    leaf_name = path.rsplit(".", 1)[-1]
    return (
        getattr(module, _TE_LINEAR_ATTRIBUTE, False)
        or (isinstance(module, nn.Linear) and _eligible_projection(path, leaf_name))
    )


def load_canonical_model_state_dict(model: nn.Module,
                                    state: Mapping[str, Any]) -> None:
    """Load either a canonical or old TE-bearing checkpoint strictly.

    Transformer Engine adds ``_extra_state`` entries for quantized workspaces.
    Current-scaling FP8 and NVFP4 rebuild those caches from the canonical master
    weights, so accepting only those missing keys preserves normal strict-load
    protection while allowing BF16 <-> FP8/NVFP4 resumes.
    """

    target = _unwrap(model)
    clean = OrderedDict(
        (key, value) for key, value in state.items()
        if not _is_disposable_te_extra_state(target, key)
    )
    incompatible = target.load_state_dict(clean, strict=False)
    allowed_missing = {
        f"{name}._extra_state" for name, module in target.named_modules()
        if getattr(module, _TE_LINEAR_ATTRIBUTE, False)
    }
    missing = [key for key in incompatible.missing_keys if key not in allowed_missing]
    unexpected = [key for key in incompatible.unexpected_keys
                  if not _is_disposable_te_extra_state(target, key)]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if unexpected:
            details.append(f"unexpected keys: {unexpected}")
        raise RuntimeError("model checkpoint mismatch (" + "; ".join(details) + ")")
