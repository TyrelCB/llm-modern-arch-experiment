#!/usr/bin/env bash
# Build the optional Transformer Engine binding against this machine's PyTorch
# and pip-provided CUDA 13 libraries. The default .venv sees the system nightly
# PyTorch without copying its multi-gigabyte install.
set -euo pipefail

lowp_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lowp_venv_path="${1:-${lowp_repo_root}/.venv}"
lowp_python="${lowp_venv_path}/bin/python"

if [[ ! -x "${lowp_python}" ]]; then
    python3 -m venv --system-site-packages "${lowp_venv_path}"
fi

readarray -t lowp_nvidia_paths < <("${lowp_python}" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

for package in ("cudnn", "nccl"):
    spec = find_spec(f"nvidia.{package}")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"PyTorch environment has no nvidia.{package} package")
    print(Path(next(iter(spec.submodule_search_locations))))
PY
)

lowp_cudnn_path="${lowp_nvidia_paths[0]}"
lowp_nccl_path="${lowp_nvidia_paths[1]}"
export CUDNN_PATH="${lowp_cudnn_path}"
export CPATH="${lowp_cudnn_path}/include:${lowp_nccl_path}/include:${CPATH:-}"
export LIBRARY_PATH="${lowp_cudnn_path}/lib:${lowp_nccl_path}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${lowp_cudnn_path}/lib:${lowp_nccl_path}/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-2}"

"${lowp_python}" -m pip install "setuptools>=68" wheel ninja cmake pybind11
"${lowp_python}" -m pip install --no-build-isolation \
    -e "${lowp_repo_root}[low-precision]"

"${lowp_python}" - <<'PY'
import torch
import transformer_engine.pytorch

print(f"Transformer Engine ready: torch={torch.__version__}, "
      f"GPU={torch.cuda.get_device_name(0)}, capability={torch.cuda.get_device_capability(0)}")
PY
