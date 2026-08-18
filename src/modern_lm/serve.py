"""Interactive single-checkpoint inference server.

This is a *inspection* tool, not an evaluation path. Nothing here may be used to
produce a benchmark number: it samples, it streams, and it lets the operator
change the prompt format, none of which the scored harness does. Benchmark
claims keep coming from `evaluate_benchmarks.py`, whose greedy/no-cache decode
is deliberately locked to the reference's semantics.

Decoding is reimplemented here rather than calling `ModernLM.generate` because
that method is greedy-only, returns just the final sequence, and mirrors the
reference signature on purpose. Streaming and temperature/top-p live here so
that contract stays untouched -- with `temperature=0` this loop is plain greedy
argmax and agrees token-for-token with `generate(use_cache=True)`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from .config import ModernConfig
from .data import default_paths
from .model import ModernLM

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

# The prompt format the SFT stage and the benchmark harness both use. Exposed in
# the UI as a template so an operator can see raw base-model behaviour too, but
# this default is the one the model was actually trained to answer.
DEFAULT_TEMPLATE = "Question: {prompt}\nAnswer:"

# Checkpoint filenames carry their step/token count; sort on that number rather
# than lexically so checkpoint-000600014848 follows checkpoint-000400031744.
_STEP_IN_NAME = re.compile(r"(\d+)")


def _sort_key(path: Path) -> tuple[int, int | str]:
    if path.name == "latest.pt":
        return (2, 0)  # pinned last; it is a duplicate of some numbered file
    match = _STEP_IN_NAME.search(path.stem)
    return (0, int(match.group(1))) if match else (1, path.stem)


def _display_path(path: Path) -> str:
    """Repo-relative when possible, absolute otherwise.

    Only `resolve_checkpoint` enforces containment; this is presentation, so a
    path from outside the tree (a custom --runs-dir, a test tmpdir) must render
    rather than raise.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def discover_checkpoints(runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Group every `runs/*/**.pt` under its run directory, newest run first."""
    runs = []
    for directory in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        files = sorted(directory.glob("*.pt"), key=_sort_key)
        if not files:
            continue
        runs.append({
            "run": directory.name,
            "modified": max(f.stat().st_mtime for f in files),
            "checkpoints": [
                {"path": _display_path(f),
                 "name": f.name,
                 "size_mb": round(f.stat().st_size / 1e6, 1)}
                for f in files
            ],
        })
    runs.sort(key=lambda r: r["modified"], reverse=True)
    return runs


def resolve_checkpoint(raw: str) -> Path:
    """Resolve a client-supplied path, refusing anything outside the repo.

    The UI only ever sends paths that came from `discover_checkpoints`, but this
    server binds a port and the parameter reaches `torch.load`, which executes
    pickles. Containment is checked against the resolved path so `..` segments
    and symlinks out of the tree are both rejected.
    """
    candidate = (REPO_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not candidate.is_relative_to(REPO_ROOT):
        raise ValueError("checkpoint path must stay inside the repository")
    if candidate.suffix != ".pt" or not candidate.is_file():
        raise ValueError(f"not a checkpoint file: {raw}")
    return candidate


def _describe(payload: dict, model: ModernLM, path: Path) -> dict:
    """Pull the run metadata the checkpoint already carries into a flat summary."""
    state = payload.get("state") or {}
    evaluation = payload.get("evaluation") or {}
    config = model.config
    embedding = config.vocab_size * config.dim
    head = 0 if config.tie_embeddings else config.vocab_size * config.dim
    total = model.num_params()
    return {
        "path": _display_path(path),
        "run": path.parent.name,
        "name": path.name,
        "stage": payload.get("stage", "pretrain"),
        "total_params": total,
        "non_embedding_params": total - embedding - head,
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "max_seq_len": config.max_seq_len,
        "use_siamese_norm": config.use_siamese_norm,
        "optimizer_step": state.get("optimizer_step"),
        "tokens_seen": state.get("tokens_seen") or state.get("supervised_tokens_seen"),
        "examples_seen": state.get("examples_seen"),
        "eval_loss": evaluation.get("loss"),
        "eval_perplexity": evaluation.get("perplexity"),
    }


@dataclass
class LoadedModel:
    model: ModernLM
    info: dict
    device: torch.device


class ModelRegistry:
    """Holds at most one model. Loads are serialized; the GPU here is shared.

    Only one checkpoint stays resident because this box routinely has ~60GB of
    llama-server resident alongside training jobs, and a UI that silently
    accumulated models would OOM the machine's real work.
    """

    def __init__(self, tokenizer_path: Path):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None

    @property
    def current(self) -> LoadedModel | None:
        return self._loaded

    def load(self, path: Path, device_name: str) -> dict:
        with self._lock:
            self.unload_locked()
            device = torch.device(device_name)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = ModernLM(ModernConfig(**payload["config"]))
            model.load_state_dict(payload["model"])
            info = _describe(payload, model, path)
            del payload
            model.to(device).eval()
            info["device"] = str(device)
            self._loaded = LoadedModel(model=model, info=info, device=device)
            return info

    def unload_locked(self) -> None:
        if self._loaded is None:
            return
        was_cuda = self._loaded.device.type == "cuda"
        self._loaded = None
        if was_cuda:
            torch.cuda.empty_cache()

    def unload(self) -> None:
        with self._lock:
            self.unload_locked()

    @torch.no_grad()
    def stream(self, prompt: str, max_new_tokens: int, temperature: float,
               top_p: float, seed: int | None, stop_on_eos: bool) -> Iterator[dict]:
        """Yield one event per decoded token, then a final `done` event.

        Holds the registry lock for the whole generation so a concurrent load
        cannot swap the weights out from under an in-flight decode.
        """
        with self._lock:
            if self._loaded is None:
                raise RuntimeError("no checkpoint is loaded")
            model, device = self._loaded.model, self._loaded.device
            config = model.config

            ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
            budget = config.max_seq_len - max_new_tokens
            truncated = len(ids) > budget
            ids = ids[-budget:] if truncated else ids
            if not ids:
                raise ValueError("prompt is empty after truncation")

            generator = None
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed)

            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            caches: list[dict] | None = [{} for _ in model.blocks]
            amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

            emitted: list[int] = []
            text_so_far = ""
            started = time.perf_counter()
            first_token_at: float | None = None
            finish = "length"

            for step in range(max_new_tokens):
                if caches is not None and input_ids.shape[1] >= config.max_seq_len:
                    caches = None  # cache cannot be extended past the context
                if caches is not None:
                    step_input = input_ids if step == 0 else input_ids[:, -1:]
                else:
                    step_input = input_ids[:, -config.max_seq_len:]

                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(step_input, caches=caches).logits[:, -1, :]
                logits = logits.float()

                if temperature <= 0:
                    next_token = logits.argmax(dim=-1)
                else:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    if 0 < top_p < 1:
                        probabilities = _top_p_filter(probabilities, top_p)
                    next_token = torch.multinomial(probabilities, 1, generator=generator)[:, 0]

                token_id = int(next_token)
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                if stop_on_eos and token_id == self.eos_id:
                    finish = "eos"
                    break

                emitted.append(token_id)
                input_ids = torch.cat([input_ids, next_token[:, None]], dim=1)

                # Decode the whole run each step and emit only the delta: a
                # byte-level BPE token can be a partial UTF-8 sequence, which
                # decodes to a replacement char on its own but resolves once its
                # continuation arrives.
                decoded = self.tokenizer.decode(emitted)
                if decoded != text_so_far:
                    yield {"type": "token", "text": decoded[len(text_so_far):]}
                    text_so_far = decoded

            elapsed = time.perf_counter() - started
            yield {
                "type": "done",
                "finish_reason": finish,
                "prompt_tokens": len(ids),
                "prompt_truncated": truncated,
                "generated_tokens": len(emitted),
                "elapsed_seconds": round(elapsed, 3),
                "tokens_per_second": round(len(emitted) / elapsed, 2) if elapsed > 0 else None,
                "time_to_first_token": (round(first_token_at - started, 3)
                                        if first_token_at else None),
                "text": text_so_far,
            }


def _top_p_filter(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero all but the smallest prefix of mass >= top_p, then renormalize."""
    sorted_probabilities, indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    # Keep the token that crosses the threshold, so at least one always survives.
    remove = cumulative - sorted_probabilities > top_p
    sorted_probabilities[remove] = 0.0
    kept = torch.zeros_like(probabilities).scatter_(-1, indices, sorted_probabilities)
    return kept / kept.sum(dim=-1, keepdim=True)


# Request models live at module scope, not inside build_app: FastAPI resolves a
# handler's type hints against its module globals, so a locally-defined model is
# invisible to it and every field silently degrades into a query parameter.
class LoadRequest(BaseModel):
    path: str
    device: str = "cpu"


class GenerateRequest(BaseModel):
    prompt: str
    template: str = DEFAULT_TEMPLATE
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int | None = None
    stop_on_eos: bool = True


def build_app(registry: ModelRegistry):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse

    app = FastAPI(title="ModernLM playground")
    index_html = (Path(__file__).parent / "static" / "playground.html").read_text()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return index_html

    @app.get("/api/checkpoints")
    def checkpoints() -> dict:
        return {
            "runs": discover_checkpoints(),
            "loaded": registry.current.info if registry.current else None,
            "cuda_available": torch.cuda.is_available(),
            "default_template": DEFAULT_TEMPLATE,
        }

    @app.post("/api/load")
    def load(request: LoadRequest) -> dict:
        try:
            path = resolve_checkpoint(request.path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.device.startswith("cuda") and not torch.cuda.is_available():
            raise HTTPException(status_code=400, detail="CUDA is not available")
        try:
            return registry.load(path, request.device)
        except torch.cuda.OutOfMemoryError as exc:
            registry.unload()
            raise HTTPException(
                status_code=507,
                detail="GPU out of memory -- this box shares the GPU with training "
                       "jobs and llama-server. Load on CPU instead.") from exc
        except Exception as exc:  # bad state dict, corrupt file, config mismatch
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/api/unload")
    def unload() -> dict:
        registry.unload()
        return {"loaded": None}

    @app.post("/api/generate")
    async def generate(request: GenerateRequest) -> StreamingResponse:
        if registry.current is None:
            raise HTTPException(status_code=409, detail="load a checkpoint first")
        if "{prompt}" in request.template:
            text = request.template.replace("{prompt}", request.prompt)
        else:
            text = request.prompt

        async def events():
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def produce():
                # Decode is synchronous and GPU/CPU-bound; run it off the event
                # loop so the server keeps serving while a long generation runs.
                try:
                    for event in registry.stream(text, request.max_new_tokens,
                                                 request.temperature, request.top_p,
                                                 request.seed, request.stop_on_eos):
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                except Exception as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            threading.Thread(target=produce, daemon=True).start()
            yield f"data: {json.dumps({'type': 'start', 'prompt': text})}\n\n"
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


def _listen_urls(host: str, port: int) -> list[str]:
    """URLs a browser can actually open.

    Binding to 0.0.0.0 means "every interface", which is not an address anyone
    can type; print the concrete ones instead so the LAN URL is copyable.
    """
    if host not in ("0.0.0.0", "::"):
        return [f"http://{host}:{port}"]

    # Resolving the hostname is not enough -- on this box it maps to 127.0.1.1,
    # which is exactly the address that does not help a remote browser. Ask the
    # routing table which source address reaches off-box instead; the UDP
    # connect is address selection only and sends no packets.
    addresses = []
    import socket

    for probe in ("8.8.8.8", "1.1.1.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 53))
                address = sock.getsockname()[0]
            if address not in addresses and not address.startswith("127."):
                addresses.append(address)
        except OSError:  # no route off-box; loopback still works
            continue
    return [f"http://{a}:{port}" for a in ["127.0.0.1", *addresses]]


def main() -> None:
    paths = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=paths["tokenizer"])
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach the playground from another machine. "
                             "There is no authentication, so anyone who can route "
                             "to this port can load checkpoints and generate.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional checkpoint to load at startup")
    parser.add_argument("--device", default="cpu",
                        help="device for a startup --checkpoint, and the UI default")
    args = parser.parse_args()

    import uvicorn

    registry = ModelRegistry(args.tokenizer)
    if args.checkpoint is not None:
        info = registry.load(resolve_checkpoint(str(args.checkpoint)), args.device)
        print(json.dumps({"event": "loaded", **info}, indent=2), flush=True)
    for url in _listen_urls(args.host, args.port):
        print(f"playground on {url}", flush=True)
    uvicorn.run(build_app(registry), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
