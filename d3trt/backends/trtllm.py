"""TensorRT-LLM backend: state-once scheduler over the TRT-LLM module stack.

Text requests take this path. The state pass runs once per distinct state and is
cached by content; question blocks are right-padded into batches and run against
the cached KV + GDN snapshot. Media requests are refused here and handled by the
HF backend (which owns the processor + vision tower) until the vision path is
ported; see docs/TENSORRT.md for the GPU-day plan.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import nvtx
from ..errors import BackendUnavailable, MediaNotSupported
from ..model.cache import StateCache
from ..model.trtllm_stack import NativeStack, TrtllmStack, trtllm_available
from .base import Backend


class TrtllmBackend(Backend):
    name = "trtllm"
    stack_class: type[TrtllmStack] = TrtllmStack

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str | None = None,
        dtype: str | None = None,
        tp_size: int = 1,
        rank: int = 0,
        state_cache_bytes: int | None = None,
        recurrent_max_tokens: int = 256,
        quant: str = "auto",
        reference_dir: str | Path | None = None,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = device or "cuda:0"
        self.dtype = dtype or "bfloat16"
        self.tp_size = int(tp_size)
        self.rank = int(rank)
        self.recurrent_max_tokens = int(recurrent_max_tokens)
        self.quant = quant
        self.reference_dir = reference_dir
        self._cache_bytes = state_cache_bytes
        self.frontend: Any = None
        self.stack: TrtllmStack | None = None
        self.cache: StateCache | None = None
        self.last_timings: dict[str, Any] = {}
        self._loaded = False
        self._load_seconds = 0.0

    def attach(self, frontend: Any) -> None:
        self.frontend = frontend

    def start(self) -> None:
        if self._loaded:
            return
        if self.frontend is None:
            raise RuntimeError("TrtllmBackend.attach(frontend) must be called before start()")
        ok, reason = self.stack_class.available()
        if not ok:
            raise BackendUnavailable(self.name, reason)
        started = time.perf_counter()
        self.stack = self.stack_class(
            str(self.model_dir),
            device=self.device,
            dtype=self.dtype,
            tp_size=self.tp_size,
            rank=self.rank,
            recurrent_max_tokens=self.recurrent_max_tokens,
            quant=self.quant,
        )
        self.stack.pad_id = self.frontend.d3.pad_id
        budget = self._cache_bytes
        if budget is None:
            budget = self.frontend.config.state_cache_bytes
        if self._cache_bytes is None:
            budget = _cap_to_free_memory(budget, self.stack.device)
        self.cache = StateCache(budget)
        self._load_seconds = time.perf_counter() - started
        self._loaded = True

    # ------------------------------------------------------------------
    def media_unavailable(self, kind: str) -> str | None:
        if kind in ("image", "video"):
            return (
                f"the {self.name} backend serves text only; use backend='hf' for {kind} "
                "requests until the TensorRT vision path is enabled"
            )
        return None

    def run(self, prepared: Any) -> tuple[dict[str, list[float]], int]:
        if not self._loaded:
            self.start()
        if prepared.images or prepared.videos:
            raise MediaNotSupported(
                f"the {self.name} backend currently serves text; media needs the HF backend"
            )
        keys = list(prepared.runnable)
        if not keys:
            return {}, 0
        assert self.stack is not None and self.cache is not None
        d3 = self.frontend.d3

        reversed_keys = [k for k in keys if k in prepared.reversed_sequences]
        blocks = [(k, False) for k in keys] + [(k, True) for k in reversed_keys]
        sizes = {
            prepared.reversed_states[k] if rev else prepared.states[k] for k, rev in blocks
        }
        if len(sizes) != 1:
            raise RuntimeError(f"the questions of a request have different state blocks: {sorted(sizes)}")
        state_len = sizes.pop()

        first = list(prepared.sequences[keys[0]][:state_len])
        pieces: list[list[int]] = []
        for key, rev in blocks:
            sequence = list(prepared.reversed_sequences[key] if rev else prepared.sequences[key])
            if sequence[:state_len] != first:
                raise RuntimeError("the questions of a request do not share the state block's tokens")
            pieces.append(sequence[state_len:])

        state_key = self.frontend.config.state_key(first)
        import torch

        _sync(self.stack.device)
        started = time.perf_counter()
        entry = self.cache.get(state_key)
        cache_hit = entry is not None
        readout_ms = 0.0
        batches = 0
        with torch.inference_mode():
            with nvtx.range("d3.state"):
                if entry is None:
                    ids = torch.as_tensor([first], dtype=torch.long, device=self.stack.device)
                    snapshot = self.stack.state(ids)
                    entry = self.cache.put(state_key, snapshot, state_len)
                    snapshot_used = snapshot if entry is None else entry.snapshot
                else:
                    snapshot_used = entry.snapshot
            _sync(self.stack.device)
            state_ms = (time.perf_counter() - started) * 1000

            probabilities: list[list[float]] = [[] for _ in pieces]
            # Length-sorted batches keep right-padding waste small; results are
            # written back by index, so the output order is unchanged.
            order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
            question_started = time.perf_counter()
            batch_size = max(1, int(d3.batch_size))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                rows = [pieces[i] for i in batch]
                width = max(map(len, rows))
                ids = torch.full(
                    (len(rows), width), self.stack.pad_id, dtype=torch.long, device=self.stack.device
                )
                for r, piece in enumerate(rows):
                    ids[r, : len(piece)] = torch.as_tensor(
                        piece, dtype=torch.long, device=self.stack.device
                    )
                with nvtx.range("d3.questions"):
                    hidden = self.stack.questions(
                        snapshot_used, ids, [len(p) for p in rows], state_len=state_len
                    )
                counts = [len(prepared.questions[blocks[i][0]].keys) for i in batch]
                r0 = time.perf_counter()
                with nvtx.range("d3.readout"):
                    values = self.stack.probabilities(
                        hidden,
                        counts,
                        temperature=d3.temperature,
                        readout_dtype=d3.readout_dtype,
                    )
                readout_ms += (time.perf_counter() - r0) * 1000
                batches += 1
                for i, value in zip(batch, values):
                    probabilities[i] = value
            _sync(self.stack.device)
            question_ms = (time.perf_counter() - question_started) * 1000 - readout_ms

        from d3_runtime import average_orders

        out: dict[str, list[float]] = {}
        for (key, rev), p in zip(blocks, probabilities):
            if not rev:
                out[key] = p
        for (key, rev), p in zip(blocks, probabilities):
            if rev:
                out[key] = average_orders(out[key], p)
        self.last_timings = {
            "path": self.name,
            "state_ms": round(state_ms, 3),
            "question_ms": round(max(0.0, question_ms), 3),
            "readout_ms": round(readout_ms, 3),
            "batches": batches,
            "cache_hit": cache_hit,
            "state_tokens": state_len,
            "question_tokens": sum(len(p) for p in pieces),
        }
        return out, state_len + sum(len(p) for p in pieces)

    # ------------------------------------------------------------------
    def warmup(self, *, state_tokens: int = 256, questions: int = 8) -> dict[str, float]:
        """Compile kernels and touch the state cache with a synthetic request."""
        if not self._loaded:
            self.start()
        import torch

        assert self.stack is not None
        ids = torch.randint(0, 1000, (1, state_tokens), device=self.stack.device)
        started = time.perf_counter()
        with torch.inference_mode():
            snapshot = self.stack.state(ids)
            widths = 64
            qids = torch.randint(0, 1000, (questions, widths), device=self.stack.device)
            self.stack.questions(snapshot, qids, [widths] * questions, state_len=state_tokens)
        torch.cuda.synchronize()
        return {"warmup_ms": round((time.perf_counter() - started) * 1000, 3)}

    def info(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.name,
            "device": self.device,
            "dtype": self.dtype,
            "tp_size": self.tp_size,
            "quant": self.stack.quant if self.stack is not None else self.quant,
            "load_seconds": round(self._load_seconds, 3),
        }
        if self.stack is not None:
            payload["stack"] = self.stack.info()
        if self.cache is not None:
            payload["state_cache"] = self.cache.stats()
        return payload

    def close(self) -> None:
        self.stack = None
        self.cache = None
        self._loaded = False
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _cap_to_free_memory(budget: int, device: Any, fraction: float = 0.6) -> int:
    """Lower the default cache budget so small cards (L40 48 GB, A100 40 GB) keep headroom.

    Only applied when the caller gave no explicit budget; activations for a long
    state pass need the rest of the free memory. Never raises the budget.
    """
    try:
        import torch

        if str(device).startswith("cuda") or getattr(device, "type", None) == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            return min(int(budget), int(free * fraction))
    except Exception:  # noqa: BLE001
        pass
    return int(budget)


def _sync(device: Any) -> None:
    try:
        import torch

        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        elif str(device).startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


class NativeBackend(TrtllmBackend):
    """Same scheduler and kernels, built from plain PyTorch modules: no
    TensorRT-LLM and no Transformers on the hot path."""

    name = "native"
    stack_class = NativeStack
