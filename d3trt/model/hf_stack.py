"""Transformers Qwen3.5 adapter for :class:`D3Core`."""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch

from .. import nvtx
from .cache import StateCache
from .core import D3Core, Snapshot


class HFStack:
    """A Qwen3.5 model's text layers, with the reference math."""

    def __init__(self, backbone: Any, readout: torch.Tensor, *, device: torch.device):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        language = backbone.language_model
        self.backbone = backbone
        self.language = language
        self.device = device
        self.readout = readout
        self.embed = backbone.get_input_embeddings()
        self.core = D3Core(
            layers=language.layers,
            layer_types=language.config.layer_types,
            final_norm=language.norm,
            rotary_emb=language.rotary_emb,
            apply_rotary=apply_rotary_pos_emb,
            attention_state="flash",
            attention_question="triton",
            # Match the reference chunk kernel exactly on the parity path.
            recurrent_max_tokens=0,
        )
        self.rotary = language.rotary_emb
        self.pad_id = None
        self.max_options = int(readout.shape[0])
        self.last_timings: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def state(self, ids: torch.Tensor) -> Snapshot:
        """``ids`` [1, S] -> snapshot."""
        embeds = self.embed(ids)
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, 1, -1)
        rope = self.rotary(embeds, positions)
        return self.core.state(embeds, rope)

    def questions(
        self,
        snapshot: Snapshot,
        ids: torch.Tensor,
        lengths: Sequence[int],
        *,
        state_len: int,
    ) -> torch.Tensor:
        """``ids`` [B, W] right-padded -> final-norm hidden at each row's last token."""
        width = ids.shape[1]
        embeds = self.embed(ids)
        steps = (state_len + torch.arange(width, device=ids.device)).view(1, 1, -1)
        rope = self.rotary(embeds, steps.expand(3, ids.shape[0], -1))
        return self.core.questions(snapshot, embeds, lengths, rope)

    # ------------------------------------------------------------------
    def probabilities(
        self,
        hidden: torch.Tensor,
        counts: Sequence[int],
        *,
        temperature: float,
        readout_dtype: str = "float32",
    ) -> list[list[float]]:
        if readout_dtype == "float32":
            logits = hidden.float() @ self.readout.T
        else:
            logits = torch.nn.functional.linear(hidden, self.readout).float()
        counts_t = torch.as_tensor(list(counts), device=hidden.device)[:, None]
        logits = logits.masked_fill(
            torch.arange(self.max_options, device=hidden.device)[None] >= counts_t, float("-inf")
        )
        values = (logits / temperature).softmax(-1).cpu().tolist()
        return [row[:count] for row, count in zip(values, counts)]

    # ------------------------------------------------------------------
    def run_text(
        self,
        prepared: Any,
        *,
        batch_size: int,
        temperature: float,
        readout_dtype: str,
        cache: StateCache | None = None,
        state_key: str | None = None,
        lengths_by_key: dict[str, int] | None = None,
    ) -> tuple[dict[str, list[float]], int]:
        """Reference ``_run_state_once`` with an optional cross-request cache."""
        keys = list(prepared.runnable)
        if not keys:
            return {}, 0
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
            sequence = list(
                prepared.reversed_sequences[key] if rev else prepared.sequences[key]
            )
            if sequence[:state_len] != first:
                raise RuntimeError("the questions of a request do not share the state block's tokens")
            pieces.append(sequence[state_len:])

        ids = torch.as_tensor([first], dtype=torch.long, device=self.device)
        _sync(self.device)
        started = time.perf_counter()
        cache_hit = False
        with nvtx.range("d3.state"):
            if cache is not None and state_key is not None:
                entry = cache.get(state_key)
                if entry is None:
                    snapshot = self.state(ids)
                    entry = cache.put(state_key, snapshot, state_len)
                    snapshot_used = snapshot if entry is None else entry.snapshot
                else:
                    snapshot_used = entry.snapshot
                    cache_hit = True
            else:
                snapshot_used = self.state(ids)
        _sync(self.device)
        state_ms = (time.perf_counter() - started) * 1000
        last_position = state_len - 1

        probabilities: list[list[float]] = [[] for _ in pieces]
        question_started = time.perf_counter()
        readout_ms = 0.0
        batches = 0
        order = list(range(len(pieces)))
        for start in range(0, len(order), max(1, batch_size)):
            batch = order[start : start + max(1, batch_size)]
            rows = [pieces[i] for i in batch]
            width = max(map(len, rows))
            ids = torch.full(
                (len(rows), width), self._pad_id(), dtype=torch.long, device=self.device
            )
            for r, piece in enumerate(rows):
                ids[r, : len(piece)] = torch.as_tensor(piece, dtype=torch.long, device=self.device)
            with nvtx.range("d3.questions"):
                hidden = self.questions(
                    snapshot_used, ids, [len(p) for p in rows], state_len=last_position + 1
                )
            counts = [len(prepared.questions[blocks[i][0]].keys) for i in batch]
            r0 = time.perf_counter()
            with nvtx.range("d3.readout"):
                values = self.probabilities(
                    hidden, counts, temperature=temperature, readout_dtype=readout_dtype
                )
            readout_ms += (time.perf_counter() - r0) * 1000
            batches += 1
            for i, value in zip(batch, values):
                probabilities[i] = value
        _sync(self.device)
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
            "path": "core",
            "state_ms": round(state_ms, 3),
            "question_ms": round(max(0.0, question_ms), 3),
            "readout_ms": round(readout_ms, 3),
            "batches": batches,
            "cache_hit": cache_hit,
            "state_tokens": state_len,
            "question_tokens": sum(len(p) for p in pieces),
        }
        return out, state_len + sum(len(p) for p in pieces)

    def _pad_id(self) -> int:
        if self.pad_id is None:
            raise RuntimeError("pad_id must be set before run_text")
        return int(self.pad_id)


def _sync(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass
