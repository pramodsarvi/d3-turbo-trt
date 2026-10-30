"""Prefill-only d3 backbone.

The forward pass is deliberately independent of the module library: the same
code runs on Transformers' Qwen3.5 layers (CPU parity) and on TensorRT-LLM
``Linear``/``RMSNorm``/``GatedMLP`` modules (GPU fast path). A layer only has to
expose the attributes below; both adapters provide them.

This is the exact math of the reference ``StateOnce`` (state block
bidirectionally over itself, question block over state + itself, GDN causal with
a carried recurrent/conv state), with the attention and delta-rule calls routed
through :mod:`d3trt.model.ops` so the hot kernels are swappable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from . import ops

_WINDOW = 4


@dataclass(slots=True)
class AttentionEntry:
    keys: torch.Tensor      # [1, H_kv, S, D]
    values: torch.Tensor    # [1, H_kv, S, D]


@dataclass(slots=True)
class DeltaEntry:
    tail: torch.Tensor       # [1, conv_dim, window]
    recurrent: torch.Tensor  # [1, H_v, D_k, D_v]


Snapshot = list[AttentionEntry | DeltaEntry]


@dataclass(slots=True)
class CoreStats:
    state_layers: int = 0
    question_layers: int = 0
    attention_state: str = "sdpa"
    attention_question: str = "sdpa"
    delta: str = "torch"


class D3Core:
    """The 64-layer d3 backbone for one state and its question blocks."""

    def __init__(
        self,
        *,
        layers: Sequence[Any],
        layer_types: Sequence[str],
        final_norm: Any,
        rotary_emb: Any,
        apply_rotary: Any,
        delta_impl: Any = None,
        attention_state: str = "auto",
        attention_question: str = "auto",
        compile_layers: bool | str = False,
        recurrent_max_tokens: int = 0,
    ):
        if len(layers) != len(layer_types):
            raise ValueError("one layer type per layer is required")
        self.layers = list(layers)
        self.layer_types = list(layer_types)
        self.final_norm = final_norm
        self.rotary_emb = rotary_emb
        self.apply_rotary = apply_rotary
        self.delta = delta_impl or DefaultDeltaOps()
        self.attention_state = attention_state
        self.attention_question = attention_question
        self.compile_layers = compile_layers
        self.recurrent_max_tokens = int(recurrent_max_tokens)
        self._compiled: dict[int, Any] = {}
        self.stats = CoreStats()
        self.profile_layers = False
        self._layer_spans: list[tuple[str, int, str, Any, Any, float]] = []
        self._layer_profile_device: Any = None

    # ------------------------------------------------------------------
    @property
    def depth(self) -> int:
        return len(self.layers)

    def _compiles(self, phase: str) -> bool:
        """``compile_layers``: False, True / "all", "state" or "questions"."""
        mode = self.compile_layers
        if mode is True or mode == "all":
            return True
        return bool(mode) and mode == phase

    def _layer_call(self, index: int, fn: Any, *args: Any) -> Any:
        # The phase ("state" / "questions") is _layer's last argument.
        if not self._compiles(args[-1]):
            return fn(*args)
        compiled = self._compiled.get(index)
        if compiled is None:
            # Every layer is the same function with different modules, so dynamo
            # keeps one entry per layer (and per dynamic-shape specialisation).
            # The default limit of 8 would silently run most layers eagerly.
            import torch._dynamo.config as dynamo_config

            limit = 4 * len(self.layers)
            for name in ("recompile_limit", "cache_size_limit"):
                if getattr(dynamo_config, name, limit) < limit:
                    setattr(dynamo_config, name, limit)
            if getattr(dynamo_config, "accumulated_recompile_limit", limit) < limit:
                dynamo_config.accumulated_recompile_limit = limit
            compiled = torch.compile(fn, mode="max-autotune-no-cudagraphs", fullgraph=False)
            self._compiled[index] = compiled
        return compiled(*args)

    def reset_stats(self) -> None:
        self.stats = CoreStats()

    def begin_layer_profile(self) -> None:
        self._layer_spans.clear()
        self._layer_profile_device = None
        self.profile_layers = True

    def end_layer_profile(self) -> dict[str, Any]:
        self.profile_layers = False
        cuda = False
        try:
            cuda = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            cuda = False
        if cuda:
            torch.cuda.synchronize(self._layer_profile_device)
        raw = []
        for phase, index, kind, start, end, cpu_ms in self._layer_spans:
            if cuda and start is not None and end is not None:
                ms = float(start.elapsed_time(end))
            else:
                ms = float(cpu_ms)
            raw.append({"phase": phase, "index": index, "kind": kind, "ms": round(ms, 4)})
        aggregate: dict[tuple[int, str], dict[str, Any]] = {}
        for row in raw:
            key = (row["index"], row["kind"])
            entry = aggregate.setdefault(
                key,
                {"index": row["index"], "kind": row["kind"], "calls": 0, "total_ms": 0.0,
                 "state_ms": 0.0, "question_ms": 0.0},
            )
            entry["calls"] += 1
            entry["total_ms"] += row["ms"]
            if row["phase"] == "state":
                entry["state_ms"] += row["ms"]
            else:
                entry["question_ms"] += row["ms"]
        layers = sorted(aggregate.values(), key=lambda item: item["index"])
        for entry in layers:
            for key in ("total_ms", "state_ms", "question_ms"):
                entry[key] = round(entry[key], 3)
        return {"layers": layers, "raw": raw, "gpu": cuda}

    # ------------------------------------------------------------------
    def state(self, embeds: torch.Tensor, rope: Any) -> Snapshot:
        """Run the state block once, [1, S, D] -> per-layer KV / GDN state."""
        hidden = embeds
        cache: Snapshot = []
        for index, (layer, kind) in enumerate(zip(self.layers, self.layer_types)):
            hidden, entry = self._layer_call(
                index, self._layer, index, layer, kind, hidden, rope, None, None, None, "state"
            )
            cache.append(entry)
        self.stats.state_layers = len(cache)
        return cache

    def questions(
        self,
        cache: Snapshot,
        embeds: torch.Tensor,
        lengths: Sequence[int],
        rope: Any,
    ) -> torch.Tensor:
        """Run question blocks on a state snapshot, return [B, D] last-token hidden."""
        rows, width = embeds.shape[:2]
        lengths = [int(n) for n in lengths]
        present = torch.arange(width, device=embeds.device)[None, :] < torch.as_tensor(
            lengths, device=embeds.device
        )[:, None]
        hidden = embeds * present[..., None].to(embeds.dtype)
        for index, ((layer, kind), entry) in enumerate(
            zip(zip(self.layers, self.layer_types), cache)
        ):
            hidden, _ = self._layer_call(
                index, self._layer, index, layer, kind, hidden, rope, entry, present, lengths, "questions"
            )
        last = torch.as_tensor([n - 1 for n in lengths], device=hidden.device)
        rows_idx = torch.arange(rows, device=hidden.device)
        self.stats.question_layers = len(cache)
        return self.final_norm(hidden[rows_idx, last])

    # ------------------------------------------------------------------
    def _layer(
        self,
        index: int,
        layer: Any,
        kind: str,
        hidden: torch.Tensor,
        rope: Any,
        entry: Any,
        present: Any,
        lengths: Any,
        phase: str,
    ):
        profiling = self.profile_layers
        started_event = None
        started_cpu = 0.0
        if profiling:
            if hidden.is_cuda:
                self._layer_profile_device = hidden.device
                started_event = torch.cuda.Event(enable_timing=True)
                started_event.record(torch.cuda.current_stream(hidden.device))
            else:
                started_cpu = time.perf_counter()
        residual = hidden
        normed = layer.input_layernorm(hidden)
        if kind == "full_attention":
            mixed, entry = self._attention(layer.self_attn, normed, rope, entry, lengths)
        elif kind == "linear_attention":
            mixed, entry = self._delta(layer.linear_attn, normed, entry, present)
        else:  # pragma: no cover - config validation catches this
            raise ValueError(f"unknown layer type {kind!r}")
        hidden = residual + mixed
        result = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if profiling:
            if started_event is not None:
                ended_event = torch.cuda.Event(enable_timing=True)
                ended_event.record(torch.cuda.current_stream(hidden.device))
                self._layer_spans.append((phase, index, kind, started_event, ended_event, 0.0))
            else:
                self._layer_spans.append(
                    (phase, index, kind, None, None, (time.perf_counter() - started_cpu) * 1000)
                )
        return result, entry

    # ------------------------------------------------------------------
    def _attention(self, attn: Any, hidden: torch.Tensor, rope: Any, entry: Any, lengths: Any):
        shape = hidden.shape[:-1]
        heads = (*shape, -1, attn.head_dim)
        query, gate = torch.chunk(attn.q_proj(hidden).view(*shape, -1, attn.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        query = attn.q_norm(query.view(heads)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(hidden).view(heads)).transpose(1, 2)
        value = attn.v_proj(hidden).view(heads).transpose(1, 2)
        cos, sin = rope
        query, key = self.apply_rotary(query, key, cos, sin)

        if entry is None:
            flash_ok = (
                query.is_cuda
                and query.dtype in (torch.float16, torch.bfloat16)
                and ops.flash_attn_available()
            )
            if self.attention_state in ("flash", "auto") and flash_ok:
                out = ops.flash_state(query, key, value, scale=attn.scaling)
                self.stats.attention_state = "flash"
            else:
                out = ops.sdpa_state(query, key, value, scale=attn.scaling)
                self.stats.attention_state = "sdpa"
            entry = AttentionEntry(keys=key, values=value)
        else:
            if self.attention_question in ("triton", "auto") and query.is_cuda:
                out, kernel = ops.question_attention(
                    query,
                    entry.keys[0],
                    entry.values[0],
                    key,
                    value,
                    lengths,
                    scale=attn.scaling,
                    prefer_triton=True,
                    return_kernel=True,
                )
                self.stats.attention_question = kernel
            elif (
                self.attention_question in ("flash", "auto")
                and query.is_cuda
                and query.dtype in (torch.float16, torch.bfloat16)
            ):
                out = ops.flash_questions(
                    query,
                    entry.keys[0],
                    entry.values[0],
                    key,
                    value,
                    lengths,
                    scale=attn.scaling,
                )
                self.stats.attention_question = (
                    "flash"
                    if ops.flash_attn_available() and query.dtype in (torch.float16, torch.bfloat16)
                    else "sdpa"
                )
            else:
                out = ops.sdpa_questions(
                    query,
                    entry.keys[0],
                    entry.values[0],
                    key,
                    value,
                    lengths,
                    scale=attn.scaling,
                )
                self.stats.attention_question = "sdpa"
        out = out.transpose(1, 2).reshape(*shape, -1)
        return attn.o_proj(out * torch.sigmoid(gate)), entry

    # ------------------------------------------------------------------
    def _delta(self, delta: Any, hidden: torch.Tensor, entry: Any, present: Any):
        rows, length, _ = hidden.shape
        mixed = delta.in_proj_qkv(hidden).transpose(1, 2)
        z = delta.in_proj_z(hidden).reshape(rows, length, -1, delta.head_v_dim)
        b = delta.in_proj_b(hidden)
        a = delta.in_proj_a(hidden)
        window = delta.conv_kernel_size - 1

        conv_kernel = (
            "fla-conv"
            if hidden.is_cuda and ops.fla_available("conv") and mixed.dtype in (torch.float16, torch.bfloat16)
            else "torch-conv"
        )
        if entry is None:
            conv = self.delta.causal_conv1d(
                mixed, delta.conv1d.weight.squeeze(1), delta.conv1d.bias, activation=delta.activation
            )
            chunk_kernel = (
                "fla-chunk"
                if hidden.is_cuda and ops.fla_available("chunk")
                else "torch-chunk"
            )
            self.stats.delta = f"{chunk_kernel}+{conv_kernel}"
            tail = mixed[:, :, -window:]
            if tail.shape[-1] < window:
                tail = F.pad(tail, (window - tail.shape[-1], 0))
            initial = None
        else:
            tail, initial = entry.tail, entry.recurrent
            joined = torch.cat([tail.expand(rows, -1, -1), mixed], dim=-1)
            conv = self.delta.causal_conv1d(
                joined, delta.conv1d.weight.squeeze(1), delta.conv1d.bias, activation=delta.activation
            )[:, :, window:]
            initial = initial.expand(rows, -1, -1, -1).contiguous()

        query, key, value = torch.split(
            conv.transpose(1, 2), [delta.key_dim, delta.key_dim, delta.value_dim], dim=-1
        )
        query = query.reshape(rows, length, -1, delta.head_k_dim)
        key = key.reshape(rows, length, -1, delta.head_k_dim)
        value = value.reshape(rows, length, -1, delta.head_v_dim)
        beta = b.sigmoid()
        g = -delta.A_log.float().exp() * F.softplus(a.float() + delta.dt_bias)
        if delta.num_v_heads // delta.num_k_heads > 1:
            query = query.repeat_interleave(delta.num_v_heads // delta.num_k_heads, dim=2)
            key = key.repeat_interleave(delta.num_v_heads // delta.num_k_heads, dim=2)

        if entry is None:
            core, final = self.delta.chunk(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            entry = DeltaEntry(tail=tail, recurrent=final)
        else:
            # Short question blocks are faster with the recurrent kernel; long
            # ones use the chunked kernel. The HF parity path disables this
            # (recurrent_max_tokens=0) to match the reference chunk kernel.
            scan = (
                self.delta.recurrent
                if self.recurrent_max_tokens and length <= self.recurrent_max_tokens
                else self.delta.chunk
            )
            core, _ = scan(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            uses_recurrent = bool(self.recurrent_max_tokens and length <= self.recurrent_max_tokens)
            if uses_recurrent:
                scan_kernel = (
                    "fla-recurrent"
                    if hidden.is_cuda and ops.fla_available("recurrent")
                    else "torch-recurrent"
                )
            else:
                scan_kernel = (
                    "fla-chunk" if hidden.is_cuda and ops.fla_available("chunk") else "torch-chunk"
                )
            self.stats.delta = f"{scan_kernel}+{conv_kernel}"
        out = delta.norm(core.reshape(-1, delta.head_v_dim), z.reshape(-1, delta.head_v_dim))
        return delta.out_proj(out.reshape(rows, length, -1)), entry


class DefaultDeltaOps:
    """The functional kernels used by :class:`D3Core`."""

    def causal_conv1d(self, x, weight, bias, *, activation):
        return ops.causal_conv1d(x, weight, bias, activation=activation)

    def chunk(self, q, k, v, **kwargs):
        return ops.chunk_gated_delta_rule(q, k, v, **kwargs)

    def recurrent(self, q, k, v, **kwargs):
        return ops.recurrent_gated_delta_rule(q, k, v, **kwargs)
