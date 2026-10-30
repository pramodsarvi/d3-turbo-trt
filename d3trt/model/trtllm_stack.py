"""TensorRT-LLM module stack for the d3 backbone.

Why a custom runner instead of `trtllm-serve` / `LLM.generate`:

* d3 does one prefill per request and never decodes or samples. A generation
  runtime would spend its complexity (sampling, stop criteria, speculative
  decoding) and its KV-cache policy on a workload that has neither.
* The same state prefix is reused by up to 255 question blocks. The runner needs
  an explicit "state pass once, then batched question passes" schedule and a
  content-addressed state cache. That is not expressible through the generation
  API without a core change.
* The output is a 255-way readout at selected positions, not a vocabulary
  distribution.

The stack still uses TensorRT-LLM's optimized building blocks: `Linear` (CUTLASS
/ cuBLAS-Lt kernels, TP sharding), `Embedding` and `RMSNorm`. With ``quant`` set to
``fp8`` or ``int8`` the projections are :class:`~d3trt.model.quant.QuantLinear`
instead, which is what lets the model fit 40-48 GB cards. Full attention runs
on FlashAttention (or the shared-prefix Triton kernel for the question pass),
and the GDN scan and convolution run on FLA / causal-conv1d, with pure-PyTorch
fallbacks. This is the performance path; the reference Transformers path remains
the oracle.

The class is intentionally import-safe on machines without TensorRT-LLM: all
NVIDIA imports happen inside `_build`, and `trtllm_available()` is the only
entry point the registry probes.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..errors import BackendUnavailable, ConfigurationError
from . import native
from . import quant as quantization
from .core import D3Core
from .quant import QuantLinear, make_linear
from .weights import ShardedWeights


def trtllm_available() -> tuple[bool, str]:
    """Return ``(ok, reason)`` without importing CUDA-only modules eagerly."""
    try:
        import tensorrt_llm  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"tensorrt_llm is not importable ({type(exc).__name__}: {exc})"
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA is not available"
    except Exception as exc:  # noqa: BLE001
        return False, f"torch/CUDA check failed ({type(exc).__name__}: {exc})"
    return True, "ok"


def _tp_mode(mapping: Any, name: str) -> Any:
    """TensorRT-LLM TP mode, or ``None`` on the native path (``mapping is None``)."""
    if mapping is None:
        return None
    from tensorrt_llm._torch.modules.linear import TensorParallelMode

    return getattr(TensorParallelMode, name)


def _rms_norm(dim: int, eps: float, dtype: torch.dtype, mapping: Any) -> nn.Module:
    if mapping is None:
        return native.RMSNorm(dim, eps, dtype=dtype)
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    return RMSNorm(hidden_size=dim, eps=eps, dtype=dtype, use_gemma=True)


class _TrtllmAttention(nn.Module):
    def __init__(
        self, config: Any, layer_idx: int, *, dtype: torch.dtype, mapping: Any, quant: str, device: Any
    ):
        super().__init__()
        linear = functools.partial(make_linear, quant=quant, dtype=dtype, mapping=mapping, device=device)

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.q_proj = linear(
            self.hidden_size,
            self.num_heads * self.head_dim * 2,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.k_proj = linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.v_proj = linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.o_proj = linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            tensor_parallel_mode=_tp_mode(mapping, "ROW"),
        )
        self.q_norm = _rms_norm(self.head_dim, config.rms_norm_eps, dtype, mapping)
        self.k_norm = _rms_norm(self.head_dim, config.rms_norm_eps, dtype, mapping)


class _TrtllmMLP(nn.Module):
    def __init__(self, config: Any, *, dtype: torch.dtype, mapping: Any, quant: str, device: Any):
        super().__init__()
        linear = functools.partial(make_linear, quant=quant, dtype=dtype, mapping=mapping, device=device)

        self.gate_up_proj = linear(
            config.hidden_size,
            2 * config.intermediate_size,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.down_proj = linear(
            config.intermediate_size,
            config.hidden_size,
            tensor_parallel_mode=_tp_mode(mapping, "ROW"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class _TrtllmGDN(nn.Module):
    def __init__(
        self, config: Any, layer_idx: int, *, dtype: torch.dtype, mapping: Any, quant: str, device: Any
    ):
        super().__init__()
        linear = functools.partial(make_linear, quant=quant, dtype=dtype, mapping=mapping, device=device)

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.activation = config.hidden_act

        self.in_proj_qkv = linear(
            self.hidden_size,
            self.conv_dim,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.in_proj_z = linear(
            self.hidden_size,
            self.value_dim,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.in_proj_b = linear(
            self.hidden_size,
            self.num_v_heads,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.in_proj_a = linear(
            self.hidden_size,
            self.num_v_heads,
            tensor_parallel_mode=_tp_mode(mapping, "COLUMN"),
        )
        self.out_proj = linear(
            self.value_dim,
            self.hidden_size,
            tensor_parallel_mode=_tp_mode(mapping, "ROW"),
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            dtype=dtype,
        )
        # The scan always consumes g as fp32, so keep these in fp32 even when the
        # linear layers run in bf16. The norm weight must match the activation
        # dtype because its output feeds the bf16 out_proj.
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = native.RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps, dtype=dtype)


class _TrtllmLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_idx: int,
        kind: str,
        *,
        dtype: torch.dtype,
        mapping: Any,
        quant: str,
        device: Any,
    ):
        super().__init__()
        self.kind = kind
        self.input_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps, dtype, mapping)
        self.post_attention_layernorm = _rms_norm(config.hidden_size, config.rms_norm_eps, dtype, mapping)
        parts = dict(dtype=dtype, mapping=mapping, quant=quant, device=device)
        self.mlp = _TrtllmMLP(config, **parts)
        if kind == "full_attention":
            self.self_attn = _TrtllmAttention(config, layer_idx, **parts)
        else:
            self.linear_attn = _TrtllmGDN(config, layer_idx, **parts)


class TrtllmStack:
    """A d3 backbone built from TensorRT-LLM modules (``impl="trtllm"``) or from
    the dependency-free modules in :mod:`d3trt.model.native` (``impl="native"``).
    Both share the weight loader, the state-once core and the kernels."""

    impl = "trtllm"

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        tp_size: int = 1,
        rank: int = 0,
        recurrent_max_tokens: int = 256,
        quant: str = "none",
    ):
        ok, reason = self.available()
        if not ok:
            raise BackendUnavailable(self.impl, reason)
        if tp_size != 1:
            raise BackendUnavailable(
                self.impl,
                "this build starts every rank independently and supports tp_size=1; "
                "use one 80 GB+ GPU, or quant='fp8' (sm_89+) / 'int8' (A100) to fit 40-48 GB. TP support "
                "is isolated in d3trt/model/trtllm_stack.py and is the next GPU-day item.",
            )
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.tp_size = tp_size
        self.rank = rank
        self.recurrent_max_tokens = recurrent_max_tokens
        self.weights = ShardedWeights(model_dir)
        self.quant = quantization.resolve_mode(quant, self.device, self.weights.total_bytes())
        self._build()

    # ------------------------------------------------------------------
    @staticmethod
    def available() -> tuple[bool, str]:
        return trtllm_available()

    def _build(self) -> None:
        torch.cuda.set_device(self.device)
        text = native.load_text_config(self.model_dir)
        self.text_config = text
        if self.impl == "native":
            self.mapping = None
            self.embed = nn.Embedding(text.vocab_size, text.hidden_size, dtype=self.dtype, device=self.device)
        else:
            from tensorrt_llm._torch.modules.embedding import Embedding
            from tensorrt_llm.mapping import Mapping

            self.mapping = Mapping(
                world_size=self.tp_size,
                tp_size=self.tp_size,
                rank=self.rank,
                gpus_per_node=self.tp_size,
            )
            self.embed = Embedding(
                text.vocab_size,
                text.hidden_size,
                dtype=self.dtype,
                mapping=self.mapping,
                tensor_parallel_mode=_tp_mode(self.mapping, "COLUMN") if self.tp_size > 1 else None,
                gather_output=self.tp_size > 1,
            ).to(self.device)

        self.layers = nn.ModuleList()
        for index, kind in enumerate(text.layer_types):
            layer = _TrtllmLayer(
                text,
                index,
                kind,
                dtype=self.dtype,
                mapping=self.mapping,
                quant=self.quant,
                device=self.device,
            )
            self.layers.append(layer)
        self.layers.to(self.device)
        self.norm = _rms_norm(text.hidden_size, text.rms_norm_eps, self.dtype, self.mapping).to(self.device)
        self.rotary = native.RotaryEmbedding(text, device=self.device).to(self.device)
        self.apply_rotary = native.apply_rotary_pos_emb
        self._load_weights()
        self.layers.eval().requires_grad_(False)
        self.norm.eval().requires_grad_(False)
        self.embed.eval().requires_grad_(False)
        self.rotary.eval().requires_grad_(False)

        self.core = D3Core(
            layers=list(self.layers),
            layer_types=list(text.layer_types),
            final_norm=self.norm,
            rotary_emb=self.rotary,
            apply_rotary=self.apply_rotary,
            attention_state="flash",
            attention_question="triton",
            recurrent_max_tokens=self.recurrent_max_tokens,
        )
        self.readout = self.weights.readout(device=self.device).float()
        self.max_options = int(self.readout.shape[0])
        self.pad_id = None

    # ------------------------------------------------------------------
    def _load_weights(self) -> None:
        prefix = "language_model"
        head = self.weights.get_many(
            [f"{prefix}.embed_tokens.weight", f"{prefix}.norm.weight"], device="cpu"
        )
        _copy_linear(self.embed, head[f"{prefix}.embed_tokens.weight"], self.device)
        _copy_param(self.norm.weight, head[f"{prefix}.norm.weight"], self.device)

        for index, layer in enumerate(self.layers):
            # Load one layer at a time: the full checkpoint is ~52 GB and must
            # not be mirrored in host RAM while the GPU copy is built.
            tensors = self.weights.get_many(self._layer_names(index), device="cpu")
            base = f"{prefix}.layers.{index}"
            _copy_param(layer.input_layernorm.weight, tensors[f"{base}.input_layernorm.weight"], self.device)
            _copy_param(
                layer.post_attention_layernorm.weight,
                tensors[f"{base}.post_attention_layernorm.weight"],
                self.device,
            )
            _copy_linear(layer.mlp.down_proj, tensors[f"{base}.mlp.down_proj.weight"], self.device)
            gate = tensors[f"{base}.mlp.gate_proj.weight"]
            up = tensors[f"{base}.mlp.up_proj.weight"]
            _copy_linear(layer.mlp.gate_up_proj, torch.cat([gate, up], dim=0), self.device)

            if layer.kind == "full_attention":
                attn = layer.self_attn
                _copy_linear(attn.q_proj, tensors[f"{base}.self_attn.q_proj.weight"], self.device)
                _copy_linear(attn.k_proj, tensors[f"{base}.self_attn.k_proj.weight"], self.device)
                _copy_linear(attn.v_proj, tensors[f"{base}.self_attn.v_proj.weight"], self.device)
                _copy_linear(attn.o_proj, tensors[f"{base}.self_attn.o_proj.weight"], self.device)
                _copy_param(attn.q_norm.weight, tensors[f"{base}.self_attn.q_norm.weight"], self.device)
                _copy_param(attn.k_norm.weight, tensors[f"{base}.self_attn.k_norm.weight"], self.device)
            else:
                gdn = layer.linear_attn
                _copy_linear(gdn.in_proj_qkv, tensors[f"{base}.linear_attn.in_proj_qkv.weight"], self.device)
                _copy_linear(gdn.in_proj_z, tensors[f"{base}.linear_attn.in_proj_z.weight"], self.device)
                _copy_linear(gdn.in_proj_b, tensors[f"{base}.linear_attn.in_proj_b.weight"], self.device)
                _copy_linear(gdn.in_proj_a, tensors[f"{base}.linear_attn.in_proj_a.weight"], self.device)
                _copy_linear(gdn.out_proj, tensors[f"{base}.linear_attn.out_proj.weight"], self.device)
                _copy_param(
                    gdn.conv1d.weight,
                    tensors[f"{base}.linear_attn.conv1d.weight"],
                    self.device,
                )
                _copy_param(gdn.A_log, tensors[f"{base}.linear_attn.A_log"], self.device)
                _copy_param(gdn.dt_bias, tensors[f"{base}.linear_attn.dt_bias"], self.device)
                _copy_param(gdn.norm.weight, tensors[f"{base}.linear_attn.norm.weight"], self.device)

    def _layer_names(self, index: int) -> list[str]:
        base = f"language_model.layers.{index}"
        names = [
            f"{base}.input_layernorm.weight",
            f"{base}.post_attention_layernorm.weight",
            f"{base}.mlp.down_proj.weight",
            f"{base}.mlp.gate_proj.weight",
            f"{base}.mlp.up_proj.weight",
        ]
        if self.text_config.layer_types[index] == "full_attention":
            names.extend(
                [
                    f"{base}.self_attn.q_proj.weight",
                    f"{base}.self_attn.k_proj.weight",
                    f"{base}.self_attn.v_proj.weight",
                    f"{base}.self_attn.o_proj.weight",
                    f"{base}.self_attn.q_norm.weight",
                    f"{base}.self_attn.k_norm.weight",
                ]
            )
        else:
            names.extend(
                [
                    f"{base}.linear_attn.in_proj_qkv.weight",
                    f"{base}.linear_attn.in_proj_z.weight",
                    f"{base}.linear_attn.in_proj_b.weight",
                    f"{base}.linear_attn.in_proj_a.weight",
                    f"{base}.linear_attn.out_proj.weight",
                    f"{base}.linear_attn.conv1d.weight",
                    f"{base}.linear_attn.A_log",
                    f"{base}.linear_attn.dt_bias",
                    f"{base}.linear_attn.norm.weight",
                ]
            )
        return names

    # ------------------------------------------------------------------
    def state(self, ids: torch.Tensor) -> Any:
        embeds = self.embed(ids.to(self.device))
        positions = torch.arange(ids.shape[1], device=self.device).view(1, 1, -1).expand(3, 1, -1)
        rope = self.rotary(embeds, positions)
        return self.core.state(embeds, rope)

    def questions(
        self,
        snapshot: Any,
        ids: torch.Tensor,
        lengths: Sequence[int],
        *,
        state_len: int,
    ) -> torch.Tensor:
        width = ids.shape[1]
        embeds = self.embed(ids.to(self.device))
        steps = (state_len + torch.arange(width, device=self.device)).view(1, 1, -1)
        rope = self.rotary(embeds, steps.expand(3, ids.shape[0], -1))
        return self.core.questions(snapshot, embeds, lengths, rope)

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
            logits = F.linear(hidden, self.readout).float()
        counts_t = torch.as_tensor(list(counts), device=hidden.device)[:, None]
        logits = logits.masked_fill(
            torch.arange(self.max_options, device=hidden.device)[None] >= counts_t, float("-inf")
        )
        values = (logits / temperature).softmax(-1).cpu().tolist()
        return [row[:count] for row, count in zip(values, counts)]

    def info(self) -> dict[str, Any]:
        from . import ops

        return {
            "backend": self.impl,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "layers": len(self.layers),
            "tp_size": self.tp_size,
            "quant": self.quant,
            "linear": quantization.summarize(self.layers.modules()),
            "weights": dict(self.weights.describe()),
            "kernels": ops.kernel_report(),
            "core": asdict(self.core.stats),
        }


def _copy_param(param: nn.Parameter, full: torch.Tensor, device: torch.device) -> None:
    if tuple(param.shape) != tuple(full.shape):
        raise ConfigurationError(
            f"weight shape mismatch: module {tuple(param.shape)} vs checkpoint {tuple(full.shape)}"
        )
    param.data.copy_(full.to(device=device, dtype=param.dtype))


def _copy_linear(linear: Any, full: torch.Tensor, device: torch.device) -> None:
    """Copy a full HF weight into a TRT-LLM Linear (TP=1) or quantize it into a QuantLinear."""
    if isinstance(linear, QuantLinear):
        linear.load_full(full.to(device))
        return
    if getattr(linear, "tp_size", 1) != 1:
        # Let TRT-LLM shard the full tensor across ranks.
        linear.load_weights([{"weight": full.to(device)}])
        return
    if tuple(linear.weight.shape) != tuple(full.shape):
        raise ConfigurationError(
            f"linear shape mismatch: module {tuple(linear.weight.shape)} vs checkpoint {tuple(full.shape)}"
        )
    linear.weight.data.copy_(full.to(device=device, dtype=linear.weight.dtype))


class NativeStack(TrtllmStack):
    """The d3 backbone with no TensorRT-LLM or Transformers dependency."""

    impl = "native"

    @staticmethod
    def available() -> tuple[bool, str]:
        return native_available()


def native_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    return True, "ok"
