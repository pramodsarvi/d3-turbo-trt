"""Shared-prefix bidirectional attention (Triton).

The question pass has a shape that stock FlashAttention cannot express without
copying the state's K/V once per question: every question block attends to the
same state prefix plus its own block, and to nothing else. This kernel reads the
shared state K/V directly for every question, so the 16 full-attention layers
store the state exactly once.

Layouts (all contiguous):
    q       [total_q, H, D]        packed real question tokens
    k_state [S, H_kv, D]           shared prefix, RoPE already applied
    v_state [S, H_kv, D]
    k_q     [total_q, H_kv, D]     packed question keys
    v_q     [total_q, H_kv, D]
    q_lens  [B] int32              real tokens per question
    out     [total_q, H, D]

The kernel is intentionally a straightforward FlashAttention forward pass with a
non-causal mask; it has no causal early-exit, so it is also correct for the
state pass (B=1, state as k_q). If Triton or a supported dtype is unavailable the
caller falls back to FlashAttention/SDPA.
"""

from __future__ import annotations

from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl

    _TRITON = True
except Exception:  # noqa: BLE001
    _TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore


def triton_available() -> bool:
    return _TRITON


if _TRITON:

    @triton.jit
    def _prefix_bidir_fwd(
        Q,
        K_STATE,
        V_STATE,
        K_Q,
        V_Q,
        Q_STARTS,
        Q_LENS,
        OUT,
        stride_qh,
        stride_kvh,
        stride_oh,
        S,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)

        q_len = tl.load(Q_LENS + pid_b)
        start_m = pid_m * BLOCK_M
        if start_m >= q_len:
            return

        q_base = tl.load(Q_STARTS + pid_b)
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)

        q_ptrs = Q + (q_base + offs_m)[:, None] * stride_qh + pid_h * D + offs_d[None, :]
        q = tl.load(q_ptrs, mask=offs_m[:, None] < q_len, other=0.0)

        kv_head = pid_h // (H // H_KV)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

        # Shared state prefix: every real state token is visible.
        for start_n in tl.range(0, S, BLOCK_N):
            ns = start_n + offs_n
            k_ptrs = K_STATE + ns[:, None] * stride_kvh + kv_head * D + offs_d[None, :]
            v_ptrs = V_STATE + ns[:, None] * stride_kvh + kv_head * D + offs_d[None, :]
            k = tl.load(k_ptrs, mask=ns[:, None] < S, other=0.0)
            v = tl.load(v_ptrs, mask=ns[:, None] < S, other=0.0)
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(ns[None, :] < S, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # The question's own block: causal=False, padding excluded.
        # K_Q/V_Q are packed [total_q, H_kv, D] buffers, so their row stride is
        # H_kv * D. It is not the state KV stride (S * D); using one stride for
        # both was an aliasing bug whenever S != H_kv.
        packed_stride = H_KV * D
        kb = K_Q + q_base * packed_stride + kv_head * D
        vb = V_Q + q_base * packed_stride + kv_head * D
        for start_n in tl.range(0, q_len, BLOCK_N):
            nn = start_n + offs_n
            k_ptrs = kb + nn[:, None] * packed_stride + offs_d[None, :]
            v_ptrs = vb + nn[:, None] * packed_stride + offs_d[None, :]
            k = tl.load(k_ptrs, mask=nn[:, None] < q_len, other=0.0)
            v = tl.load(v_ptrs, mask=nn[:, None] < q_len, other=0.0)
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(nn[None, :] < q_len, qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = OUT + (q_base + offs_m)[:, None] * stride_oh + pid_h * D + offs_d[None, :]
        tl.store(o_ptrs, acc.to(OUT.dtype.element_ty), mask=offs_m[:, None] < q_len)


class _Meta:
    """Index tensors for one (lengths, width, device); built once, reused by all layers."""

    __slots__ = ("rows", "pos", "q_starts", "q_lens", "total", "max_len")

    def __init__(self, lengths: Sequence[int], device: torch.device):
        counts = torch.tensor(lengths, dtype=torch.int64)
        self.total = int(counts.sum())
        self.max_len = max(lengths)
        rows = torch.repeat_interleave(torch.arange(len(lengths)), counts)
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(self.total) - torch.repeat_interleave(starts, counts)
        self.rows = rows.to(device)
        self.pos = pos.to(device)
        self.q_starts = starts.to(device=device, dtype=torch.int32)
        self.q_lens = counts.to(device=device, dtype=torch.int32)


_META: dict[tuple, _Meta] = {}
_META_MAX = 16


def _meta(lengths: Sequence[int], device: torch.device) -> _Meta:
    key = (tuple(lengths), str(device))
    meta = _META.get(key)
    if meta is None:
        if len(_META) >= _META_MAX:
            _META.pop(next(iter(_META)))
        meta = _META[key] = _Meta(lengths, device)
    return meta


def _pack_rows(x: torch.Tensor, meta: _Meta) -> torch.Tensor:
    """[B, H, L, D] right-padded -> [sum(lengths), H, D] (one gather, no host loop)."""
    return x[meta.rows, :, meta.pos].contiguous()


def _unpack_rows(x: torch.Tensor, meta: _Meta, rows: int, width: int) -> torch.Tensor:
    """[sum(lengths), H, D] -> [B, H, width, D] right-padded with zeros."""
    _, heads, dim = x.shape
    out = x.new_zeros((rows, heads, width, dim))
    out[meta.rows, :, meta.pos] = x
    return out


def _token_major(x: torch.Tensor) -> torch.Tensor:
    """Return an [H, S, D] view whose head stride is D and token stride is H*D."""
    heads, tokens, dim = x.shape
    if x.stride() == (dim, heads * dim, 1):
        return x
    return x.transpose(0, 1).contiguous().transpose(0, 1)


def prefix_bidir_attention(
    q: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    k_q: torch.Tensor,
    v_q: torch.Tensor,
    lengths: Sequence[int],
    *,
    scale: float,
) -> torch.Tensor:
    """Bidirectional question attention over a shared state prefix.

    Inputs follow the model's natural layout: q/k_q/v_q are [B, H, L, D],
    k_state/v_state are [H_kv, S, D]. Returns [B, H, L, D].
    """
    if not _TRITON:
        raise RuntimeError("Triton is not available")
    if not q.is_cuda:
        raise RuntimeError("the prefix-bidir kernel needs CUDA")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("the prefix-bidir kernel needs fp16 or bf16")
    if v_q.dtype != q.dtype or k_q.dtype != q.dtype:
        raise RuntimeError("q, k and v must share a dtype")
    if q.shape[-1] not in (32, 64, 128, 256):
        raise RuntimeError(f"unsupported head_dim {q.shape[-1]}")

    rows, heads, width, dim = q.shape
    kv_heads = k_state.shape[0]
    if heads % kv_heads:
        raise RuntimeError("heads must be divisible by kv heads")
    lengths = [int(n) for n in lengths]
    total = sum(lengths)
    if total == 0:
        return q.new_zeros(q.shape)

    # The kernel addresses K/V as a token-major [S, H_kv, D] buffer viewed as
    # [H_kv, S, D]: head stride D, token stride H_kv*D. A plain contiguous
    # [H_kv, S, D] tensor has the opposite geometry, so normalise it once here.
    k_state = _token_major(k_state)
    v_state = _token_major(v_state)

    meta = _meta(lengths, q.device)
    q_packed = _pack_rows(q, meta)
    kq_packed = _pack_rows(k_q, meta)
    vq_packed = _pack_rows(v_q, meta)
    out = torch.empty_like(q_packed)
    q_starts, q_lens = meta.q_starts, meta.q_lens

    block_m = 16 if dim > 128 else 32
    block_n = 32 if dim > 128 else 64
    grid = (triton.cdiv(meta.max_len, block_m), rows, heads)
    _prefix_bidir_fwd[grid](
        q_packed,
        k_state,
        v_state,
        kq_packed,
        vq_packed,
        q_starts,
        q_lens,
        out,
        q_packed.stride(0),
        k_state.stride(1),
        out.stride(0),
        k_state.shape[1],
        H=heads,
        H_KV=kv_heads,
        D=dim,
        SCALE=scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return _unpack_rows(out, meta, rows, width)
