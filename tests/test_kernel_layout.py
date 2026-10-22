"""Checks for the layout contract used by the shared-prefix Triton kernel.

The kernel addresses K/V with a head stride of D and a token stride of H*D. A
plain contiguous [H, S, D] tensor has the opposite geometry, so the wrapper must
normalise it before launch. No Triton/CUDA is needed to test that transformation.
"""

from __future__ import annotations

import torch

from d3trt.kernels.prefix_bidir import _token_major


def test_token_major_normalises_contiguous_heads_first():
    heads, tokens, dim = 4, 7, 32
    source = torch.randn(heads, tokens, dim)
    assert source.stride() == (tokens * dim, dim, 1)

    view = _token_major(source)
    assert view.shape == source.shape
    assert view.stride() == (dim, heads * dim, 1)
    assert torch.equal(view, source)


def test_token_major_is_a_noop_for_the_core_layout():
    heads, tokens, dim = 4, 7, 32
    # [S, H, D] contiguous viewed as [H, S, D] is the layout produced by
    # transposing the attention projection output.
    view = torch.randn(tokens, heads, dim).transpose(0, 1)
    assert view.stride() == (dim, heads * dim, 1)
    assert _token_major(view) is view
