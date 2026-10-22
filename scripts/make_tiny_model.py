#!/usr/bin/env python3
"""Build a tiny random d3 checkpoint for CPU tests.

The tiny model keeps the real tokenizer, the real `decision_config.json` and the
real Qwen3.5 architecture contract; only the dimensions and weights are small.
Every tokenizer id, prompt byte and answer code is therefore identical to the
production package, which is what the parity tests need.

    python scripts/make_tiny_model.py --out .research/tiny/model
    D3_TEST_MODEL=.research/tiny/model pytest

If `--tokenizer-from` points at a directory with the d3 package files
(`d3_runtime.py`, `decision_config.json`, tokenizer files), those are copied so
the generated checkpoint is self-contained. The public reference package is a
good source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5Model, Qwen3_5TextConfig, Qwen3_5VisionConfig

ROOT = Path(__file__).resolve().parents[1]


def build(out: Path, tokenizer_from: Path, *, hidden: int, layers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    head_dim = 32
    text = Qwen3_5TextConfig(
        vocab_size=248320,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        output_gate_type="swish",
        layer_types=["linear_attention" if i % 4 != 3 else "full_attention" for i in range(layers)],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        mamba_ssm_dtype="float32",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
        partial_rotary_factor=0.25,
        tie_word_embeddings=False,
        bos_token_id=248044,
        eos_token_id=248044,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=hidden,
        num_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
    )
    config = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
    )
    torch.manual_seed(0)
    model = Qwen3_5Model(config)
    model.save_pretrained(out)
    print(f"saved {sum(p.numel() for p in model.parameters()):,} parameters to {out}")

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "decision_config.json",
        "d3_runtime.py",
        "d3_format.py",
        "d3_compose.py",
        "d3_server.py",
        "modeling_d3.py",
    ):
        source = tokenizer_from / name
        if source.exists():
            shutil.copy(source, out / name)

    if not (out / "decision_config.json").exists():
        raise FileNotFoundError(
            f"{tokenizer_from}/decision_config.json is required; point --tokenizer-from at the d3 package"
        )

    decision = json.loads((out / "decision_config.json").read_text())
    if not decision.get("codes") or not decision.get("token_ids"):
        raise ValueError("decision_config.json has no answer codes")
    readout = torch.randn(
        len(decision["codes"]), hidden, generator=torch.Generator().manual_seed(1), dtype=torch.float32
    )
    save_file({"weight": readout}, str(out / "readout.safetensors"))
    print(f"readout: {tuple(readout.shape)} fp32")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / ".research" / "tiny" / "model"))
    parser.add_argument(
        "--tokenizer-from",
        default=str(ROOT / "reference"),
        help="directory holding the d3 tokenizer + decision_config",
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args(argv)
    build(Path(args.out), Path(args.tokenizer_from), hidden=args.hidden, layers=args.layers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
