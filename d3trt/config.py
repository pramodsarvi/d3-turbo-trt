"""Checkpoint and runtime configuration.

The d3 package ships three sources of truth:

``config.json``
    the stock HuggingFace Qwen3.5 configuration (backbone, vision tower, mrope).
``decision_config.json``
    the d3 contract: answer codes, readout offsets, prompt, attention mode,
    temperature, calibration and System One feature gates.
``MODEL_MANIFEST.json``
    package identity, file digests, modalities and limits.

This module parses all three once and exposes the handful of architectural
facts the fast path needs. It never imports torch, so it is safe to use from a
configuration check or the CLI on a machine without CUDA.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError

DEFAULT_STATE_CACHE_BYTES = 24 * 1024**3


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"missing {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must hold a JSON object")
    return data


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Qwen3.5 keeps the language model under ``text_config``; Qwen3-VL too."""
    text = config.get("text_config")
    if isinstance(text, Mapping):
        return text
    return config


@dataclass(frozen=True)
class LayerSpec:
    index: int
    kind: str

    @property
    def is_full_attention(self) -> bool:
        return self.kind == "full_attention"

    @property
    def is_linear_attention(self) -> bool:
        return self.kind == "linear_attention"


@dataclass(frozen=True)
class VisionSpec:
    present: bool
    depth: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_heads: int = 0
    patch_size: int = 16
    merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 0

    @property
    def patch_dim(self) -> int:
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size


@dataclass(frozen=True)
class D3Config:
    """Everything the runtime needs to know about a d3 checkpoint."""

    root: Path
    hf_config: Mapping[str, Any]
    decision: Mapping[str, Any]
    manifest: Mapping[str, Any]

    # ---- backbone ---------------------------------------------------------
    num_layers: int
    layer_specs: tuple[LayerSpec, ...]
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, ...]
    mrope_interleaved: bool
    max_position_embeddings: int
    dtype: str
    tie_word_embeddings: bool

    # ---- d3 contract ------------------------------------------------------
    attention_mode: str
    prompt: str
    temperature: float
    readout_dtype: str
    codes: tuple[str, ...]
    token_ids: tuple[int, ...]
    validated_length: int | None
    context_limit: int | None
    max_options: int
    image_placements: tuple[str, ...]
    question_types: tuple[str, ...]
    systemone: Mapping[str, Any]

    # ---- modalities -------------------------------------------------------
    vision: VisionSpec
    image_token_id: int | None
    video_token_id: int | None
    vision_start_token_id: int | None
    vision_end_token_id: int | None

    # ---- runtime ----------------------------------------------------------
    state_cache_bytes: int = DEFAULT_STATE_CACHE_BYTES

    @property
    def full_attention_layers(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.layer_specs if s.is_full_attention)

    @property
    def linear_attention_layers(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.layer_specs if s.is_linear_attention)

    @property
    def kv_bytes_per_state_token(self) -> int:
        """Bytes of state KV per token across every full-attention layer (BF16)."""
        # K and V, each [num_kv_heads, head_dim], two bytes per element.
        return len(self.full_attention_layers) * 2 * self.num_kv_heads * self.head_dim * 2

    @property
    def gdn_state_bytes(self) -> int:
        """Bytes of one FP32 GDN snapshot (conv tail + recurrent state)."""
        conv_dim = 2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim
        conv = conv_dim * (self.conv_kernel_size - 1) * 4
        recurrent = self.num_v_heads * self.head_k_dim * self.head_v_dim * 4
        return len(self.linear_attention_layers) * (conv + recurrent)

    def state_cache_slots(self) -> int:
        """How many full state snapshots fit in the configured cache budget."""
        per_state = self.gdn_state_bytes
        if per_state <= 0:
            return 0
        return max(1, self.state_cache_bytes // per_state)

    # ---- helpers ----------------------------------------------------------
    @property
    def d3_root(self) -> Path:
        return self.root

    @property
    def has_vision(self) -> bool:
        return self.vision.present

    @property
    def model_name(self) -> str:
        return str(self.manifest.get("model_name") or self.root.name)

    def state_key(self, token_ids: tuple[int, ...] | list[int], media_hash: str = "") -> str:
        """Stable content hash for the shared state block.

        Two requests with the same state tokens (and, for multimodal states, the
        same decoded media) can reuse the same KV + GDN snapshot. The token ids
        include the expanded image/video placeholders, so the media hash only
        needs to disambiguate prompts whose placeholders are identical but whose
        pixels differ.
        """
        digest = hashlib.sha256()
        digest.update(b"d3-state-v1\0")
        digest.update(media_hash.encode("utf-8"))
        digest.update(b"\0")
        for token in token_ids:
            digest.update(int(token).to_bytes(4, "little", signed=False))
        return digest.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "model_name": self.model_name,
            "backbone": self.hf_config.get("model_type"),
            "layers": self.num_layers,
            "full_attention_layers": len(self.full_attention_layers),
            "linear_attention_layers": len(self.linear_attention_layers),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "heads": f"{self.num_heads}/{self.num_kv_heads} x {self.head_dim}",
            "attention_mode": self.attention_mode,
            "dtype": self.dtype,
            "vision": self.vision.present,
            "codes": len(self.codes),
            "state_cache_bytes": self.state_cache_bytes,
            "kv_bytes_per_state_token": self.kv_bytes_per_state_token,
            "gdn_state_bytes": self.gdn_state_bytes,
        }


def _layer_types(text: Mapping[str, Any]) -> list[str]:
    layer_types = text.get("layer_types")
    depth = int(text.get("num_hidden_layers", 0))
    if isinstance(layer_types, list) and len(layer_types) == depth:
        return [str(kind) for kind in layer_types]
    interval = int(text.get("full_attention_interval", 0) or 0)
    if depth and interval:
        # Qwen3-Next convention: one full-attention layer every `interval`
        # layers, the last one in each group.
        return [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(depth)
        ]
    return ["full_attention"] * depth


def _vision_spec(hf: Mapping[str, Any], text: Mapping[str, Any]) -> VisionSpec:
    vision = hf.get("vision_config")
    if not isinstance(vision, Mapping):
        return VisionSpec(present=False)
    depth = int(vision.get("depth", 0) or 0)
    if depth == 0:
        return VisionSpec(present=False)
    return VisionSpec(
        present=True,
        depth=depth,
        hidden_size=int(vision.get("hidden_size", 0) or 0),
        intermediate_size=int(vision.get("intermediate_size", 0) or 0),
        num_heads=int(vision.get("num_heads", 0) or 0),
        patch_size=int(vision.get("patch_size", 16) or 16),
        merge_size=int(vision.get("spatial_merge_size", 2) or 2),
        temporal_patch_size=int(vision.get("temporal_patch_size", 2) or 2),
        out_hidden_size=int(vision.get("out_hidden_size", text.get("hidden_size", 0)) or 0),
    )


def load_config(
    model_dir: str | Path,
    *,
    state_cache_bytes: int | None = None,
) -> D3Config:
    """Parse a d3 checkpoint directory (the HF repo layout)."""
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"model directory {root} does not exist")

    hf = _read_json(root / "config.json")
    decision = _read_json(root / "decision_config.json")
    manifest_path = root / "MODEL_MANIFEST.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    text = _text_config(hf)

    depth = int(text.get("num_hidden_layers", 0) or 0)
    hidden = int(text.get("hidden_size", 0) or 0)
    if depth <= 0 or hidden <= 0:
        raise ConfigurationError("config.json has no usable text backbone dimensions")

    specs = tuple(LayerSpec(i, kind) for i, kind in enumerate(_layer_types(text)))
    num_heads = int(text.get("num_attention_heads", 0) or 0)
    num_kv_heads = int(text.get("num_key_value_heads", num_heads) or num_heads)
    head_dim = int(text.get("head_dim", hidden // max(num_heads, 1)) or 0)

    attention_mode = str(decision.get("attention_mode", "causal"))
    if attention_mode not in ("causal", "noncausal_full_attention", "prefix_bidir"):
        raise ConfigurationError(f"unsupported attention_mode {attention_mode!r}")
    if attention_mode != "causal" and hf.get("model_type") != "qwen3_5":
        raise ConfigurationError(
            f"{attention_mode} needs a qwen3_5 backbone, got {hf.get('model_type')!r}"
        )

    rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if not isinstance(rope, Mapping):
        rope = {}
    partial = float(
        rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0)) or 1.0
    )
    mrope = rope.get("mrope_section") or text.get("mrope_section") or ()
    if not isinstance(mrope, (list, tuple)):
        mrope = ()

    vision = _vision_spec(hf, text)
    if vision.present and vision.out_hidden_size not in (0, hidden):
        raise ConfigurationError(
            f"vision output {vision.out_hidden_size} does not match text hidden {hidden}"
        )

    codes = decision.get("codes")
    token_ids = decision.get("token_ids")
    if not isinstance(codes, list) or not isinstance(token_ids, list):
        raise ConfigurationError("decision_config.json needs codes and token_ids lists")
    if len(codes) != len(token_ids) or not codes:
        raise ConfigurationError("codes and token_ids must have the same nonzero length")

    max_length = decision.get("max_length")
    validated = decision.get("validated_length")
    budget = int(state_cache_bytes) if state_cache_bytes is not None else DEFAULT_STATE_CACHE_BYTES

    return D3Config(
        root=root,
        hf_config=hf,
        decision=decision,
        manifest=manifest,
        num_layers=depth,
        layer_specs=specs,
        hidden_size=hidden,
        intermediate_size=int(text.get("intermediate_size", 0) or 0),
        vocab_size=int(text.get("vocab_size", 0) or 0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_k_heads=int(text.get("linear_num_key_heads", 0) or 0),
        num_v_heads=int(text.get("linear_num_value_heads", 0) or 0),
        head_k_dim=int(text.get("linear_key_head_dim", 0) or 0),
        head_v_dim=int(text.get("linear_value_head_dim", 0) or 0),
        conv_kernel_size=int(text.get("linear_conv_kernel_dim", 4) or 4),
        rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
        rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 10000.0)) or 10000.0),
        partial_rotary_factor=partial,
        mrope_section=tuple(int(x) for x in mrope),
        mrope_interleaved=bool(rope.get("mrope_interleaved", False)),
        max_position_embeddings=int(text.get("max_position_embeddings", 0) or 0),
        dtype=str(text.get("dtype") or hf.get("dtype") or "bfloat16"),
        tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
        attention_mode=attention_mode,
        prompt=str(decision.get("prompt", "d3")),
        temperature=float(decision.get("temperature", 1.0)),
        readout_dtype=str(decision.get("readout_dtype", "float32")),
        codes=tuple(str(c) for c in codes),
        token_ids=tuple(int(t) for t in token_ids),
        validated_length=int(validated) if validated else None,
        context_limit=int(max_length) if max_length else None,
        max_options=len(codes),
        image_placements=tuple(decision.get("systemone", {}).get("image_placements", ()) or ()),
        question_types=tuple(decision.get("systemone", {}).get("question_types", ()) or ()),
        systemone=decision.get("systemone", {}) or {},
        vision=vision,
        image_token_id=hf.get("image_token_id"),
        video_token_id=hf.get("video_token_id"),
        vision_start_token_id=hf.get("vision_start_token_id"),
        vision_end_token_id=hf.get("vision_end_token_id"),
        state_cache_bytes=budget,
    )
