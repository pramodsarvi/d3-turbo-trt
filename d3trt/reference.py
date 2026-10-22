"""Adapter around the reference d3 package runtime.

The d3 model package (the HuggingFace repo) contains the only authoritative
implementation of its prompt format, answer-code mapping, readout offsets,
Box/Track composition and error contract. We import those modules rather than
copying them.

``build_frontend`` constructs the reference ``D3`` object *without* loading the
52 GB backbone. The reference ``D3.prepare`` and ``D3.respond`` are host-only
(they tokenize, render, validate and assemble answers), so the object works with
``backbone = None``. The expensive forward pass is supplied by a backend.

This is deliberately the same code path the model card's reference runtime uses;
the first parity gate is that our prepared batches are byte-identical to it.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import D3Config, load_config
from .errors import ConfigurationError

_LOCK = threading.RLock()
_MODULES: dict[str, ModuleType] = {}


def _candidate_dirs(model_dir: Path, reference_dir: str | Path | None) -> list[Path]:
    dirs: list[Path] = []
    if reference_dir:
        dirs.append(Path(reference_dir).expanduser().resolve())
    env = os.environ.get("D3_REFERENCE_DIR")
    if env:
        dirs.append(Path(env).expanduser().resolve())
    dirs.append(model_dir)
    dirs.append(Path(__file__).resolve().parent / "_vendor")
    return dirs


def import_reference(
    model_dir: str | Path,
    reference_dir: str | Path | None = None,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    """Import ``d3_format``, ``d3_compose`` and ``d3_runtime``.

    ``d3_runtime`` must come from the model package (it contains the reference
    forward pass and the prompt/response contract). ``d3_format`` and
    ``d3_compose`` fall back to the vendored copies shipped with this package.
    """
    root = Path(model_dir).expanduser().resolve()
    candidates = _candidate_dirs(root, reference_dir)

    with _LOCK:
        # Put the model directory first, then the vendored contract modules.
        for path in reversed(candidates):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        try:
            fmt = _MODULES.get("d3_format") or importlib.import_module("d3_format")
            compose = _MODULES.get("d3_compose") or importlib.import_module("d3_compose")
        except ImportError as exc:
            raise ConfigurationError(
                "cannot import the d3 prompt contract (d3_format.py / d3_compose.py); "
                "pass reference_dir= or install the d3 model package"
            ) from exc
        try:
            runtime = _MODULES.get("d3_runtime") or importlib.import_module("d3_runtime")
        except ImportError as exc:
            raise ConfigurationError(
                f"cannot import d3_runtime from {root}; the model directory must contain "
                "d3_runtime.py (the reference runtime), or set D3_REFERENCE_DIR"
            ) from exc
        _MODULES.update(d3_format=fmt, d3_compose=compose, d3_runtime=runtime)
        return fmt, compose, runtime


def import_server(
    model_dir: str | Path,
    reference_dir: str | Path | None = None,
) -> ModuleType:
    """Import the reference HTTP server module (``d3_server``).

    The d3-trt server reuses the reference routing, auth and request parsing
    through this module; only the loaded model is swapped.
    """
    import_reference(model_dir, reference_dir)
    try:
        with _LOCK:
            module = _MODULES.get("d3_server") or importlib.import_module("d3_server")
            _MODULES["d3_server"] = module
            return module
    except ImportError as exc:
        raise ConfigurationError(
            "cannot import d3_server; the d3 package (or d3trt/_vendor) must provide it"
        ) from exc


class Frontend:
    """The reference d3 frontend without a backbone.

    ``prepare`` returns the reference ``Prepared`` object. ``respond`` assembles
    the reference answer mapping. The backend is responsible only for turning a
    ``Prepared`` into ``{question_id: probabilities}`` plus a token count.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        reference_dir: str | Path | None = None,
        config: D3Config | None = None,
        batch_size: int = 8,
        device: str = "cpu",
        permutation_average: bool | None = None,
        model_name: str | None = None,
        readout_dtype: str | None = None,
        max_length: int | None = None,
    ):
        self.config = config or load_config(model_dir)
        self.fmt, self.compose, self.rt = import_reference(self.config.root, reference_dir)
        self._d3 = self._build_skeleton(
            batch_size=batch_size,
            device=device,
            permutation_average=permutation_average,
            model_name=model_name,
            readout_dtype=readout_dtype,
            max_length=max_length,
        )

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_skeleton(
        self,
        *,
        batch_size: int,
        device: str,
        permutation_average: bool | None,
        model_name: str | None,
        readout_dtype: str | None,
        max_length: int | None,
    ) -> Any:
        rt = self.rt
        import torch
        from transformers import AutoTokenizer

        D3 = rt.D3
        self_obj = object.__new__(D3)
        root = self.config.root
        config = dict(self.config.decision)

        self_obj.torch = torch
        self_obj.root = root
        self_obj.manifest = self.config.manifest or None
        self_obj.config = config
        self_obj.backbone_type = rt.backbone_type(root)

        self_obj.prompt = self.config.prompt
        self_obj.attention_mode = self.config.attention_mode
        self_obj.blocked = self_obj.attention_mode == "prefix_bidir"

        temperature = float(config.get("temperature", 1.0))
        if not (math.isfinite(temperature) and temperature > 0):
            raise ConfigurationError(f"temperature must be finite and positive, got {temperature}")
        self_obj.temperature = temperature
        self_obj.readout_dtype = readout_dtype or config.get("readout_dtype", "float32")
        if self_obj.readout_dtype not in ("float32", "bfloat16"):
            raise ConfigurationError(
                f"readout_dtype must be float32 or bfloat16, got {self_obj.readout_dtype!r}"
            )

        limit = config.get("max_length") if max_length is None else max_length
        validated = config.get("validated_length")
        self_obj.context_limit = int(limit) if limit else None
        self_obj.validated_length = int(validated) if validated else None
        self_obj.max_length = self_obj.question_cap(False)

        systemone = config.get("systemone") or {}
        self_obj.question_types = ("choice", "noul", "score") + tuple(
            kind for kind in ("box", "track") if kind in (systemone.get("question_types") or ())
        )
        self_obj.image_placements = ("state",) + tuple(
            p for p in ("instructions", "criteria") if p in (systemone.get("image_placements") or ())
        )
        self_obj.box_calibration = systemone.get("box_calibration", config.get("box_calibration"))

        self_obj.batch_size = int(batch_size)
        if self_obj.batch_size < 1:
            raise ConfigurationError(f"batch_size must be at least 1, got {self_obj.batch_size}")
        self_obj.model_name = model_name or self.config.model_name
        self_obj.permutation_average = bool(
            config.get("permutation_average", False)
            if permutation_average is None
            else permutation_average
        )

        self_obj.tokenizer = AutoTokenizer.from_pretrained(str(root))
        self_obj.tokenizer.padding_side = "left"
        codes, token_ids = rt.answer_codes(self_obj.tokenizer)
        if list(config.get("codes") or []) != codes or list(config.get("token_ids") or []) != token_ids:
            raise ConfigurationError(
                "the answer codes or token ids in decision_config.json differ from the tokenizer"
            )
        self_obj.codes = codes
        self_obj.token_ids = token_ids
        probe = "State:\nA"
        if (
            self_obj.tokenizer(probe)["input_ids"]
            != self_obj.tokenizer(probe, add_special_tokens=False)["input_ids"]
        ):
            raise ConfigurationError("tokenizer adds special tokens; prompts are tokenized as rendered")
        self_obj.pad_id = self_obj.tokenizer.pad_token_id
        if self_obj.pad_id is None:
            raise ConfigurationError("tokenizer has no pad token")

        self_obj.device = torch.device(device)
        self_obj.kernels = {}
        self_obj.backbone = None
        self_obj.readout = None
        self_obj.encoder = None
        self_obj.state_once = None
        self_obj.fast = None
        self_obj.fast_skipped = None
        self_obj.state_once_fused = None
        self_obj.loaded_seconds = 0.0

        # Media preprocessing is host-side. Keep the reference's feature gates
        # so a text-only checkpoint still refuses images exactly like the
        # reference runtime.
        self_obj.processor = None
        self_obj.image_unavailable = None
        self_obj.video_unavailable = None
        if not rt.vision_weights_present(root):
            self_obj.image_unavailable = "the checkpoint has no vision tower (visual.* weights)"
        else:
            try:
                self_obj.processor = rt.load_processor(root)
            except Exception as exc:  # noqa: BLE001
                self_obj.image_unavailable = (
                    f"the image processor failed to load ({type(exc).__name__}: {exc})"
                )
        if self_obj.image_unavailable is not None:
            self_obj.video_unavailable = self_obj.image_unavailable
        elif self_obj.backbone_type != "qwen3_5":
            self_obj.video_unavailable = "video inputs need a Qwen3.5 backbone"
        elif getattr(self_obj.processor, "video_processor", None) is None:
            self_obj.video_unavailable = "the checkpoint has no video processor"
        else:
            try:
                rt.configure_video_processor(self_obj.processor)
            except Exception as exc:  # noqa: BLE001
                self_obj.video_unavailable = (
                    f"the video processor failed to load ({type(exc).__name__}: {exc})"
                )
        return self_obj

    # ------------------------------------------------------------------
    # reference-shaped API
    # ------------------------------------------------------------------
    @property
    def d3(self) -> Any:
        return self._d3

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        return self._d3.prepare(*args, **kwargs)

    def respond(self, prepared: Any, probabilities: Any, tokens: int) -> dict[str, Any]:
        return self._d3.respond(prepared, probabilities, tokens)

    def question_cap(self, long_context: bool) -> int | None:
        return self._d3.question_cap(long_context)

    def state_block(self, state: Any, text: str) -> str:
        return self._d3.state_block(state, text)

    def runtime_info(self) -> dict[str, Any]:
        base = {
            "model": self._d3.model_name,
            "attention_mode": self._d3.attention_mode,
            "prompt": self._d3.prompt,
            "max_options": len(self._d3.codes),
            "validated_length": self._d3.validated_length,
            "max_length": self._d3.context_limit,
            "image": self._d3.image_unavailable is None,
            "video": self._d3.video_unavailable is None,
            "image_reason": self._d3.image_unavailable,
            "video_reason": self._d3.video_unavailable,
        }
        return base

    def describe_tokens(self, prepared: Any) -> dict[str, Any]:
        """Small helper for parity tests: the exact state/question token split."""
        first = prepared.keys[0] if prepared.keys else None
        if first is None:
            return {}
        sequence = prepared.sequences.get(first)
        if sequence is None:
            return {"state": prepared.states.get(first)}
        state = prepared.states[first]
        return {
            "state": state,
            "state_ids": list(sequence[:state]),
            "question_ids": list(sequence[state:]),
        }

    # Image/video preparation is exposed for the fast backend.
    def image_embeddings(self, inputs: Any) -> Any:
        return self._d3.image_embeddings(inputs)
