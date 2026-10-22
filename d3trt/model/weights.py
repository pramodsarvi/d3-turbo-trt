"""Streaming reader for the d3 safetensors shards.

The full checkpoint is ~52 GB BF16 across 11 shards. Loading every tensor into
host memory before building the model doubles peak RAM, so this reader keeps a
name -> shard map and loads only the tensors for the layer being built.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from safetensors import safe_open


class ShardedWeights:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        index_path = self.root / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = {}
        self._total_bytes: int | None = None
        if index_path.exists():
            data = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(data.get("weight_map") or {})
            total = (data.get("metadata") or {}).get("total_size")
            self._total_bytes = int(total) if total else None
        else:
            for shard in sorted(self.root.glob("*.safetensors")):
                if shard.name == "readout.safetensors":
                    continue
                with safe_open(str(shard), framework="pt") as handle:
                    for key in handle.keys():
                        self.weight_map[key] = shard.name
        if not self.weight_map:
            raise FileNotFoundError(f"no safetensors weights under {self.root}")
        self.shards = sorted(set(self.weight_map.values()))

    def total_bytes(self) -> int:
        """Checkpoint size; the index metadata when present, else the shard files on disk."""
        if self._total_bytes is None:
            self._total_bytes = sum((self.root / shard).stat().st_size for shard in self.shards)
        return self._total_bytes

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def keys(self) -> Iterable[str]:
        return self.weight_map.keys()

    def get(self, name: str, *, device: str | torch.device = "cpu") -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"weight {name!r} is not in the checkpoint")
        with safe_open(str(self.root / shard), framework="pt") as handle:
            tensor = handle.get_tensor(name)
        return tensor.to(device) if str(device) != "cpu" else tensor

    def get_many(
        self,
        names: Iterable[str],
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        by_shard: dict[str, list[str]] = {}
        for name in names:
            shard = self.weight_map.get(name)
            if shard is None:
                raise KeyError(f"weight {name!r} is not in the checkpoint")
            by_shard.setdefault(shard, []).append(name)
        out: dict[str, torch.Tensor] = {}
        for shard, keys in by_shard.items():
            with safe_open(str(self.root / shard), framework="pt") as handle:
                for key in keys:
                    tensor = handle.get_tensor(key)
                    out[key] = tensor.to(device) if str(device) != "cpu" else tensor
        return out

    def readout(self, *, device: str | torch.device = "cpu") -> torch.Tensor:
        path = self.root / "readout.safetensors"
        with safe_open(str(path), framework="pt") as handle:
            return handle.get_tensor("weight").to(device)

    def describe(self) -> Mapping[str, object]:
        return {
            "root": str(self.root),
            "shards": len(self.shards),
            "tensors": len(self.weight_map),
        }
