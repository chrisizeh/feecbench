"""PyTorch dataset and variable-size-mesh (disjoint-union) batching.

* ``FEECDataset.__getitem__`` returns a dict: dense torch field tensors (float64, optionally
  normalized), operators kept as scipy CSR (batched once in ``collate_feec``), cell counts,
  boundary masks, identity and params.
* ``collate_feec`` builds the disjoint union: block-diagonal operators (converted once to
  ``torch.sparse_csr_tensor``, float64 -- torch sparse has no int8 matmul), fields
  concatenated along the cell axis, per-degree ``batch_index`` owner vectors. It asserts a
  homogeneous field schema and carries per-sample params.
* ``get_config`` resolves against the FULL index (cross-resolution identity is a corpus
  property, not a property of the filtered view).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from .io import load_sample
from .normalization import Normalizer, load_stats
from .schema import LEVELS, FEECSample, surface_identity


class FEECDataset(Dataset):
    """Dataset over ``<root>/index.json``; ``split`` and exact-match ``filters`` select entries;
    ``normalize`` in {None, "zscore", "rms"} applies the train-split stats from ``stats.json``."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        filters: dict | None = None,
        normalize: str | None = None,
    ):
        self.root = Path(root)
        self.index_metadata = json.loads((self.root / "index.json").read_text())
        if self.index_metadata.get("complete") is False:
            raise ValueError("corpus is explicitly incomplete")
        self._all_entries = []
        for original in self.index_metadata["samples"]:
            entry = dict(original)
            if "resolution" in entry:
                identity = surface_identity(entry)
                for key, value in identity.items():
                    entry.setdefault(key, value)
            self._all_entries.append(entry)
        wanted = {**(filters or {}), **({"split": split} if split is not None else {})}
        self.entries = [e for e in self._all_entries if all(e.get(k) == v for k, v in wanted.items())]
        if not self.entries:
            raise ValueError("no samples match the given split/filters")
        self.normalizer = Normalizer(load_stats(self.root), normalize) if normalize else None

    def __len__(self) -> int:
        return len(self.entries)

    def load_full(self, idx: int) -> FEECSample:
        """The full FEECSample behind item ``idx`` (always unnormalized)."""
        return load_sample(self.root / self.entries[idx]["path"])

    def denormalize(self, name: str, data):
        return data if self.normalizer is None else self.normalizer.denormalize(name, data)

    def __getitem__(self, idx: int) -> dict:
        s = self.load_full(idx)
        norm = self.normalizer.normalize if self.normalizer else (lambda _n, d: d)
        meta = {n: {"form_degree": f.form_degree, "role": f.role, "units": f.units} for n, f in s.fields.items()}
        return {
            "fields": {n: torch.from_numpy(np.ascontiguousarray(norm(n, f.data))) for n, f in s.fields.items()},
            "field_meta": meta,
            "operators": dict(s.operators),
            "n_cells": s.mesh.n_cells,
            "boundary_vertices": torch.from_numpy(s.mesh.boundary_vertices),
            "boundary_edges": torch.from_numpy(s.mesh.boundary_edges),
            "config_id": s.ids.config_id,
            "sample_id": s.ids.sample_id,
            "resolution_level": s.ids.resolution_level,
            "split": s.ids.split,
            "params": s.params,
            "star_types": s.star_types,
            "orientation_convention": s.mesh.orientation_convention,
            "auxiliary": s.auxiliary,
            "normalized": self.normalizer.mode if self.normalizer else None,
        }

    def get_config(self, config_id: str, refinement: str = "nested") -> dict[str, FEECSample]:
        """All discretizations of one configuration from the full index: ``{level: sample}`` in
        coarse -> fine order for ``"nested"``, ``{sample_id: sample}`` for ``"remeshed"``."""
        candidates = [e for e in self._all_entries if e["config_id"] == config_id]
        for entry in candidates:
            if any(key not in entry for key in ("sample_id", "resolution_level", "refinement")):
                # Nonstandard indexes need inspection only when this configuration is requested.
                import h5py

                with h5py.File(self.root / entry["path"], "r") as stream:
                    if "ids" in stream:
                        identity = dict(stream["ids"].attrs)
                    elif "metadata" in stream:
                        identity = surface_identity(json.loads(stream["metadata"][()].decode()))
                    else:
                        raise ValueError(f"missing identity metadata in {entry['path']}")
                for key in ("sample_id", "resolution_level", "refinement"):
                    entry.setdefault(key, identity[key])
        entries = [e for e in candidates if e["refinement"] == refinement]
        if not entries:
            raise KeyError(f"{config_id} ({refinement})")
        if refinement == "nested":
            entries.sort(key=lambda e: LEVELS.index(e["resolution_level"]))
            return {e["resolution_level"]: load_sample(self.root / e["path"]) for e in entries}
        return {e["sample_id"]: load_sample(self.root / e["path"]) for e in entries}


@dataclass
class FEECBatch:
    """Disjoint union of B samples."""

    fields: dict[str, torch.Tensor]  # concatenated along the cell axis
    field_meta: dict[str, dict]  # shared schema
    operators: dict[str, torch.Tensor]  # block-diagonal sparse CSR
    batch_index: dict[int, torch.Tensor]  # degree -> (sum n_k,) owner
    n_cells: list[tuple[int, int, int]]
    sample_ids: list[str]
    params: list[dict]
    boundary_vertices: torch.Tensor
    boundary_edges: torch.Tensor
    star_types: list[dict] = field(default_factory=list)
    auxiliary: list[dict] = field(default_factory=list)
    orientation_conventions: list[str] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return len(self.sample_ids)

    def to(self, device) -> FEECBatch:
        """Move all tensors; a missing CUDA device raises torch's own clear error."""
        mv = lambda d: {k: v.to(device) for k, v in d.items()}  # noqa: E731
        return FEECBatch(
            mv(self.fields), self.field_meta, mv(self.operators), mv(self.batch_index), self.n_cells,
            self.sample_ids, self.params, self.boundary_vertices.to(device), self.boundary_edges.to(device),
            self.star_types, self.auxiliary, self.orientation_conventions,
        )


def _to_torch_csr(m: sp.csr_matrix) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.from_numpy(m.indptr.astype(np.int64)),
        torch.from_numpy(m.indices.astype(np.int64)),
        torch.from_numpy(m.data.astype(np.float64)),
        size=m.shape,
    )


def collate_feec(batch: list[dict]) -> FEECBatch:
    """Disjoint-union collate; raises ValueError on heterogeneous field schemas."""
    if not batch:
        raise ValueError("collate_feec: empty batch")
    meta = batch[0]["field_meta"]
    for item in batch:
        for name, values in item["fields"].items():
            if name not in meta or item["field_meta"] != meta:
                raise ValueError("collate_feec: heterogeneous field schemas")
            if values.shape[:-1] != batch[0]["fields"][name].shape[:-1]:
                raise ValueError(f"collate_feec: incompatible query lengths for {name}")
            if values.shape[-1] != item["n_cells"][meta[name]["form_degree"]]:
                raise ValueError(f"collate_feec: invalid cell dimension for {name}")
    for item in batch[1:]:
        if item["field_meta"] != meta:
            raise ValueError(
                f"collate_feec: heterogeneous field schemas ({item['sample_id']} vs {batch[0]['sample_id']})"
            )
    return FEECBatch(
        fields={n: torch.cat([it["fields"][n] for it in batch], dim=-1) for n in meta},
        field_meta=meta,
        operators={
            n: _to_torch_csr(sp.block_diag([it["operators"][n] for it in batch], format="csr"))
            for n in batch[0]["operators"]
        },
        batch_index={
            k: torch.cat([torch.full((it["n_cells"][k],), i, dtype=torch.int64) for i, it in enumerate(batch)])
            for k in (0, 1, 2)
        },
        n_cells=[it["n_cells"] for it in batch],
        sample_ids=[it["sample_id"] for it in batch],
        params=[it["params"] for it in batch],
        boundary_vertices=torch.cat([it["boundary_vertices"] for it in batch]),
        boundary_edges=torch.cat([it["boundary_edges"] for it in batch]),
        star_types=[it.get("star_types", {}) for it in batch],
        auxiliary=[it.get("auxiliary", {}) for it in batch],
        orientation_conventions=[it.get("orientation_convention", "") for it in batch],
    )
