"""Load individual samples from the dataset."""

import os
import trimesh
from pathlib import Path
from typing import Optional, Tuple

from ..io.mesh_io import load_mesh
from ..io.meta_io import load_meta


class SampleLoader:
    """Load a single sample's meshes and metadata."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self, sample_id: str) -> dict:
        """Load all data for a sample.

        Returns dict with keys:
            - sample_id: str
            - complete_mesh: trimesh.Trimesh
            - damaged_mesh: trimesh.Trimesh
            - removed_part_mesh: trimesh.Trimesh
            - meta: dict
        """
        sample_dir = self.data_dir / sample_id

        complete = load_mesh(str(sample_dir / "complete.obj"))
        damaged = load_mesh(str(sample_dir / "damaged.obj"))
        removed = load_mesh(str(sample_dir / "removed_part.obj"))
        meta = load_meta(str(sample_dir / "meta.json"))

        return {
            "sample_id": sample_id,
            "complete_mesh": complete,
            "damaged_mesh": damaged,
            "removed_part_mesh": removed,
            "meta": meta,
        }

    def load_damaged(self, sample_id: str) -> trimesh.Trimesh:
        """Load only the damaged mesh."""
        return load_mesh(str(self.data_dir / sample_id / "damaged.obj"))

    def load_removed_part(self, sample_id: str) -> trimesh.Trimesh:
        """Load only the removed part mesh."""
        return load_mesh(str(self.data_dir / sample_id / "removed_part.obj"))

    def load_meta(self, sample_id: str) -> dict:
        """Load only the metadata."""
        return load_meta(str(self.data_dir / sample_id / "meta.json"))

    def has_sample(self, sample_id: str) -> bool:
        """Check if a sample exists."""
        sample_dir = self.data_dir / sample_id
        return (sample_dir / "damaged.obj").exists()
