"""Metadata I/O for dataset samples."""

import os
import json
from typing import Optional


def save_meta(meta: dict, path: str):
    """Save sample metadata to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_meta(path: str) -> dict:
    """Load sample metadata from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_sample_meta(sample_id: str, original_dir: str, category: str,
                       removed_part_name: str, removed_obj_files: list,
                       n_verts_before: int, n_faces_before: int,
                       n_verts_after: int, n_faces_after: int,
                       removed_part_count: int = 1,
                       removed_obj_groups: list = None) -> dict:
    return {
        "sample_id": sample_id,
        "original_dir": original_dir,
        "category": category,
        "removed_part_name": removed_part_name,
        "removed_part_count": removed_part_count,
        "removed_obj_files": removed_obj_files,
        "removed_obj_groups": removed_obj_groups if removed_obj_groups is not None else [],
        "mesh_before": {
            "n_vertices": n_verts_before,
            "n_faces": n_faces_before,
        },
        "mesh_after": {
            "n_vertices": n_verts_after,
            "n_faces": n_faces_after,
        },
    }
