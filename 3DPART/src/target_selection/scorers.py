"""Scoring functions for boundary loop target selection."""

import numpy as np
import trimesh
from typing import List

from ..geometry.bbox import compute_bbox, expand_bbox, loop_near_bbox, loop_bbox_overlap_score


def loop_proximity_score(mesh: trimesh.Trimesh, loop: List[int],
                         removed_part_mesh: trimesh.Trimesh,
                         margin: float = 0.05) -> float:
    """Score a loop based on proximity to the removed part bounding box.

    Higher score = closer to removed part.
    """
    bbox_min, bbox_max = compute_bbox(removed_part_mesh)
    bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)

    loop_verts = mesh.vertices[loop]
    bbox_center = (bbox_min + bbox_max) / 2.0
    loop_center = loop_verts.mean(axis=0)

    dist = np.linalg.norm(loop_center - bbox_center)
    diag = np.linalg.norm(bbox_max - bbox_min)

    if diag < 1e-10:
        return 0.0

    # Inverse distance score, normalized by diagonal
    return float(max(0, 1.0 - dist / (diag * 2.0)))


def loop_overlap_score(mesh: trimesh.Trimesh, loop: List[int],
                       removed_part_mesh: trimesh.Trimesh,
                       margin: float = 0.05) -> float:
    """Score a loop based on vertex overlap with the removed part bbox."""
    bbox_min, bbox_max = compute_bbox(removed_part_mesh)
    bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)

    loop_verts = mesh.vertices[loop]
    return loop_bbox_overlap_score(loop_verts, bbox_min_exp, bbox_max_exp)
