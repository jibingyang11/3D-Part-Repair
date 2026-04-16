"""Target boundary loop selection strategies."""

import numpy as np
import trimesh
from typing import List, Tuple

from ..geometry.bbox import compute_bbox, expand_bbox, loop_near_bbox, loop_bbox_overlap_score
from ..geometry.boundary import extract_boundary_loops, loop_perimeter, largest_loop


def _split_removed_components(removed_part_mesh):
    if hasattr(removed_part_mesh, "split"):
        try:
            parts = list(removed_part_mesh.split(only_watertight=False))
        except TypeError:
            parts = list(removed_part_mesh.split())
        parts = [p for p in parts if hasattr(p, "faces") and len(p.faces) > 0]
        if parts:
            return parts
    return [removed_part_mesh]


def select_target_loops_by_bbox(
    mesh: trimesh.Trimesh,
    loops: List[List[int]],
    removed_part_mesh: trimesh.Trimesh,
    margin: float = 0.05,
    proximity_threshold: float = 0.1,
) -> List[List[int]]:
    """
    Multi-part aware target selection:
    for each connected component of removed_part_mesh, select nearby loops.
    """
    removed_components = _split_removed_components(removed_part_mesh)

    selected_idx = set()

    for comp in removed_components:
        bbox_min, bbox_max = compute_bbox(comp)
        bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)

        # strict pass
        local_hits = []
        for i, loop in enumerate(loops):
            loop_verts = mesh.vertices[loop]
            if loop_near_bbox(loop_verts, bbox_min_exp, bbox_max_exp, proximity_threshold):
                local_hits.append(i)

        # relaxed pass
        if not local_hits:
            for i, loop in enumerate(loops):
                loop_verts = mesh.vertices[loop]
                if loop_near_bbox(loop_verts, bbox_min_exp, bbox_max_exp, proximity_threshold * 3.0):
                    local_hits.append(i)

        # fallback: closest loop to this component center
        if not local_hits and loops:
            bbox_center = (bbox_min + bbox_max) / 2.0
            min_dist = float("inf")
            closest_i = 0
            for i, loop in enumerate(loops):
                loop_center = mesh.vertices[loop].mean(axis=0)
                dist = np.linalg.norm(loop_center - bbox_center)
                if dist < min_dist:
                    min_dist = dist
                    closest_i = i
            local_hits = [closest_i]

        selected_idx.update(local_hits)

    return [loops[i] for i in sorted(selected_idx)]


def select_largest_loop(
    mesh: trimesh.Trimesh,
    loops: List[List[int]],
) -> List[List[int]]:
    """Select only the largest boundary loop by perimeter.

    This is the naive largest-hole-only baseline.
    """
    if not loops:
        return []
    lg = largest_loop(loops, mesh)
    return [lg] if lg else []


def select_all_loops(
    mesh: trimesh.Trimesh,
    loops: List[List[int]],
) -> List[List[int]]:
    """Select all boundary loops (fill everything)."""
    return loops
