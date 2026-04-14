"""Target boundary loop selection strategies."""

import numpy as np
import trimesh
from typing import List, Tuple

from ..geometry.bbox import compute_bbox, expand_bbox, loop_near_bbox, loop_bbox_overlap_score
from ..geometry.boundary import extract_boundary_loops, loop_perimeter, largest_loop


def select_target_loops_by_bbox(
    mesh: trimesh.Trimesh,
    loops: List[List[int]],
    removed_part_mesh: trimesh.Trimesh,
    margin: float = 0.05,
    proximity_threshold: float = 0.1,
) -> List[List[int]]:
    """Select boundary loops near the removed part bounding box.

    This is the removed-part-aware target selection strategy.

    Args:
        mesh: The damaged mesh
        loops: All boundary loops (vertex index lists)
        removed_part_mesh: The mesh of the removed part
        margin: Bounding box expansion margin
        proximity_threshold: Distance threshold for proximity test

    Returns: List of target boundary loops
    """
    bbox_min, bbox_max = compute_bbox(removed_part_mesh)
    bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)

    target_loops = []
    for loop in loops:
        loop_verts = mesh.vertices[loop]
        if loop_near_bbox(loop_verts, bbox_min_exp, bbox_max_exp, proximity_threshold):
            target_loops.append(loop)

    # If no loops found with strict threshold, try with relaxed threshold
    if not target_loops:
        for loop in loops:
            loop_verts = mesh.vertices[loop]
            if loop_near_bbox(loop_verts, bbox_min_exp, bbox_max_exp,
                              proximity_threshold * 3.0):
                target_loops.append(loop)

    # If still no loops, pick the one closest to bbox center
    if not target_loops and loops:
        bbox_center = (bbox_min + bbox_max) / 2.0
        min_dist = float('inf')
        closest = loops[0]
        for loop in loops:
            loop_center = mesh.vertices[loop].mean(axis=0)
            dist = np.linalg.norm(loop_center - bbox_center)
            if dist < min_dist:
                min_dist = dist
                closest = loop
        target_loops = [closest]

    return target_loops


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
