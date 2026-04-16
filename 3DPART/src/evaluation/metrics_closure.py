import numpy as np
import trimesh
from typing import Dict, List

from src.geometry.boundary import extract_boundary_loops
from src.geometry.bbox import compute_bbox, expand_bbox, loop_near_bbox


def loop_perimeter(mesh: trimesh.Trimesh, loop: List[int]) -> float:
    """
    Compute perimeter length of a boundary loop.
    """
    if loop is None or len(loop) < 2:
        return 0.0

    verts = mesh.vertices
    total = 0.0
    n = len(loop)

    for i in range(n):
        v0 = verts[loop[i]]
        v1 = verts[loop[(i + 1) % n]]
        total += np.linalg.norm(v1 - v0)

    return float(total)


def _split_removed_components(removed_part_mesh: trimesh.Trimesh):
    """
    Split removed_part_mesh into connected components.
    Works for both single-part and multi-part removal.
    """
    if hasattr(removed_part_mesh, "split"):
        try:
            parts = list(removed_part_mesh.split(only_watertight=False))
        except TypeError:
            parts = list(removed_part_mesh.split())
        parts = [p for p in parts if hasattr(p, "faces") and len(p.faces) > 0]
        if parts:
            return parts
    return [removed_part_mesh]


def compute_closure_metrics(
    damaged_mesh: trimesh.Trimesh,
    repaired_mesh: trimesh.Trimesh,
    removed_part_mesh: trimesh.Trimesh,
    target_loops_before: List[List[int]],
    margin: float = 0.05,
    proximity_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Compute closure metrics for the target opening(s).

    For multi-part removal, residual loops after repair are counted if they are
    near ANY removed connected component.
    """
    if repaired_mesh is None:
        return {
            "closure_residual": float("inf"),
            "improvement": 0.0,
        }

    total_before = 0.0
    for loop in target_loops_before:
        total_before += loop_perimeter(damaged_mesh, loop)

    loops_after = extract_boundary_loops(repaired_mesh)

    removed_components = _split_removed_components(removed_part_mesh)

    expanded_boxes = []
    for comp in removed_components:
        bbox_min, bbox_max = compute_bbox(comp)
        bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)
        expanded_boxes.append((bbox_min_exp, bbox_max_exp))

    total_after = 0.0
    for loop in loops_after:
        loop_verts = repaired_mesh.vertices[loop]

        hit = False
        for bbox_min_exp, bbox_max_exp in expanded_boxes:
            if loop_near_bbox(loop_verts, bbox_min_exp, bbox_max_exp, proximity_threshold):
                hit = True
                break

        if hit:
            total_after += loop_perimeter(repaired_mesh, loop)

    improvement = total_before - total_after

    return {
        "closure_residual": float(total_after),
        "improvement": float(improvement),
    }