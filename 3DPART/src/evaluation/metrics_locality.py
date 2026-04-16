import numpy as np
import trimesh
from typing import Dict

from src.geometry.bbox import compute_bbox, expand_bbox, point_in_bbox


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


def compute_locality_metrics(
    repair_result: Dict,
    removed_part_mesh: trimesh.Trimesh,
    margin: float = 0.05,
) -> Dict[str, float]:
    """
    Compute locality statistics for newly added faces.

    For multi-part removal, a new face is counted as 'inside' if its centroid
    lies inside the expanded bounding box of ANY removed connected component.
    """
    new_faces = repair_result.get("new_faces", np.empty((0, 3), dtype=int))
    repaired_mesh = repair_result.get("repaired_mesh")

    if repaired_mesh is None or len(new_faces) == 0:
        return {
            "n_faces_inside": 0.0,
            "n_faces_outside": 0.0,
            "locality_ratio": 0.0,
        }

    removed_components = _split_removed_components(removed_part_mesh)

    expanded_boxes = []
    for comp in removed_components:
        bbox_min, bbox_max = compute_bbox(comp)
        bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)
        expanded_boxes.append((bbox_min_exp, bbox_max_exp))

    vertices = repaired_mesh.vertices
    n_inside = 0
    n_outside = 0

    for face in new_faces:
        centroid = vertices[face].mean(axis=0)

        inside_any = False
        for bbox_min_exp, bbox_max_exp in expanded_boxes:
            if point_in_bbox(centroid, bbox_min_exp, bbox_max_exp):
                inside_any = True
                break

        if inside_any:
            n_inside += 1
        else:
            n_outside += 1

    total = n_inside + n_outside
    locality_ratio = n_inside / max(total, 1)

    return {
        "n_faces_inside": float(n_inside),
        "n_faces_outside": float(n_outside),
        "locality_ratio": float(locality_ratio),
    }