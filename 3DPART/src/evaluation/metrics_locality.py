"""Locality metrics for minimal-change evaluation.

Measures whether the repair patch stays within the repair region
near the removed part, rather than spreading to unaffected areas.
"""

import numpy as np
import trimesh
from typing import Dict

from ..geometry.bbox import compute_bbox, expand_bbox, point_in_bbox


def compute_locality_metrics(
    repair_result: Dict,
    removed_part_mesh: trimesh.Trimesh,
    margin: float = 0.05,
) -> Dict[str, float]:
    """Compute locality metrics for the repair.

    The repair zone is defined as the expanded bounding box of the removed part.

    Metrics:
        - n_faces_inside: number of new faces with centroid inside repair zone
        - n_faces_outside: number of new faces with centroid outside repair zone
        - locality_ratio: fraction of new faces inside the repair zone
    """
    new_faces = repair_result.get("new_faces", np.empty((0, 3), dtype=int))
    repaired_mesh = repair_result.get("repaired_mesh")

    if repaired_mesh is None or len(new_faces) == 0:
        return {
            "n_faces_inside": 0.0,
            "n_faces_outside": 0.0,
            "locality_ratio": 0.0,
        }

    # Compute repair zone
    bbox_min, bbox_max = compute_bbox(removed_part_mesh)
    bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)

    vertices = repaired_mesh.vertices

    n_inside = 0
    n_outside = 0

    for face in new_faces:
        # Compute centroid of this triangle
        centroid = vertices[face].mean(axis=0)
        if point_in_bbox(centroid, bbox_min_exp, bbox_max_exp):
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
