"""Opening closure metrics.

Measures whether the target hole caused by part removal is effectively
closed after repair.
"""

import numpy as np
import trimesh
from typing import List, Dict

from ..geometry.boundary import extract_boundary_loops, loop_perimeter
from ..geometry.bbox import compute_bbox, expand_bbox, loop_near_bbox


def compute_closure_metrics(
    damaged_mesh: trimesh.Trimesh,
    repaired_mesh: trimesh.Trimesh,
    removed_part_mesh: trimesh.Trimesh,
    target_loops_before: List[List[int]],
    margin: float = 0.05,
    proximity_threshold: float = 0.1,
) -> Dict[str, float]:
    """Compute closure metrics for repair evaluation.

    Metrics:
        - target_loop_length_before: total perimeter of target loops before repair
        - target_loop_length_after: residual target loop length after repair
        - improvement: reduction in target loop length
        - closure_ratio: fraction of target loop length closed
    """
    # Compute target loop length before repair
    total_before = 0.0
    for loop in target_loops_before:
        total_before += loop_perimeter(damaged_mesh, loop)

    # Find residual target loops after repair
    loops_after = extract_boundary_loops(repaired_mesh)
    bbox_min, bbox_max = compute_bbox(removed_part_mesh)
    bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)

    total_after = 0.0
    for loop in loops_after:
        loop_verts = repaired_mesh.vertices[loop]
        if loop_near_bbox(loop_verts, bbox_min_exp, bbox_max_exp, proximity_threshold):
            total_after += loop_perimeter(repaired_mesh, loop)

    improvement = total_before - total_after
    closure_ratio = 1.0 - (total_after / max(total_before, 1e-10))

    return {
        "target_loop_length_before": float(total_before),
        "target_loop_length_after": float(total_after),
        "improvement": float(improvement),
        "closure_ratio": float(np.clip(closure_ratio, 0, 1)),
    }
