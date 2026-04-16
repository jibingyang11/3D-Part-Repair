"""Labeling boundary loops as target or non-target for evaluation."""

import numpy as np
import trimesh
from typing import List, Dict

from ..geometry.bbox import compute_bbox, expand_bbox, loop_near_bbox


def label_loops(mesh: trimesh.Trimesh, loops: List[List[int]],
                removed_part_mesh: trimesh.Trimesh,
                margin: float = 0.05,
                threshold: float = 0.1) -> List[Dict]:
    """Label each boundary loop as target (1) or non-target (0).

    A loop is labeled as target if it's near the removed part bounding box.

    Returns list of dicts with loop info and label.
    """
    bbox_min, bbox_max = compute_bbox(removed_part_mesh)
    bbox_min_exp, bbox_max_exp = expand_bbox(bbox_min, bbox_max, margin)

    labeled = []
    for i, loop in enumerate(loops):
        loop_verts = mesh.vertices[loop]
        is_target = loop_near_bbox(loop_verts, bbox_min_exp, bbox_max_exp, threshold)
        labeled.append({
            "loop_index": i,
            "n_vertices": len(loop),
            "is_target": int(is_target),
        })

    return labeled
