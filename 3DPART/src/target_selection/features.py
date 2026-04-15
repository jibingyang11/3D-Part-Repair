"""Feature extraction for boundary loops."""

import numpy as np
import trimesh
from typing import List, Dict

from ..geometry.boundary import loop_perimeter, loop_centroid
from ..geometry.bbox import compute_bbox


def extract_loop_features(mesh: trimesh.Trimesh, loop: List[int],
                          removed_part_mesh: trimesh.Trimesh = None) -> Dict[str, float]:
    """Extract geometric features from a boundary loop.

    Features:
        - n_vertices: number of vertices in the loop
        - perimeter: total edge length
        - centroid_x/y/z: loop centroid coordinates
        - bbox_volume: volume of loop bounding box
        - planarity: how planar the loop is (0=very planar, higher=less)
        - dist_to_removed: distance to removed part center (if provided)
    """
    verts = mesh.vertices[loop]

    features = {
        "n_vertices": float(len(loop)),
        "perimeter": loop_perimeter(mesh, loop),
    }

    centroid = verts.mean(axis=0)
    features["centroid_x"] = float(centroid[0])
    features["centroid_y"] = float(centroid[1])
    features["centroid_z"] = float(centroid[2])

    # Bounding box volume
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    features["bbox_volume"] = float(np.prod(np.maximum(bbox_size, 1e-10)))

    # Planarity: variance along the normal direction (SVD)
    centered = verts - centroid
    if len(centered) >= 3:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        features["planarity"] = float(S[-1] / (S[0] + 1e-10))
    else:
        features["planarity"] = 0.0

    # Distance to removed part
    if removed_part_mesh is not None:
        rem_bbox_min, rem_bbox_max = compute_bbox(removed_part_mesh)
        rem_center = (rem_bbox_min + rem_bbox_max) / 2.0
        features["dist_to_removed"] = float(np.linalg.norm(centroid - rem_center))
    else:
        features["dist_to_removed"] = -1.0

    return features
