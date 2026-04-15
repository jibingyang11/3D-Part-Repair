"""Triangle quality metrics for newly added patch faces."""

import numpy as np
from typing import Dict

from ..geometry.quality import compute_face_qualities


def compute_quality_metrics(repair_result: Dict) -> Dict[str, float]:
    """Compute triangle quality metrics for the repair patch.

    Metrics:
        - avg_new_face_quality: mean quality of new faces
        - min_new_face_quality: minimum quality of new faces
        - std_new_face_quality: std deviation of quality
    """
    new_faces = repair_result.get("new_faces", np.empty((0, 3), dtype=int))
    repaired_mesh = repair_result.get("repaired_mesh")

    if repaired_mesh is None or len(new_faces) == 0:
        return {
            "avg_new_face_quality": 0.0,
            "min_new_face_quality": 0.0,
            "std_new_face_quality": 0.0,
        }

    vertices = repaired_mesh.vertices
    qualities = compute_face_qualities(vertices, new_faces)

    return {
        "avg_new_face_quality": float(np.mean(qualities)) if len(qualities) > 0 else 0.0,
        "min_new_face_quality": float(np.min(qualities)) if len(qualities) > 0 else 0.0,
        "std_new_face_quality": float(np.std(qualities)) if len(qualities) > 0 else 0.0,
    }
