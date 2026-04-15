"""Patch complexity metrics.

Counts the number of new vertices and faces introduced during repair.
"""

from typing import Dict
import numpy as np


def compute_complexity_metrics(repair_result: Dict) -> Dict[str, float]:
    """Compute complexity metrics from a repair result.

    Metrics:
        - n_new_vertices: number of new vertices added
        - n_new_faces: number of new faces added
    """
    return {
        "n_new_vertices": float(repair_result.get("n_new_vertices", 0)),
        "n_new_faces": float(repair_result.get("n_new_faces", 0)),
    }
