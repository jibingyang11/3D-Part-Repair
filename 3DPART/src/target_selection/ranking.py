"""Ranking boundary loops by various criteria."""

import numpy as np
import trimesh
from typing import List, Tuple

from .scorers import loop_proximity_score, loop_overlap_score
from ..geometry.boundary import loop_perimeter


def rank_loops(mesh: trimesh.Trimesh, loops: List[List[int]],
               removed_part_mesh: trimesh.Trimesh,
               method: str = "proximity") -> List[Tuple[int, float]]:
    """Rank boundary loops by score.

    Args:
        method: "proximity" | "overlap" | "perimeter"

    Returns: list of (loop_index, score) sorted by score descending.
    """
    scores = []
    for i, loop in enumerate(loops):
        if method == "proximity":
            s = loop_proximity_score(mesh, loop, removed_part_mesh)
        elif method == "overlap":
            s = loop_overlap_score(mesh, loop, removed_part_mesh)
        elif method == "perimeter":
            s = loop_perimeter(mesh, loop)
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        scores.append((i, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
