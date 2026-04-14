"""Largest-hole-only baseline.

Uses the same planar triangulation method but selects only
the largest boundary loop as the repair target.
"""

import trimesh
from typing import Dict

from ..geometry.boundary import extract_boundary_loops
from ..target_selection.selectors import select_largest_loop
from ..repair.planar_patch import planar_triangulation_repair
from ..repair.center_fan import center_fan_repair


def largest_hole_baseline(mesh: trimesh.Trimesh,
                          repair_method: str = "planar") -> Dict:
    """Run largest-hole-only baseline."""
    loops = extract_boundary_loops(mesh)
    target_loops = select_largest_loop(mesh, loops)

    if repair_method == "center_fan":
        return center_fan_repair(mesh, target_loops)
    else:
        return planar_triangulation_repair(mesh, target_loops)
