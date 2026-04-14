"""Run repair experiment on a single sample — all methods."""

import os
import gc
import numpy as np
import trimesh
from typing import Dict

from ..data.sample_loader import SampleLoader
from ..geometry.boundary import extract_boundary_loops
from ..target_selection.selectors import select_target_loops_by_bbox, select_largest_loop
from ..repair.center_fan import center_fan_repair
from ..repair.planar_patch import planar_triangulation_repair
from ..baselines.trimesh_fill import trimesh_fill_all_holes, trimesh_fill_target_loops
from ..baselines.advancing_front import advancing_front_repair
from ..evaluation.evaluator import Evaluator
from ..io.mesh_io import save_mesh


def run_single_experiment(
    sample_id: str,
    data_dir: str,
    output_dir: str = None,
    margin: float = 0.05,
    proximity_threshold: float = 0.1,
    save_meshes: bool = True,
    include_distance: bool = True,
    include_sota: bool = True,
) -> Dict:
    """Run all repair methods on a single sample.

    Methods:
      1. center_fan                 (ours, RPA target)
      2. planar_removed_part_aware  (ours, RPA target)
      3. planar_largest_hole_only   (ablation)
      4. trimesh_fill_all           (SOTA: fill every hole)
      5. advancing_front_rpa        (SOTA: adv-front + RPA target)
      6. open3d_poisson             (SOTA: global Poisson recon)
    """
    loader = SampleLoader(data_dir)
    sample = loader.load(sample_id)

    damaged_mesh = sample["damaged_mesh"]
    removed_part_mesh = sample["removed_part_mesh"]
    complete_mesh = sample["complete_mesh"] if include_distance else None

    evaluator = Evaluator(margin=margin, proximity_threshold=proximity_threshold)

    loops = extract_boundary_loops(damaged_mesh)
    if not loops:
        return {"error": "No boundary loops found", "sample_id": sample_id}

    # Target loops for RPA
    target_loops_rpa = select_target_loops_by_bbox(
        damaged_mesh, loops, removed_part_mesh, margin, proximity_threshold
    )

    # Helper to evaluate one method
    def _eval(result, target_loops):
        return evaluator.evaluate(
            damaged_mesh, result["repaired_mesh"], removed_part_mesh,
            result, target_loops, complete_mesh
        )

    results = {
        "sample_id": sample_id,
        "n_boundary_loops": len(loops),
        "n_target_loops_rpa": len(target_loops_rpa),
    }

    # --- 1. Center-fan (ours) ---
    res = center_fan_repair(damaged_mesh, target_loops_rpa)
    results["center_fan"] = _eval(res, target_loops_rpa)
    if save_meshes and output_dir:
        _save(res, output_dir, sample_id, "center_fan")
    del res

    # --- 2. Planar + RPA (ours) ---
    res = planar_triangulation_repair(damaged_mesh, target_loops_rpa)
    results["planar_removed_part_aware"] = _eval(res, target_loops_rpa)
    if save_meshes and output_dir:
        _save(res, output_dir, sample_id, "planar_rpa")
    del res

    # --- 3. Planar + largest-hole-only (ablation) ---
    target_loops_lh = select_largest_loop(damaged_mesh, loops)
    res = planar_triangulation_repair(damaged_mesh, target_loops_lh)
    results["planar_largest_hole_only"] = _eval(res, target_loops_rpa)
    if save_meshes and output_dir:
        _save(res, output_dir, sample_id, "planar_lh")
    del res

    if include_sota:
        # --- 4. Trimesh fill-all (SOTA) ---
        try:
            res = trimesh_fill_all_holes(damaged_mesh)
            results["trimesh_fill_all"] = _eval(res, target_loops_rpa)
            if save_meshes and output_dir:
                _save(res, output_dir, sample_id, "trimesh_fill_all")
            del res
        except Exception as e:
            results["trimesh_fill_all"] = {"error": str(e)}

        # --- 5. Advancing front + RPA (SOTA) ---
        try:
            res = advancing_front_repair(damaged_mesh, target_loops_rpa)
            results["advancing_front_rpa"] = _eval(res, target_loops_rpa)
            if save_meshes and output_dir:
                _save(res, output_dir, sample_id, "advancing_front_rpa")
            del res
        except Exception as e:
            results["advancing_front_rpa"] = {"error": str(e)}

        # --- 6. Open3D Poisson (SOTA, optional) ---
        try:
            from ..baselines.open3d_fill import open3d_poisson_fill
            res = open3d_poisson_fill(damaged_mesh)
            results["open3d_poisson"] = _eval(res, target_loops_rpa)
            del res
        except (ImportError, Exception) as e:
            results["open3d_poisson"] = {"error": str(e)}

    gc.collect()
    return results


def _save(result, output_dir, sample_id, method_name):
    mesh_dir = os.path.join(output_dir, sample_id)
    os.makedirs(mesh_dir, exist_ok=True)
    save_mesh(result["repaired_mesh"],
              os.path.join(mesh_dir, f"repaired_{method_name}.obj"))
