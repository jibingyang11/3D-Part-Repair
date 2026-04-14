"""Run repair experiments on the full dataset — all methods."""

import os
import gc
import json
import traceback
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm

from .run_single import run_single_experiment
from ..data.dataset_index import DatasetIndex
from ..utils import setup_logger, save_json


# All methods that run_single may produce
ALL_METHODS = [
    "center_fan",
    "planar_removed_part_aware",
    "planar_largest_hole_only",
    "trimesh_fill_all",
    "advancing_front_rpa",
    "open3d_poisson",
]


def run_batch_experiment(
    data_dir: str,
    output_dir: str,
    sample_ids: List[str] = None,
    margin: float = 0.05,
    proximity_threshold: float = 0.1,
    save_meshes: bool = True,
    include_distance: bool = True,
    include_sota: bool = True,
) -> pd.DataFrame:
    """Run experiments on all samples."""
    logger = setup_logger("BatchExperiment", os.path.join(output_dir, "logs"))

    if sample_ids is None:
        index = DatasetIndex(data_dir)
        sample_ids = index.sample_ids

    logger.info(f"Running on {len(sample_ids)} samples "
                f"(SOTA={include_sota}, distance={include_distance})")

    mesh_output_dir = os.path.join(output_dir, "repaired_meshes") if save_meshes else None
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    all_results = []
    failed = []

    for i, sid in enumerate(tqdm(sample_ids, desc="Running experiments")):
        try:
            result = run_single_experiment(
                sample_id=sid, data_dir=data_dir,
                output_dir=mesh_output_dir, margin=margin,
                proximity_threshold=proximity_threshold,
                save_meshes=save_meshes,
                include_distance=include_distance,
                include_sota=include_sota,
            )

            if "error" in result and isinstance(result.get("error"), str):
                logger.warning(f"Sample {sid}: {result['error']}")
                failed.append(sid)
                continue

            # Flatten into a row
            row = {
                "sample_id": sid,
                "n_boundary_loops": result.get("n_boundary_loops", 0),
                "n_target_loops_rpa": result.get("n_target_loops_rpa", 0),
            }

            for method in ALL_METHODS:
                metrics = result.get(method, {})
                if isinstance(metrics, dict) and "error" not in metrics:
                    for mk, mv in metrics.items():
                        row[f"{method}/{mk}"] = mv

            all_results.append(row)

        except Exception as e:
            logger.error(f"Sample {sid} failed: {e}\n{traceback.format_exc()}")
            failed.append(sid)

        if (i + 1) % 20 == 0:
            gc.collect()

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(metrics_dir, "all_results.csv"), index=False)
    save_json({"failed": failed, "n_success": len(all_results),
               "n_failed": len(failed)},
              os.path.join(metrics_dir, "run_summary.json"))
    save_json(all_results, os.path.join(metrics_dir, "all_results.json"))

    logger.info(f"Done: {len(all_results)} success, {len(failed)} failed")
    return df
