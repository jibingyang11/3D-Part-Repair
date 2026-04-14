"""Unified evaluator that combines all evaluation dimensions.

Five dimensions:
1. Closure      – is the target hole sealed?
2. Complexity   – how many new elements were added?
3. Quality      – how good are the new triangles?
4. Locality     – does the patch stay near the removal zone?
5. Distance     – geometric distance to the original complete mesh
"""

import numpy as np
import trimesh
from typing import List, Dict, Optional

from .metrics_closure import compute_closure_metrics
from .metrics_complexity import compute_complexity_metrics
from .metrics_quality import compute_quality_metrics
from .metrics_locality import compute_locality_metrics
from .metrics_distance import compute_distance_metrics


class Evaluator:
    """Evaluate repair results across all dimensions."""

    def __init__(self, margin: float = 0.05, proximity_threshold: float = 0.1,
                 distance_samples: int = 10000):
        self.margin = margin
        self.proximity_threshold = proximity_threshold
        self.distance_samples = distance_samples

    def evaluate(
        self,
        damaged_mesh: trimesh.Trimesh,
        repaired_mesh: trimesh.Trimesh,
        removed_part_mesh: trimesh.Trimesh,
        repair_result: Dict,
        target_loops_before: List[List[int]],
        complete_mesh: trimesh.Trimesh = None,
    ) -> Dict[str, float]:
        """Run all evaluations on a repair result.

        Args:
            complete_mesh: if provided, also compute distance metrics.

        Returns a flat dict with all metrics.
        """
        if "repaired_mesh" not in repair_result:
            repair_result["repaired_mesh"] = repaired_mesh

        # 1. Closure
        closure = compute_closure_metrics(
            damaged_mesh, repaired_mesh, removed_part_mesh,
            target_loops_before, self.margin, self.proximity_threshold
        )

        # 2. Complexity
        complexity = compute_complexity_metrics(repair_result)

        # 3. Quality
        quality = compute_quality_metrics(repair_result)

        # 4. Locality
        locality = compute_locality_metrics(
            repair_result, removed_part_mesh, self.margin
        )

        # 5. Distance (optional, needs complete mesh)
        distance = {}
        if complete_mesh is not None:
            try:
                distance = compute_distance_metrics(
                    repaired_mesh, complete_mesh, self.distance_samples
                )
            except Exception:
                distance = {
                    "chamfer_distance": np.nan,
                    "hausdorff_distance": np.nan,
                    "dev_mean": np.nan,
                    "dev_max": np.nan,
                }

        # Merge all metrics
        result = {}
        result.update(closure)
        result.update(complexity)
        result.update(quality)
        result.update(locality)
        result.update(distance)

        return result

    def evaluate_batch(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics over a batch of samples."""
        if not results:
            return {}

        keys = results[0].keys()
        agg = {}

        for key in keys:
            values = [r[key] for r in results if key in r and not np.isnan(r.get(key, np.nan))]
            if values:
                agg[f"{key}_mean"] = float(np.mean(values))
                agg[f"{key}_std"] = float(np.std(values))
                agg[f"{key}_median"] = float(np.median(values))

        return agg
