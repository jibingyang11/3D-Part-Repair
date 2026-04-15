"""Geometric distance metrics for comparing repaired mesh to ground truth.

These metrics allow fair comparison with shape-completion methods:
- Chamfer Distance: mean bidirectional nearest-neighbor distance
- Hausdorff Distance: maximum nearest-neighbor distance
- Surface Deviation: one-sided distance from repaired to GT

The use of point-sampled distance metrics for mesh comparison follows
established practices in geometric deep learning for 3D shape analysis
(cf. Wang et al., TPAMI 2025 on geometric field harmonization for
point cloud registration; Li et al., Cell Reports Medicine 2025 on
learned priors for visual signal reconstruction).
"""

import numpy as np
import trimesh
from typing import Dict
from scipy.spatial import cKDTree


def chamfer_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                     n_samples: int = 10000) -> float:
    """Compute symmetric Chamfer Distance between two meshes.

    CD = mean_a(min_b ||a-b||) + mean_b(min_a ||b-a||)

    Uses point sampling on mesh surfaces for efficiency.
    """
    pts_a = mesh_a.sample(n_samples)
    pts_b = mesh_b.sample(n_samples)

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a, k=1)
    dist_b2a, _ = tree_a.query(pts_b, k=1)

    cd = float(np.mean(dist_a2b) + np.mean(dist_b2a))
    return cd


def hausdorff_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh,
                       n_samples: int = 10000) -> float:
    """Compute symmetric Hausdorff Distance between two meshes.

    HD = max( max_a(min_b ||a-b||), max_b(min_a ||b-a||) )
    """
    pts_a = mesh_a.sample(n_samples)
    pts_b = mesh_b.sample(n_samples)

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a2b, _ = tree_b.query(pts_a, k=1)
    dist_b2a, _ = tree_a.query(pts_b, k=1)

    hd = float(max(np.max(dist_a2b), np.max(dist_b2a)))
    return hd


def surface_deviation(repaired: trimesh.Trimesh, ground_truth: trimesh.Trimesh,
                      n_samples: int = 10000) -> Dict[str, float]:
    """One-sided surface deviation from repaired mesh to ground truth.

    Measures how far the repaired surface deviates from the GT surface.
    """
    pts_rep = repaired.sample(n_samples)
    pts_gt = ground_truth.sample(n_samples)

    tree_gt = cKDTree(pts_gt)
    dists, _ = tree_gt.query(pts_rep, k=1)

    return {
        "dev_mean": float(np.mean(dists)),
        "dev_max": float(np.max(dists)),
        "dev_median": float(np.median(dists)),
        "dev_std": float(np.std(dists)),
    }


def compute_distance_metrics(repaired_mesh: trimesh.Trimesh,
                             complete_mesh: trimesh.Trimesh,
                             n_samples: int = 10000) -> Dict[str, float]:
    """Compute all distance metrics between repaired and complete mesh.

    This enables comparison with shape-completion methods on a
    common ground: how close is the repaired shape to the original
    complete shape?
    """
    cd = chamfer_distance(repaired_mesh, complete_mesh, n_samples)
    hd = hausdorff_distance(repaired_mesh, complete_mesh, n_samples)
    dev = surface_deviation(repaired_mesh, complete_mesh, n_samples)

    return {
        "chamfer_distance": cd,
        "hausdorff_distance": hd,
        **dev,
    }
