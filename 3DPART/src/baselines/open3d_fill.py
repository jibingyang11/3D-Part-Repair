"""SOTA Baseline: Open3D-based hole filling.

Uses Open3D's Poisson surface reconstruction as a shape-completion
reference, and also provides an alpha-shape based hole filling.
These represent reconstruction-oriented approaches that contrast
with our local minimal-change repair.
"""

import numpy as np
import trimesh
from typing import Dict, List, Optional
from copy import deepcopy

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> "o3d.geometry.TriangleMesh":
    """Convert trimesh → Open3D mesh."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def _o3d_to_trimesh(o3d_mesh: "o3d.geometry.TriangleMesh") -> trimesh.Trimesh:
    """Convert Open3D mesh → trimesh."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def open3d_poisson_fill(mesh: trimesh.Trimesh, depth: int = 8) -> Dict:
    """Poisson surface reconstruction via Open3D.

    This is a global shape completion approach — it reconstructs the
    entire surface from point samples, serving as a reference for
    comparison with our local repair methods.

    Note: Poisson reconstruction changes the entire mesh topology,
    so patch-level metrics are less meaningful.  We report overall
    geometric distance instead.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for Poisson fill")

    o3d_mesh = _trimesh_to_o3d(mesh)

    # Sample points + normals for Poisson
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=10000)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    recon_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Crop to original bounding box (Poisson often overshoots)
    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.1, bbox.get_center())
    recon_mesh = recon_mesh.crop(bbox)

    repaired = _o3d_to_trimesh(recon_mesh)

    n_new_faces = max(0, len(repaired.faces) - len(mesh.faces))

    return {
        "repaired_mesh": repaired,
        "new_vertices": np.empty((0, 3)),
        "new_faces": np.empty((0, 3), dtype=int),
        "n_new_vertices": max(0, len(repaired.vertices) - len(mesh.vertices)),
        "n_new_faces": n_new_faces,
        "method": "poisson_reconstruction",
    }


def open3d_ball_pivoting_fill(mesh: trimesh.Trimesh) -> Dict:
    """Ball-pivoting surface reconstruction via Open3D.

    Another global reconstruction approach for comparison.
    """
    if not HAS_OPEN3D:
        raise ImportError("open3d is required for ball-pivoting fill")

    o3d_mesh = _trimesh_to_o3d(mesh)
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=10000)
    pcd.estimate_normals()

    # Estimate ball radii from point cloud distances
    dists = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(dists)
    radii = [avg_dist * f for f in [0.5, 1.0, 2.0, 4.0]]

    recon_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )

    repaired = _o3d_to_trimesh(recon_mesh)

    return {
        "repaired_mesh": repaired,
        "new_vertices": np.empty((0, 3)),
        "new_faces": np.empty((0, 3), dtype=int),
        "n_new_vertices": max(0, len(repaired.vertices) - len(mesh.vertices)),
        "n_new_faces": max(0, len(repaired.faces) - len(mesh.faces)),
        "method": "ball_pivoting",
    }
