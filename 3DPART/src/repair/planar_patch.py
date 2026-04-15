"""Planar triangulation repair baseline.

For each target boundary loop:
1. Fit a local 2D plane using SVD/PCA
2. Project boundary vertices onto the plane
3. Perform 2D Delaunay triangulation
4. Apply polygon interior filtering
5. Map triangles back to 3D
"""

import numpy as np
import trimesh
from typing import List, Dict

from ..geometry.projection import fit_plane, project_to_2d
from ..geometry.triangulation import delaunay_2d, polygon_interior_filter, ear_clipping_triangulate


def planar_triangulation_repair(mesh: trimesh.Trimesh,
                                target_loops: List[List[int]]) -> Dict:
    """Apply planar triangulation repair to target boundary loops.

    This method does NOT introduce new vertices.

    Args:
        mesh: The damaged mesh (will not be modified)
        target_loops: List of target boundary loops (vertex index lists)

    Returns dict with:
        - repaired_mesh: the repaired trimesh
        - new_vertices: empty (no new vertices added)
        - new_faces: (M, 3) array of new faces added
        - n_new_vertices: 0
        - n_new_faces: number of new faces
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()

    all_new_faces = []

    for loop in target_loops:
        if len(loop) < 3:
            continue

        loop_verts = vertices[loop]

        # Fit plane and project
        centroid, normal, u_axis, v_axis = fit_plane(loop_verts)
        points_2d = project_to_2d(loop_verts, centroid, u_axis, v_axis)

        # Create local index mapping
        n_loop = len(loop)
        local_indices = np.arange(n_loop)

        # Try Delaunay + polygon filtering first
        triangles = delaunay_2d(points_2d)

        if len(triangles) > 0:
            triangles = polygon_interior_filter(points_2d, triangles, local_indices)

        # Fallback to ear clipping if Delaunay fails
        if len(triangles) == 0:
            triangles = ear_clipping_triangulate(points_2d)

        # Map local indices back to global vertex indices
        for tri in triangles:
            global_face = [loop[tri[0]], loop[tri[1]], loop[tri[2]]]
            all_new_faces.append(global_face)

    # Build repaired mesh
    new_faces = np.array(all_new_faces) if all_new_faces else np.empty((0, 3), dtype=int)
    all_faces = np.array(faces + all_new_faces) if all_new_faces else np.array(faces)

    repaired_mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=False)

    return {
        "repaired_mesh": repaired_mesh,
        "new_vertices": np.empty((0, 3)),
        "new_faces": new_faces,
        "n_new_vertices": 0,
        "n_new_faces": len(all_new_faces),
    }
