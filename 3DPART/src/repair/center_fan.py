"""Center-fan repair baseline.

For each target boundary loop, compute the geometric centroid,
add it as a new vertex, and connect it to all adjacent boundary
vertex pairs to form a fan-shaped patch.
"""

import numpy as np
import trimesh
from typing import List, Dict


def center_fan_repair(mesh: trimesh.Trimesh,
                      target_loops: List[List[int]]) -> Dict:
    """Apply center-fan repair to target boundary loops.

    Args:
        mesh: The damaged mesh (will not be modified)
        target_loops: List of target boundary loops (vertex index lists)

    Returns dict with:
        - repaired_mesh: the repaired trimesh
        - new_vertices: (K, 3) array of new vertices added
        - new_faces: (M, 3) array of new faces added
        - n_new_vertices: number of new vertices
        - n_new_faces: number of new faces
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()

    all_new_verts = []
    all_new_faces = []

    for loop in target_loops:
        if len(loop) < 3:
            continue

        # Compute centroid of boundary loop
        loop_verts = vertices[loop]
        centroid = loop_verts.mean(axis=0)

        # Add centroid as new vertex
        center_idx = len(vertices) + len(all_new_verts)
        all_new_verts.append(centroid)

        # Create fan triangles
        n = len(loop)
        for i in range(n):
            v1 = loop[i]
            v2 = loop[(i + 1) % n]
            new_face = [v1, v2, center_idx]
            all_new_faces.append(new_face)

    # Build repaired mesh
    if all_new_verts:
        new_vertices = np.array(all_new_verts)
        new_faces = np.array(all_new_faces)
        all_vertices = np.vstack([vertices, new_vertices])
        all_faces = np.array(faces + all_new_faces)
    else:
        new_vertices = np.empty((0, 3))
        new_faces = np.empty((0, 3), dtype=int)
        all_vertices = vertices
        all_faces = np.array(faces)

    repaired_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces,
                                    process=False)

    return {
        "repaired_mesh": repaired_mesh,
        "new_vertices": new_vertices,
        "new_faces": new_faces if len(all_new_faces) > 0 else np.empty((0, 3), dtype=int),
        "n_new_vertices": len(all_new_verts),
        "n_new_faces": len(all_new_faces),
    }
