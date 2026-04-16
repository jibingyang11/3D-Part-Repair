"""Minimal area patch repair method.

An alternative repair that tries to minimize patch surface area.
Uses ear-clipping on the projected boundary loop.
"""

import numpy as np
import trimesh
from typing import List, Dict

from ..geometry.projection import fit_plane, project_to_2d
from ..geometry.triangulation import ear_clipping_triangulate


def minimal_area_repair(mesh: trimesh.Trimesh,
                        target_loops: List[List[int]]) -> Dict:
    """Apply minimal-area ear-clipping repair.

    Similar to planar but uses ear-clipping which tends to
    produce more compact patches.
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()
    all_new_faces = []

    for loop in target_loops:
        if len(loop) < 3:
            continue

        loop_verts = vertices[loop]
        centroid, normal, u_axis, v_axis = fit_plane(loop_verts)
        points_2d = project_to_2d(loop_verts, centroid, u_axis, v_axis)

        triangles = ear_clipping_triangulate(points_2d)

        for tri in triangles:
            global_face = [loop[tri[0]], loop[tri[1]], loop[tri[2]]]
            all_new_faces.append(global_face)

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
