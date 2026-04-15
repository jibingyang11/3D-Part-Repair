"""SOTA Baseline: trimesh built-in hole filling (Liepa-style).

trimesh.repair.fill_holes() wraps a classic approach: detect boundary
loops, triangulate each hole with a fan/ear-clip, and optionally
refine.  We treat it as a representative of "traditional generic hole
filling" to contrast with our removed-part-aware strategy.
"""

import numpy as np
import trimesh
from typing import Dict, List
from copy import deepcopy


def trimesh_fill_all_holes(mesh: trimesh.Trimesh) -> Dict:
    """Fill ALL holes using trimesh's built-in repair.

    This is the naive 'fill everything' approach with no target awareness.
    """
    repaired = deepcopy(mesh)
    n_faces_before = len(repaired.faces)

    trimesh.repair.fill_holes(repaired)
    trimesh.repair.fix_normals(repaired)

    n_faces_after = len(repaired.faces)
    n_new = n_faces_after - n_faces_before

    # Identify new faces
    if n_new > 0:
        new_faces = repaired.faces[n_faces_before:]
    else:
        new_faces = np.empty((0, 3), dtype=int)

    return {
        "repaired_mesh": repaired,
        "new_vertices": np.empty((0, 3)),
        "new_faces": new_faces,
        "n_new_vertices": 0,
        "n_new_faces": int(n_new),
    }


def trimesh_fill_target_loops(mesh: trimesh.Trimesh,
                              target_loops: List[List[int]]) -> Dict:
    """Fill only target boundary loops using trimesh triangulation.

    For each target loop we build a small sub-problem and use trimesh
    to triangulate it, then graft the patch onto the original mesh.
    """
    from ..geometry.projection import fit_plane, project_to_2d
    from ..geometry.triangulation import delaunay_2d, polygon_interior_filter

    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()
    all_new_faces = []

    for loop in target_loops:
        if len(loop) < 3:
            continue

        loop_verts = vertices[loop]

        # Use trimesh's own ear-clipping on the projected polygon
        centroid, normal, u_axis, v_axis = fit_plane(loop_verts)
        pts2d = project_to_2d(loop_verts, centroid, u_axis, v_axis)

        try:
            # trimesh ear clipping
            local_faces = trimesh.triangles.earclip(pts2d)
            for tri in local_faces:
                all_new_faces.append([loop[tri[0]], loop[tri[1]], loop[tri[2]]])
        except Exception:
            # Fallback to Delaunay
            tris = delaunay_2d(pts2d)
            if len(tris) > 0:
                tris = polygon_interior_filter(pts2d, tris, np.arange(len(loop)))
            for tri in tris:
                all_new_faces.append([loop[tri[0]], loop[tri[1]], loop[tri[2]]])

    new_faces = np.array(all_new_faces) if all_new_faces else np.empty((0, 3), dtype=int)
    all_faces_arr = np.array(faces + all_new_faces) if all_new_faces else np.array(faces)

    repaired = trimesh.Trimesh(vertices=vertices, faces=all_faces_arr, process=False)

    return {
        "repaired_mesh": repaired,
        "new_vertices": np.empty((0, 3)),
        "new_faces": new_faces,
        "n_new_vertices": 0,
        "n_new_faces": len(all_new_faces),
    }
