"""SOTA Baseline: Advancing-front hole filling.

Implements a simplified advancing-front triangulation inspired by
Liepa (2003) "Filling holes in meshes". The algorithm progressively
closes a boundary loop by connecting the vertex pair that forms the
best triangle (smallest angle / best quality), advancing the front
inward until the hole is closed.

This represents the classic mesh-repair literature approach.
"""

import numpy as np
import trimesh
from typing import List, Dict

from ..geometry.quality import triangle_quality


def advancing_front_repair(mesh: trimesh.Trimesh,
                           target_loops: List[List[int]]) -> Dict:
    """Advancing-front hole filling on target boundary loops.

    For each loop, iteratively pick the boundary vertex with the
    smallest interior angle and create a triangle, advancing the
    front until the loop is fully closed.
    """
    vertices = mesh.vertices.copy()
    faces = mesh.faces.tolist()
    all_new_faces = []
    all_new_verts = []

    for loop in target_loops:
        if len(loop) < 3:
            continue

        new_f, new_v = _fill_loop_advancing(vertices, loop, len(vertices) + len(all_new_verts))
        all_new_faces.extend(new_f)
        all_new_verts.extend(new_v)

    if all_new_verts:
        new_vertices = np.array(all_new_verts)
        all_vertices = np.vstack([vertices, new_vertices])
    else:
        new_vertices = np.empty((0, 3))
        all_vertices = vertices

    new_faces_arr = np.array(all_new_faces) if all_new_faces else np.empty((0, 3), dtype=int)
    all_faces_arr = np.array(faces + all_new_faces) if all_new_faces else np.array(faces)

    repaired = trimesh.Trimesh(vertices=all_vertices, faces=all_faces_arr, process=False)

    return {
        "repaired_mesh": repaired,
        "new_vertices": new_vertices,
        "new_faces": new_faces_arr,
        "n_new_vertices": len(all_new_verts),
        "n_new_faces": len(all_new_faces),
    }


def _fill_loop_advancing(vertices: np.ndarray, loop: List[int],
                         next_vert_idx: int) -> tuple:
    """Fill a single boundary loop using advancing-front.

    Returns (new_faces_list, new_verts_list).
    """
    new_faces = []
    new_verts = []
    active = list(loop)  # Working copy

    max_iters = len(loop) * 3
    iters = 0

    while len(active) > 2 and iters < max_iters:
        iters += 1
        n = len(active)

        if n == 3:
            # Last triangle
            new_faces.append([active[0], active[1], active[2]])
            break

        # Find the vertex with the smallest interior angle
        best_idx = -1
        best_angle = float('inf')

        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n

            v_prev = vertices[active[prev_i]]
            v_curr = vertices[active[i]]
            v_next = vertices[active[next_i]]

            e1 = v_prev - v_curr
            e2 = v_next - v_curr

            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            if norm1 < 1e-12 or norm2 < 1e-12:
                continue

            cos_angle = np.clip(np.dot(e1, e2) / (norm1 * norm2), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            if angle < best_angle:
                best_angle = angle
                best_idx = i

        if best_idx < 0:
            break

        # Create triangle at the best vertex
        prev_i = (best_idx - 1) % n
        next_i = (best_idx + 1) % n

        v_prev = active[prev_i]
        v_curr = active[best_idx]
        v_next = active[next_i]

        # Check angle threshold to decide strategy
        if best_angle < np.pi * 0.75:
            # Small angle: directly connect prev-curr-next
            new_faces.append([v_prev, v_curr, v_next])
            active.pop(best_idx)
        else:
            # Large angle: insert a midpoint and create two triangles
            p_mid = (vertices[v_prev] + vertices[v_next]) / 2.0
            mid_idx = next_vert_idx + len(new_verts)
            new_verts.append(p_mid)

            # Temporarily extend the vertices array for quality checks
            new_faces.append([v_prev, v_curr, mid_idx])
            new_faces.append([v_curr, v_next, mid_idx])
            # Replace curr with mid in the active front
            active[best_idx] = mid_idx

    return new_faces, new_verts
