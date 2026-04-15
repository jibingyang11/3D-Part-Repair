"""Boundary edge detection and boundary loop extraction from triangle meshes."""

import numpy as np
import trimesh
from collections import defaultdict
from typing import List, Tuple


def extract_boundary_edges(mesh: trimesh.Trimesh) -> List[Tuple[int, int]]:
    """Extract boundary edges from a triangle mesh.

    A boundary edge belongs to exactly one triangle face.

    Returns list of (v1, v2) vertex index pairs.
    """
    edge_face_count = defaultdict(int)

    for face in mesh.faces:
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[0], face[2]])),
        ]
        for e in edges:
            edge_face_count[e] += 1

    boundary_edges = [e for e, count in edge_face_count.items() if count == 1]
    return boundary_edges


def extract_boundary_loops(mesh: trimesh.Trimesh) -> List[List[int]]:
    """Extract ordered boundary loops from a triangle mesh.

    Returns a list of loops, where each loop is an ordered list of vertex indices.
    """
    boundary_edges = extract_boundary_edges(mesh)

    if not boundary_edges:
        return []

    # Build adjacency from boundary edges
    adj = defaultdict(list)
    for v1, v2 in boundary_edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    visited_edges = set()
    loops = []

    for start_v1, start_v2 in boundary_edges:
        edge_key = (min(start_v1, start_v2), max(start_v1, start_v2))
        if edge_key in visited_edges:
            continue

        # Trace a loop
        loop = [start_v1]
        current = start_v2
        prev = start_v1
        visited_edges.add(edge_key)

        max_iters = len(boundary_edges) + 10
        iters = 0

        while current != start_v1 and iters < max_iters:
            loop.append(current)
            neighbors = adj[current]

            # Find next unvisited neighbor (not going back)
            next_v = None
            for n in neighbors:
                ek = (min(current, n), max(current, n))
                if ek not in visited_edges:
                    next_v = n
                    visited_edges.add(ek)
                    break

            if next_v is None:
                break

            prev = current
            current = next_v
            iters += 1

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def loop_perimeter(mesh: trimesh.Trimesh, loop: List[int]) -> float:
    """Compute the perimeter (total edge length) of a boundary loop."""
    verts = mesh.vertices
    total = 0.0
    n = len(loop)
    for i in range(n):
        v1 = verts[loop[i]]
        v2 = verts[loop[(i + 1) % n]]
        total += np.linalg.norm(v2 - v1)
    return total


def loop_centroid(mesh: trimesh.Trimesh, loop: List[int]) -> np.ndarray:
    """Compute the centroid of a boundary loop's vertices."""
    return mesh.vertices[loop].mean(axis=0)


def largest_loop(loops: List[List[int]], mesh: trimesh.Trimesh) -> List[int]:
    """Return the largest boundary loop by perimeter."""
    if not loops:
        return []
    perimeters = [loop_perimeter(mesh, l) for l in loops]
    return loops[int(np.argmax(perimeters))]
