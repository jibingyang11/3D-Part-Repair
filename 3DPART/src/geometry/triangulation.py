"""2D Delaunay triangulation with polygon interior filtering."""

import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, List


def delaunay_2d(points_2d: np.ndarray) -> np.ndarray:
    """Perform 2D Delaunay triangulation on a set of 2D points.

    Args:
        points_2d: (N, 2) array of 2D points

    Returns: (M, 3) array of triangle vertex indices
    """
    if len(points_2d) < 3:
        return np.empty((0, 3), dtype=int)

    try:
        tri = Delaunay(points_2d)
        return tri.simplices.copy()
    except Exception:
        return np.empty((0, 3), dtype=int)


def polygon_interior_filter(points_2d: np.ndarray, triangles: np.ndarray,
                            polygon_indices: np.ndarray = None) -> np.ndarray:
    """Filter triangles to keep only those inside the boundary polygon.

    Uses the winding number / centroid-in-polygon test.

    Args:
        points_2d: (N, 2) all 2D points
        triangles: (M, 3) triangle index array
        polygon_indices: ordered indices forming the polygon boundary.
                        If None, assumes indices 0..N-1 form the polygon.

    Returns: filtered (K, 3) triangle index array
    """
    if polygon_indices is None:
        polygon_indices = np.arange(len(points_2d))

    polygon = points_2d[polygon_indices]

    filtered = []
    for tri in triangles:
        # Compute centroid of triangle
        centroid = points_2d[tri].mean(axis=0)
        if _point_in_polygon(centroid, polygon):
            filtered.append(tri)

    if not filtered:
        return np.empty((0, 3), dtype=int)
    return np.array(filtered)


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting algorithm to check if point is inside polygon."""
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i

    return inside


def ear_clipping_triangulate(polygon_2d: np.ndarray) -> np.ndarray:
    """Simple ear-clipping triangulation for a 2D polygon.

    Fallback method when Delaunay + filtering fails.

    Args:
        polygon_2d: (N, 2) ordered polygon vertices

    Returns: (N-2, 3) triangle index array
    """
    n = len(polygon_2d)
    if n < 3:
        return np.empty((0, 3), dtype=int)

    indices = list(range(n))
    triangles = []

    max_iters = n * n
    iters = 0

    while len(indices) > 2 and iters < max_iters:
        iters += 1
        found_ear = False

        for i in range(len(indices)):
            prev_idx = indices[(i - 1) % len(indices)]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % len(indices)]

            # Check if this is a convex vertex (ear candidate)
            if not _is_convex(polygon_2d[prev_idx], polygon_2d[curr_idx],
                              polygon_2d[next_idx]):
                continue

            # Check no other vertex is inside this triangle
            is_ear = True
            for j in range(len(indices)):
                if indices[j] in (prev_idx, curr_idx, next_idx):
                    continue
                if _point_in_triangle(polygon_2d[indices[j]],
                                      polygon_2d[prev_idx],
                                      polygon_2d[curr_idx],
                                      polygon_2d[next_idx]):
                    is_ear = False
                    break

            if is_ear:
                triangles.append([prev_idx, curr_idx, next_idx])
                indices.pop(i)
                found_ear = True
                break

        if not found_ear:
            break

    if not triangles:
        return np.empty((0, 3), dtype=int)
    return np.array(triangles)


def _is_convex(p0, p1, p2):
    """Check if vertex p1 is convex (left turn from p0->p1->p2)."""
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]) > 0


def _point_in_triangle(p, v0, v1, v2):
    """Check if point p is inside triangle v0, v1, v2."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)
