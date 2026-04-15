"""Triangle quality metrics for evaluating repair patches."""

import numpy as np
from typing import List


def triangle_quality(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute quality of a single triangle using the normalized ratio metric.

    Quality = 4 * sqrt(3) * area / (a^2 + b^2 + c^2)
    where a, b, c are edge lengths.

    Returns a value in [0, 1] where 1 = equilateral triangle.
    """
    a = np.linalg.norm(v1 - v0)
    b = np.linalg.norm(v2 - v1)
    c = np.linalg.norm(v0 - v2)

    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)

    if area_sq <= 0:
        return 0.0

    area = np.sqrt(max(area_sq, 0.0))
    denom = a * a + b * b + c * c

    if denom < 1e-15:
        return 0.0

    q = 4.0 * np.sqrt(3.0) * area / denom
    return float(np.clip(q, 0.0, 1.0))


def compute_face_qualities(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute quality for each face in a mesh.

    Args:
        vertices: (N, 3) vertex array
        faces: (M, 3) face index array

    Returns: (M,) quality array
    """
    qualities = np.zeros(len(faces))
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        qualities[i] = triangle_quality(v0, v1, v2)
    return qualities


def mean_triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute mean triangle quality over all faces."""
    if len(faces) == 0:
        return 0.0
    qualities = compute_face_qualities(vertices, faces)
    return float(np.mean(qualities))


def min_triangle_quality(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute minimum triangle quality over all faces."""
    if len(faces) == 0:
        return 0.0
    qualities = compute_face_qualities(vertices, faces)
    return float(np.min(qualities))
