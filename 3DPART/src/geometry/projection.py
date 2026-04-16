"""Plane fitting and 2D/3D projection for boundary loop triangulation."""

import numpy as np
from typing import Tuple


def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a local plane to 3D points using SVD/PCA.

    Returns:
        centroid: (3,) center of the points
        normal: (3,) plane normal vector
        u_axis: (3,) first in-plane axis
        v_axis: (3,) second in-plane axis
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    # SVD to find principal directions
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # The last singular vector is the normal (smallest variance direction)
    u_axis = Vt[0]  # First principal direction
    v_axis = Vt[1]  # Second principal direction
    normal = Vt[2]  # Normal (least variance)

    # Ensure right-hand coordinate system
    if np.dot(np.cross(u_axis, v_axis), normal) < 0:
        normal = -normal

    return centroid, normal, u_axis, v_axis


def project_to_2d(points_3d: np.ndarray, centroid: np.ndarray,
                  u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    """Project 3D points onto a 2D plane defined by centroid and axes.

    Returns: (N, 2) array of 2D coordinates.
    """
    centered = points_3d - centroid
    u_coords = centered @ u_axis
    v_coords = centered @ v_axis
    return np.column_stack([u_coords, v_coords])


def backproject_to_3d(points_2d: np.ndarray, centroid: np.ndarray,
                      u_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
    """Back-project 2D points to 3D using the plane basis.

    Returns: (N, 3) array of 3D coordinates.
    """
    return centroid + points_2d[:, 0:1] * u_axis + points_2d[:, 1:2] * v_axis
