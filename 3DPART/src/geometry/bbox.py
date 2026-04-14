"""Axis-aligned bounding box utilities."""

import numpy as np
import trimesh
from typing import Tuple


def compute_bbox(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the axis-aligned bounding box of a mesh.

    Returns (bbox_min, bbox_max) as 3D vectors.
    """
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    return bbox_min, bbox_max


def compute_bbox_from_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AABB from a set of points."""
    return points.min(axis=0), points.max(axis=0)


def expand_bbox(bbox_min: np.ndarray, bbox_max: np.ndarray,
                margin: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Expand bounding box by a margin (fraction of diagonal)."""
    diag = np.linalg.norm(bbox_max - bbox_min)
    delta = diag * margin
    return bbox_min - delta, bbox_max + delta


def point_in_bbox(point: np.ndarray, bbox_min: np.ndarray,
                  bbox_max: np.ndarray) -> bool:
    """Check if a 3D point is inside the bounding box."""
    return bool(np.all(point >= bbox_min) and np.all(point <= bbox_max))


def points_in_bbox(points: np.ndarray, bbox_min: np.ndarray,
                   bbox_max: np.ndarray) -> np.ndarray:
    """Check which points are inside the bounding box. Returns boolean array."""
    return np.all((points >= bbox_min) & (points <= bbox_max), axis=1)


def loop_near_bbox(loop_vertices: np.ndarray, bbox_min: np.ndarray,
                   bbox_max: np.ndarray, threshold: float = 0.1) -> bool:
    """Check if a boundary loop is near the bounding box.

    A loop is 'near' if any of its vertices is within threshold * diagonal
    of the bounding box.
    """
    diag = np.linalg.norm(bbox_max - bbox_min)
    abs_threshold = threshold * diag

    # Distance from each vertex to the nearest point on the bbox
    clamped = np.clip(loop_vertices, bbox_min, bbox_max)
    dists = np.linalg.norm(loop_vertices - clamped, axis=1)
    min_dist = dists.min()

    return min_dist <= abs_threshold


def loop_bbox_overlap_score(loop_vertices: np.ndarray, bbox_min: np.ndarray,
                            bbox_max: np.ndarray) -> float:
    """Score how much a loop overlaps with a bounding box.

    Returns fraction of loop vertices inside the expanded bbox.
    """
    inside = points_in_bbox(loop_vertices, bbox_min, bbox_max)
    return float(inside.sum()) / max(len(loop_vertices), 1)
