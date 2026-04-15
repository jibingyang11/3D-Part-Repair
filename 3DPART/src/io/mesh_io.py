"""Mesh I/O operations using trimesh - memory-safe version."""

import os
import gc
import numpy as np
import trimesh
from typing import List, Optional


def load_mesh(path: str) -> Optional[trimesh.Trimesh]:
    """Load a triangle mesh from file (.obj, .ply, .stl, etc.).

    Returns None on failure instead of raising.
    """
    try:
        # Use process=False to avoid expensive processing
        raw = trimesh.load(path, process=False)

        if isinstance(raw, trimesh.Trimesh):
            return raw

        if isinstance(raw, trimesh.Scene):
            meshes = [g for g in raw.geometry.values()
                      if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
            if not meshes:
                return None
            if len(meshes) == 1:
                return meshes[0]
            return trimesh.util.concatenate(meshes)

        # Try forcing to mesh as last resort
        return trimesh.Trimesh(vertices=raw.vertices, faces=raw.faces,
                               process=False)
    except Exception:
        return None


def load_mesh_lightweight(path: str) -> Optional[dict]:
    """Load mesh as raw numpy arrays (lower memory than trimesh objects).

    Returns dict with 'vertices' and 'faces' numpy arrays, or None.
    """
    try:
        mesh = load_mesh(path)
        if mesh is None or len(mesh.faces) == 0:
            return None
        result = {
            "vertices": np.array(mesh.vertices, dtype=np.float64),
            "faces": np.array(mesh.faces, dtype=np.int64),
        }
        del mesh
        return result
    except Exception:
        return None


def save_mesh(mesh: trimesh.Trimesh, path: str):
    """Save a triangle mesh to file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)


def merge_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Merge multiple meshes into one."""
    if not meshes:
        raise ValueError("No meshes to merge")
    if len(meshes) == 1:
        return meshes[0].copy()
    return trimesh.util.concatenate(meshes)


def merge_arrays(mesh_dicts: List[dict]) -> trimesh.Trimesh:
    """Merge multiple mesh dicts (vertices+faces arrays) into one Trimesh.

    This is more memory-efficient than merging trimesh objects.
    Each dict must have 'vertices' (N,3) and 'faces' (M,3).
    """
    if not mesh_dicts:
        raise ValueError("No meshes to merge")

    if len(mesh_dicts) == 1:
        return trimesh.Trimesh(
            vertices=mesh_dicts[0]["vertices"],
            faces=mesh_dicts[0]["faces"],
            process=False,
        )

    all_verts = []
    all_faces = []
    offset = 0

    for md in mesh_dicts:
        v = md["vertices"]
        f = md["faces"]
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += len(v)

    vertices = np.vstack(all_verts)
    faces = np.vstack(all_faces)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_and_merge_obj_files(obj_paths: List[str]) -> Optional[trimesh.Trimesh]:
    """Load multiple OBJ files and merge into one mesh.

    Memory-efficient: loads each file as raw arrays, merges with numpy,
    then builds a single Trimesh at the end.
    """
    mesh_dicts = []

    for p in obj_paths:
        if not os.path.exists(p):
            continue
        md = load_mesh_lightweight(p)
        if md is not None:
            mesh_dicts.append(md)

    if not mesh_dicts:
        return None

    try:
        result = merge_arrays(mesh_dicts)
        return result
    except Exception:
        return None
    finally:
        del mesh_dicts


def load_obj_files(obj_paths: List[str]) -> List[trimesh.Trimesh]:
    """Load multiple OBJ files as trimesh objects (legacy API)."""
    meshes = []
    for p in obj_paths:
        if os.path.exists(p):
            m = load_mesh(p)
            if m is not None and len(m.faces) > 0:
                meshes.append(m)
    return meshes


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    """Get basic mesh statistics."""
    return {
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None,
    }
