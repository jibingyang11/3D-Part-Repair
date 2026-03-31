import json
import numpy as np
import open3d as o3d
from pathlib import Path
from collections import defaultdict
from scipy.spatial import Delaunay
from matplotlib.path import Path as MplPath
import pandas as pd
def load_and_merge_obj_list(obj_paths):
    vertices_all = []
    triangles_all = []
    v_offset = 0

    for p in obj_paths:
        mesh = o3d.io.read_triangle_mesh(str(p))
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.triangles)

        if len(v) == 0 or len(f) == 0:
            continue

        vertices_all.append(v)
        triangles_all.append(f + v_offset)
        v_offset += len(v)

    if len(vertices_all) == 0:
        return None

    merged = o3d.geometry.TriangleMesh()
    merged.vertices = o3d.utility.Vector3dVector(np.vstack(vertices_all))
    merged.triangles = o3d.utility.Vector3iVector(np.vstack(triangles_all))
    merged.compute_triangle_normals()
    merged.compute_vertex_normals()
    return merged


def loop_center(vertices, loop):
    pts = vertices[[int(v) for v in loop]]
    return pts.mean(axis=0)


# ---------- boundary edge / loop ----------

def get_boundary_edges_o3d(mesh):
    triangles = np.asarray(mesh.triangles)
    edge_count = defaultdict(int)

    for tri in triangles:
        a, b, c = map(int, tri)
        edges = [
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((c, a))),
        ]
        for e in edges:
            edge_count[e] += 1

    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    return boundary_edges


def boundary_edges_to_loops(boundary_edges):
    adj = defaultdict(list)
    for u, v in boundary_edges:
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)

    visited_edges = set()
    loops = []

    def edge_key(a, b):
        return tuple(sorted((int(a), int(b))))

    for u, v in boundary_edges:
        u, v = int(u), int(v)
        if edge_key(u, v) in visited_edges:
            continue

        loop = [u]
        prev = None
        curr = u

        while True:
            neighbors = adj[curr]
            next_vertex = None

            for nb in neighbors:
                if nb == prev:
                    continue
                if edge_key(curr, nb) not in visited_edges:
                    next_vertex = nb
                    break

            if next_vertex is None:
                break

            visited_edges.add(edge_key(curr, next_vertex))
            prev, curr = curr, next_vertex

            if curr == loop[0]:
                break
            loop.append(curr)

        if len(loop) > 2:
            loops.append(loop)

    return loops


# ---------- 选需要修补的 loops ----------

def choose_nearby_loops_by_removed_part(sample_dir, loops_before, M_d, margin=0.02, min_len=6):
    sample_dir = Path(sample_dir)

    with open(sample_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    source_dir = Path(meta["source_dir"])
    removed_obj_files = meta["removed_obj_files"]
    removed_obj_paths = [source_dir / "objs" / name for name in removed_obj_files]
    removed_mesh = load_and_merge_obj_list(removed_obj_paths)

    if removed_mesh is None:
        return [max(loops_before, key=len)]

    removed_v = np.asarray(removed_mesh.vertices)
    bb_min = removed_v.min(axis=0) - margin
    bb_max = removed_v.max(axis=0) + margin

    V_d = np.asarray(M_d.vertices)
    selected = []

    for loop in loops_before:
        if len(loop) < min_len:
            continue

        pts = V_d[[int(v) for v in loop]]
        inside = np.all((pts >= bb_min) & (pts <= bb_max), axis=1)

        if inside.any():
            selected.append(loop)

    if len(selected) == 0:
        return [max(loops_before, key=len)]

    return selected
def choose_nearby_loops_by_removed_part_from_mesh(loops, mesh, removed_mesh, margin=0.02, min_len=6):
    if removed_mesh is None or len(loops) == 0:
        return []

    removed_v = np.asarray(removed_mesh.vertices)
    bb_min = removed_v.min(axis=0) - margin
    bb_max = removed_v.max(axis=0) + margin

    V = np.asarray(mesh.vertices)
    selected = []

    for loop in loops:
        if len(loop) < min_len:
            continue

        pts = V[[int(v) for v in loop]]
        inside = np.all((pts >= bb_min) & (pts <= bb_max), axis=1)

        if inside.any():
            selected.append(loop)

    return selected
def compute_loop_perimeter(vertices, loop):
    loop = [int(v) for v in loop]
    if len(loop) < 2:
        return 0.0

    total = 0.0
    for i in range(len(loop)):
        a = loop[i]
        b = loop[(i + 1) % len(loop)]
        total += np.linalg.norm(vertices[a] - vertices[b])
    return float(total)


def compute_total_loop_perimeter(vertices, loops):
    return float(sum(compute_loop_perimeter(vertices, loop) for loop in loops))
def count_new_vertices(M_d, M_r):
    return int(len(np.asarray(M_r.vertices)) - len(np.asarray(M_d.vertices)))
def show_mesh(mesh, window_name="mesh", color=None, back_face=True):
    mesh_show = o3d.geometry.TriangleMesh(mesh)
    if color is not None:
        mesh_show.paint_uniform_color(color)
    mesh_show.compute_triangle_normals()
    mesh_show.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh_show],
        window_name=window_name,
        mesh_show_back_face=back_face
    )
def triangulate_multiple_loops_on_plane(mesh, loops):
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()

    all_patch_faces = []

    for main_loop in loops:
        loop_vids = [int(v) for v in main_loop]
        loop_pts = V[loop_vids]

        if len(loop_pts) < 3:
            continue

        center = loop_pts.mean(axis=0)
        X = loop_pts - center

        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        e1, e2 = Vt[0], Vt[1]

        pts_2d = np.stack([X @ e1, X @ e2], axis=1)

        c2 = pts_2d.mean(axis=0)
        angles = np.arctan2(pts_2d[:, 1] - c2[1], pts_2d[:, 0] - c2[0])
        order = np.argsort(angles)

        pts_2d = pts_2d[order]
        loop_vids = [loop_vids[i] for i in order]

        try:
            tri = Delaunay(pts_2d)
        except Exception:
            continue

        poly = MplPath(pts_2d)

        for t in tri.simplices:
            tri_center = pts_2d[t].mean(axis=0)
            if poly.contains_point(tri_center):
                all_patch_faces.append([
                    loop_vids[t[0]],
                    loop_vids[t[1]],
                    loop_vids[t[2]],
                ])

    if len(all_patch_faces) == 0:
        return None

    F_new = np.vstack([F, np.array(all_patch_faces, dtype=np.int32)])

    M_r = o3d.geometry.TriangleMesh()
    M_r.vertices = o3d.utility.Vector3dVector(V)
    M_r.triangles = o3d.utility.Vector3iVector(F_new)
    M_r.compute_triangle_normals()
    M_r.compute_vertex_normals()
    return M_r
def repair_one_sample_planar(sample_dir):
    sample_dir = Path(sample_dir)
    M_d_path = sample_dir / "M_d.obj"

    if not M_d_path.exists():
        raise FileNotFoundError(f"Missing: {M_d_path}")

    M_d = o3d.io.read_triangle_mesh(str(M_d_path))
    M_d.compute_triangle_normals()
    M_d.compute_vertex_normals()

    boundary_edges_before = get_boundary_edges_o3d(M_d)
    loops_before = boundary_edges_to_loops(boundary_edges_before)
    
    if len(loops_before) == 0:
        return {
            "sample_id": sample_dir.name,
            "success": True,
            "reason": "no boundary loops found",
            "M_d": M_d,
            "M_r": M_d,
            "nearby_loops": [],
            "num_loops_repaired": 0,
            "total_loop_len_before": 0.0,
            "nearest_loop_len_after": 0.0,
            "improvement": 0.0,
            "num_new_vertices": 0,
            
        }

    nearby_loops = choose_nearby_loops_by_removed_part(sample_dir, loops_before, M_d)

    V_d = np.asarray(M_d.vertices)
    loop_lens_before = [compute_loop_perimeter(V_d, loop) for loop in nearby_loops]
    total_loop_len_before = float(sum(loop_lens_before))

    M_r = triangulate_multiple_loops_on_plane(M_d, nearby_loops)
    if M_r is None:
        return {
            "sample_id": sample_dir.name,
            "success": False,
            "reason": "triangulation failed"
        }

    boundary_edges_after = get_boundary_edges_o3d(M_r)
    loops_after = boundary_edges_to_loops(boundary_edges_after)

    with open(sample_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    source_dir = Path(meta["source_dir"])
    removed_obj_files = meta["removed_obj_files"]
    removed_obj_paths = [source_dir / "objs" / name for name in removed_obj_files]
    removed_mesh = load_and_merge_obj_list(removed_obj_paths)

    V_r = np.asarray(M_r.vertices)
    target_loops_after = choose_nearby_loops_by_removed_part_from_mesh(
        loops_after, M_r, removed_mesh, margin=0.02, min_len=6
    )
    nearest_loop_len_after = compute_total_loop_perimeter(V_r, target_loops_after)

    num_new_vertices = count_new_vertices(M_d, M_r)
    q = compute_added_face_quality(M_d, M_r)
    bb_min, bb_max = get_removed_part_bbox(sample_dir, margin=0.02)
    loc = compute_added_face_locality(M_d, M_r, bb_min, bb_max)
    return {
        "sample_id": sample_dir.name,
        "success": True,
        "M_d": M_d,
        "M_r": M_r,
        "nearby_loops": nearby_loops,
        "num_loops_repaired": len(nearby_loops),
        "total_loop_len_before": total_loop_len_before,
        "nearest_loop_len_after": float(nearest_loop_len_after),
        "improvement": float(total_loop_len_before - nearest_loop_len_after),
        "num_new_vertices": num_new_vertices,
        "num_added_faces": q["num_added_faces"],
        "mean_added_face_quality": q["mean_added_face_quality"],
        "min_added_face_quality": q["min_added_face_quality"],
        "num_added_faces_inside_zone": loc["num_added_faces_inside_zone"],
        "num_added_faces_outside_zone": loc["num_added_faces_outside_zone"],
        "face_locality_ratio": loc["face_locality_ratio"],
    }
def triangle_quality(v0, v1, v2, eps=1e-12):
    e0 = np.linalg.norm(v1 - v0)
    e1 = np.linalg.norm(v2 - v1)
    e2 = np.linalg.norm(v0 - v2)

    s = 0.5 * (e0 + e1 + e2)
    area_sq = max(s * (s - e0) * (s - e1) * (s - e2), 0.0)
    area = np.sqrt(area_sq)

    denom = e0 * e0 + e1 * e1 + e2 * e2 + eps
    q = 4.0 * np.sqrt(3.0) * area / denom
    return float(q)


def compute_added_face_quality(M_d, M_r):
    V = np.asarray(M_r.vertices)
    F_d = np.asarray(M_d.triangles)
    F_r = np.asarray(M_r.triangles)

    num_added = len(F_r) - len(F_d)
    if num_added <= 0:
        return {
            "num_added_faces": 0,
            "mean_added_face_quality": 0.0,
            "min_added_face_quality": 0.0,
        }

    added_faces = F_r[len(F_d):]
    qualities = []

    for f in added_faces:
        a, b, c = map(int, f)
        q = triangle_quality(V[a], V[b], V[c])
        qualities.append(q)

    qualities = np.array(qualities, dtype=np.float64)

    return {
        "num_added_faces": int(len(qualities)),
        "mean_added_face_quality": float(qualities.mean()),
        "min_added_face_quality": float(qualities.min()),
    }
def get_removed_part_bbox(sample_dir, margin=0.02):
    sample_dir = Path(sample_dir)

    with open(sample_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    source_dir = Path(meta["source_dir"])
    removed_obj_files = meta["removed_obj_files"]
    removed_obj_paths = [source_dir / "objs" / name for name in removed_obj_files]
    removed_mesh = load_and_merge_obj_list(removed_obj_paths)

    if removed_mesh is None:
        return None, None

    V = np.asarray(removed_mesh.vertices)
    bb_min = V.min(axis=0) - margin
    bb_max = V.max(axis=0) + margin
    return bb_min, bb_max


def compute_added_face_locality(M_d, M_r, bb_min, bb_max):
    F_d = np.asarray(M_d.triangles)
    F_r = np.asarray(M_r.triangles)
    V_r = np.asarray(M_r.vertices)

    num_added = len(F_r) - len(F_d)
    if num_added <= 0 or bb_min is None:
        return {
            "num_added_faces_inside_zone": 0,
            "num_added_faces_outside_zone": 0,
            "face_locality_ratio": 0.0,
        }

    added_faces = F_r[len(F_d):]

    inside = 0
    outside = 0

    for f in added_faces:
        a, b, c = map(int, f)
        center = (V_r[a] + V_r[b] + V_r[c]) / 3.0

        is_inside = np.all(center >= bb_min) and np.all(center <= bb_max)
        if is_inside:
            inside += 1
        else:
            outside += 1

    total = inside + outside
    ratio = inside / total if total > 0 else 0.0

    return {
        "num_added_faces_inside_zone": int(inside),
        "num_added_faces_outside_zone": int(outside),
        "face_locality_ratio": float(ratio),
    }