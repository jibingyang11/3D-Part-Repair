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


# ---------- 对多个 loop 做局部平面投影 + triangulation ----------

def triangulate_multiple_loops_on_plane(mesh, loops):
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()

    old_n = len(V)
    new_vertices = []
    all_patch_faces = []
    new_vertex_ids = []

    for main_loop in loops:
        loop_vids = [int(v) for v in main_loop]
        loop_pts = V[loop_vids]

        if len(loop_pts) < 3:
            continue

        center = loop_pts.mean(axis=0)

        center_vid = old_n + len(new_vertices)
        new_vertices.append(center)
        new_vertex_ids.append(center_vid)

        for i in range(len(loop_vids)):
            a = loop_vids[i]
            b = loop_vids[(i + 1) % len(loop_vids)]
            all_patch_faces.append([a, b, center_vid])

    if len(all_patch_faces) == 0:
        return None, []

    V_new = np.vstack([V, np.array(new_vertices, dtype=np.float64)])
    F_new = np.vstack([F, np.array(all_patch_faces, dtype=np.int32)])

    M_r = o3d.geometry.TriangleMesh()
    M_r.vertices = o3d.utility.Vector3dVector(V_new)
    M_r.triangles = o3d.utility.Vector3iVector(F_new)
    M_r.compute_triangle_normals()
    M_r.compute_vertex_normals()

    return M_r, new_vertex_ids


# ---------- 单样本修补 ----------

def repair_one_sample_center_fan(sample_dir):
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

   
    M_patch, new_vids = triangulate_multiple_loops_on_plane(M_d, nearby_loops)

    if M_patch is None:
        return {
            "sample_id": sample_dir.name,
            "success": False,
            "reason": "triangulation failed"
        }
    
    M_r = M_patch
    boundary_edges_after = get_boundary_edges_o3d(M_r)
    loops_after = boundary_edges_to_loops(boundary_edges_after)

    with open(sample_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    source_dir = Path(meta["source_dir"])
    removed_obj_files = meta["removed_obj_files"]
    removed_obj_paths = [source_dir / "objs" / name for name in removed_obj_files]
    removed_mesh = load_and_merge_obj_list(removed_obj_paths)

    nearest_loop_len_after = 0.0
    nearest_loop_dist_after = None
    
    V_r = np.asarray(M_r.vertices)
    target_loops_after = choose_nearby_loops_by_removed_part_from_mesh(
        loops_after, M_r, removed_mesh, margin=0.02, min_len=6
    )
    nearest_loop_len_after = compute_total_loop_perimeter(V_r, target_loops_after)
    disp_stats = compute_vertex_displacement_stats(M_d, M_r)
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
        "loop_lens_before": loop_lens_before,
        "total_loop_len_before": total_loop_len_before,
        "nearest_loop_len_after": float(nearest_loop_len_after),
        "nearest_loop_dist_after": nearest_loop_dist_after,
        "global_boundary_edges_before": int(len(boundary_edges_before)),
        "global_boundary_edges_after": int(len(boundary_edges_after)),
        "improvement": float(total_loop_len_before - nearest_loop_len_after),
        "mean_displacement": disp_stats["mean_displacement"],
        "max_displacement": disp_stats["max_displacement"],
        "num_new_vertices": num_new_vertices,
        "num_added_faces": q["num_added_faces"],
        "mean_added_face_quality": q["mean_added_face_quality"],
        "min_added_face_quality": q["min_added_face_quality"],
        "num_added_faces_inside_zone": loc["num_added_faces_inside_zone"],
        "num_added_faces_outside_zone": loc["num_added_faces_outside_zone"],
        "face_locality_ratio": loc["face_locality_ratio"],
    }

def build_vertex_adjacency(mesh):
    triangles = np.asarray(mesh.triangles)
    adj = defaultdict(set)

    for tri in triangles:
        a, b, c = map(int, tri)
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])

    return adj


def laplacian_smooth_selected_vertices(mesh, selected_vids, num_iters=10, alpha=0.4):
    V = np.asarray(mesh.vertices).copy()
    F = np.asarray(mesh.triangles).copy()

    selected_vids = [int(v) for v in selected_vids]
    selected_set = set(selected_vids)

    adj = build_vertex_adjacency(mesh)

    for _ in range(num_iters):
        V_new = V.copy()

        for vid in selected_vids:
            neighbors = list(adj[vid])
            if len(neighbors) == 0:
                continue

            nb_mean = V[neighbors].mean(axis=0)
            V_new[vid] = (1.0 - alpha) * V[vid] + alpha * nb_mean

        V = V_new

    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(V)
    out_mesh.triangles = o3d.utility.Vector3iVector(F)
    out_mesh.compute_triangle_normals()
    out_mesh.compute_vertex_normals()
    return out_mesh
# ---------- 可视化 ----------

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
def compute_vertex_displacement_stats(M_d, M_r):
    V_d = np.asarray(M_d.vertices)
    V_r = np.asarray(M_r.vertices)

    n = min(len(V_d), len(V_r))
    disp = np.linalg.norm(V_r[:n] - V_d[:n], axis=1)

    return {
        "mean_displacement": float(disp.mean()),
        "max_displacement": float(disp.max()),
    }
def count_new_vertices(M_d, M_r):
    return int(len(np.asarray(M_r.vertices)) - len(np.asarray(M_d.vertices)))
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