"""
render_paper_figures.py
=======================
Render publication-quality mesh images for paper figures.
First figure is rendered interactively to choose camera.
All following Open3D renders reuse the same saved camera.

This version:
1. uses only ONE selected chair-leg loop as the repair target
2. tries to select the corresponding removed-part component
3. increases output resolution
4. adds Adv-front+RPA baseline into comparison strip
5. keeps comparison filename as *_comparison_4col.png for LaTeX compatibility

Usage:
    cd 3DPART
    python render_paper_figures.py
    python render_paper_figures.py --sample_id 38335
    python render_paper_figures.py --sample_id 38335 --reset_camera
    python render_paper_figures.py --sample_id 38335 --leg_index 0
    python render_paper_figures.py --width 2400 --height 1800 --mpl_dpi 450
"""

import sys
import os
import argparse
import gc
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, ensure_dirs
from src.data.dataset_index import DatasetIndex
from src.data.sample_loader import SampleLoader
from src.geometry.boundary import extract_boundary_loops
from src.target_selection.selectors import select_target_loops_by_bbox, select_largest_loop
from src.repair.center_fan import center_fan_repair
from src.repair.planar_patch import planar_triangulation_repair
from src.baselines.advancing_front import advancing_front_repair
from src.geometry.bbox import compute_bbox

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARNING] open3d not installed. Using matplotlib fallback (lower quality).")
    print("         Install with: pip install open3d")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

FAILURE_CASES = [
    {
        "config": "configs/chair_leg.yaml",
        "sample_id": "35123",
        "component_index": 0,
        "rebuild_single_part": True,
        "name_prefix": "chair",
    },
    {
        "config": "configs/table_leg.yaml",
        "sample_id": "PUT_TABLE_FAILURE_ID_HERE",
        "component_index": 0,
        "rebuild_single_part": True,
        "name_prefix": "table",
    },
    {
        "config": "configs/storagefurniture_door.yaml",
        "sample_id": "PUT_STORAGE_FAILURE_ID_HERE",
        "component_index": 0,
        "rebuild_single_part": False,
        "name_prefix": "storagefurniture",
    },
]
# ═══════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════
DEFAULT_WIDTH = 2400
DEFAULT_HEIGHT = 1800
DEFAULT_MPL_DPI = 450


# ═══════════════════════════════════════════════════════════════
# Color definitions
# ═══════════════════════════════════════════════════════════════
C_MESH      = [0.82, 0.82, 0.82]   # light gray
C_REMOVED   = [0.90, 0.35, 0.25]   # red
C_PATCH_CF  = [1.00, 0.60, 0.20]   # orange
C_PATCH_PL  = [0.40, 0.78, 0.35]   # green
C_PATCH_AF  = [0.35, 0.65, 0.95]   # blue-cyan
C_PATCH_LH  = [0.95, 0.80, 0.15]   # yellow
C_BOUNDARY  = [0.20, 0.40, 0.90]   # blue
C_TARGET    = [0.15, 0.75, 0.35]   # green
C_TARGET_LH = [0.95, 0.80, 0.15]
C_BBOX      = [0.90, 0.25, 0.25]   # red
C_WHITE_BG  = [0.8, 0.8, 0.8]


# ═══════════════════════════════════════════════════════════════
# Mesh helpers
# ═══════════════════════════════════════════════════════════════
def _safe_split_mesh(mesh):
    """Split mesh into connected components."""
    if not hasattr(mesh, "split"):
        return [mesh]

    try:
        parts = list(mesh.split(only_watertight=False))
    except TypeError:
        parts = list(mesh.split())

    parts = [p for p in parts if hasattr(p, "faces") and len(p.faces) > 0]
    return parts if len(parts) > 0 else [mesh]


def _component_center(mesh):
    v = np.asarray(mesh.vertices)
    return v.mean(axis=0)


def _select_removed_component(removed_mesh, component_index=0):
    """
    从 removed_part_mesh 的连通分量中选出一个目标部件。
    对 Chair/Table 的 leg，或者 StorageFurniture 的某一扇 door，都用这个。
    """
    parts = _safe_split_mesh(removed_mesh)

    parts = sorted(
        parts,
        key=lambda m: (
            round(_component_center(m)[0], 6),
            round(_component_center(m)[1], 6),
            round(_component_center(m)[2], 6),
        )
    )

    if component_index < 0 or component_index >= len(parts):
        raise IndexError(
            f"component_index={component_index} 超出范围。"
            f"removed_part_mesh 只有 {len(parts)} 个连通分量。"
        )

    return parts[component_index]


def _build_single_part_damaged_mesh(complete_mesh, removed_part_mesh, round_digits=6):
    """
    从 complete_mesh 中真正删掉 removed_part_mesh 对应的三角面，
    得到只缺这一个目标部件的 damaged_mesh。
    """
    complete_v = np.asarray(complete_mesh.vertices)
    complete_f = np.asarray(complete_mesh.faces)
    removed_v = np.asarray(removed_part_mesh.vertices)

    def key_fn(v):
        return tuple(np.round(v, round_digits))

    coord_to_complete_vids = {}
    for i, v in enumerate(complete_v):
        k = key_fn(v)
        coord_to_complete_vids.setdefault(k, []).append(i)

    removed_vids_in_complete = set()
    for v in removed_v:
        k = key_fn(v)
        if k in coord_to_complete_vids:
            removed_vids_in_complete.update(coord_to_complete_vids[k])

    if len(removed_vids_in_complete) == 0:
        raise RuntimeError(
            "无法把 removed_part_mesh 的顶点匹配回 complete_mesh。"
            "可以尝试把 round_digits 从 6 改成 5。"
        )

    removed_vids_in_complete = np.array(sorted(removed_vids_in_complete), dtype=np.int64)

    face_remove_mask = np.all(np.isin(complete_f, removed_vids_in_complete), axis=1)

    if not np.any(face_remove_mask):
        raise RuntimeError(
            "没有找到需要删除的三角面。"
            "说明 removed_part_mesh 和 complete_mesh 的顶点对应关系没有匹配上。"
        )

    damaged_mesh = complete_mesh.copy()
    damaged_mesh.update_faces(~face_remove_mask)
    damaged_mesh.remove_unreferenced_vertices()

    return damaged_mesh
def _prepare_meshes_for_render(sample,
                               component_index=0,
                               rebuild_single_part=False,
                               round_digits=6):
    """
    统一处理 Chair / Table / StorageFurniture 的渲染输入。

    rebuild_single_part = True:
        从 complete_mesh 中删掉 selected component，得到“只缺这一个部件”的 damaged_mesh。
        适合 Chair leg / Table leg 这种 removed_part_mesh 可能包含多个连通部件的情况。

    rebuild_single_part = False:
        直接使用 sample 里已有的 damaged_mesh。
        适合 removed_part 已经是单个目标部件，或者你想保持数据集原始 damaged 版本的情况。
    """
    complete = sample["complete_mesh"]
    damaged_orig = sample["damaged_mesh"]
    removed_all = sample["removed_part_mesh"]

    parts = _safe_split_mesh(removed_all)

    if rebuild_single_part:
        removed = _select_removed_component(removed_all, component_index=component_index)
        damaged = _build_single_part_damaged_mesh(
            complete, removed, round_digits=round_digits
        )
    else:
        if len(parts) == 1:
            removed = removed_all
        else:
            removed = _select_removed_component(removed_all, component_index=component_index)
        damaged = damaged_orig

    return complete, damaged, removed, len(parts)


def _make_case_prefix(sample, sample_id, name_prefix=None):
    if name_prefix is not None and str(name_prefix).strip():
        return str(name_prefix).strip()

    meta = sample.get("meta", {})
    category = str(meta.get("category", "sample")).lower()
    removed_part = str(meta.get("removed_part_name", "part")).lower()
    return f"{category}_{removed_part}_{sample_id}"

# ═══════════════════════════════════════════════════════════════
# Open3D rendering
# ═══════════════════════════════════════════════════════════════
def _trimesh_to_o3d(mesh, color=C_MESH):
    """Convert trimesh to Open3D mesh with uniform color."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.paint_uniform_color(color)
    return o3d_mesh


def _apply_render_options(vis, bg_color=C_WHITE_BG):
    opt = vis.get_render_option()
    opt.background_color = np.array(bg_color)
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = True
    opt.light_on = True


def _apply_default_camera(vis):
    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)


def render_o3d_interactive(geometries, save_path, camera_path,
                           width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                           bg_color=C_WHITE_BG):
    """
    Interactive render for the first image only.
    Adjust camera with mouse, press S to save screenshot + camera json,
    then close the window.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=width, height=height, visible=True)

    for geom in geometries:
        vis.add_geometry(geom)

    _apply_render_options(vis, bg_color)
    _apply_default_camera(vis)

    state = {"saved": False}

    print("\n👉 这是第一次视角设定。")
    print("👉 用鼠标调整到满意角度后，按 S 保存。")
    print("👉 保存后关闭窗口，后续所有图都会复用这个视角。\n")

    def save_callback(vis_):
        print("📸 Saving first image and camera...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(camera_path), exist_ok=True)

        vis_.poll_events()
        vis_.update_renderer()
        vis_.capture_screen_image(save_path, do_render=True)

        ctr = vis_.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(camera_path, param)

        print(f"✅ Image saved to: {save_path}")
        print(f"✅ Camera saved to: {camera_path}")
        state["saved"] = True
        return False

    vis.register_key_callback(ord("S"), save_callback)
    vis.run()
    vis.destroy_window()

    if not state["saved"]:
        raise RuntimeError("未保存相机参数。请重新运行，并在第一张图窗口中按 S 保存视角。")


def render_o3d(geometries, save_path, camera_path=None,
               width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
               bg_color=C_WHITE_BG):
    """Non-interactive render. Reuse saved camera if provided."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    for geom in geometries:
        vis.add_geometry(geom)

    _apply_render_options(vis, bg_color)

    ctr = vis.get_view_control()
    if camera_path is not None and os.path.exists(camera_path):
        param = o3d.io.read_pinhole_camera_parameters(camera_path)
        ctr.convert_from_pinhole_camera_parameters(param)
    else:
        _apply_default_camera(vis)

    vis.poll_events()
    vis.update_renderer()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()


def render_o3d_with_patch(mesh, new_faces, base_color, patch_color,
                          save_path, camera_path=None,
                          width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Render repaired mesh with patch faces highlighted."""
    o3d_mesh = _trimesh_to_o3d(mesh, base_color)

    vertex_colors = np.tile(base_color, (len(mesh.vertices), 1))
    for face in new_faces:
        for vi in face:
            if vi < len(vertex_colors):
                vertex_colors[vi] = patch_color

    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    render_o3d([o3d_mesh], save_path, camera_path=camera_path,
               width=width, height=height)


def render_o3d_with_loops(mesh, loops, loop_colors, save_path,
                          camera_path=None,
                          width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Render mesh with boundary loops as colored lines."""
    o3d_mesh = _trimesh_to_o3d(mesh, C_MESH)
    geometries = [o3d_mesh]

    for loop, color in zip(loops, loop_colors):
        points = mesh.vertices[loop].tolist()
        points.append(points[0])  # close loop
        lines = [[i, i + 1] for i in range(len(points) - 1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geometries.append(line_set)

    render_o3d(geometries, save_path, camera_path=camera_path,
               width=width, height=height)


def render_o3d_with_bbox(mesh, removed_mesh, target_loops, save_path,
                         camera_path=None,
                         width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
    """Render damaged mesh with bbox of removed part and target loops."""
    o3d_mesh = _trimesh_to_o3d(mesh, C_MESH)
    geometries = [o3d_mesh]

    bbox_min, bbox_max = compute_bbox(removed_mesh)
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bbox_min,
        max_bound=bbox_max
    )
    bbox.color = np.array(C_BBOX)
    geometries.append(bbox)

    for loop in target_loops:
        points = mesh.vertices[loop].tolist()
        points.append(points[0])
        lines = [[i, i + 1] for i in range(len(points) - 1)]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([C_TARGET] * len(lines))
        geometries.append(line_set)

    render_o3d(geometries, save_path, camera_path=camera_path,
               width=width, height=height)


# ═══════════════════════════════════════════════════════════════
# Matplotlib fallback
# ═══════════════════════════════════════════════════════════════
def _mpl_render_mesh(ax, mesh, color, alpha=0.3, max_faces=3000):
    """Draw mesh as semi-transparent surface in matplotlib."""
    faces = mesh.faces
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]

    polys = [mesh.vertices[f] for f in faces]
    pc = Poly3DCollection(
        polys,
        alpha=alpha,
        facecolor=color,
        edgecolor="gray",
        linewidth=0.05
    )
    ax.add_collection3d(pc)
    _set_equal_axes(ax, mesh.vertices)
    ax.axis("off")


def _set_equal_axes(ax, vertices):
    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def render_mpl(mesh, save_path, color=C_MESH, title="",
               new_faces=None, patch_color=None, dpi=DEFAULT_MPL_DPI):
    """Matplotlib fallback renderer."""
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    _mpl_render_mesh(ax, mesh, color, alpha=0.3)

    if new_faces is not None and len(new_faces) > 0:
        polys = [mesh.vertices[f] for f in new_faces]
        pc = Poly3DCollection(
            polys,
            alpha=0.7,
            facecolor=patch_color or C_PATCH_PL,
            edgecolor="black",
            linewidth=0.3
        )
        ax.add_collection3d(pc)

    if title:
        ax.set_title(title, fontsize=11)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False
    )
    plt.close(fig)


def render_mpl_comparison(meshes, titles, save_path,
                          patch_faces_list=None,
                          patch_colors_list=None,
                          dpi=DEFAULT_MPL_DPI):
    """Matplotlib side-by-side comparison."""
    n = len(meshes)
    fig = plt.figure(figsize=(4.8 * n, 4.8), dpi=dpi)

    if patch_faces_list is None:
        patch_faces_list = [None] * n
    if patch_colors_list is None:
        patch_colors_list = [None] * n

    for i, (mesh, title) in enumerate(zip(meshes, titles)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        _mpl_render_mesh(ax, mesh, C_MESH, alpha=1)

        pf = patch_faces_list[i]
        pc_color = patch_colors_list[i]

        if pf is not None and len(pf) > 0:
            polys = [mesh.vertices[f] for f in pf]
            pc = Poly3DCollection(
                polys,
                alpha=0.65,
                facecolor=pc_color if pc_color is not None else C_PATCH_PL,
                edgecolor="black",
                linewidth=0.3
            )
            ax.add_collection3d(pc)

        ax.set_title(title, fontsize=10, pad=2)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False
    )
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Main rendering logic
# ═══════════════════════════════════════════════════════════════
def render_sample(sample_id, cfg, out_dir,
                  reset_camera=False,
                  component_index=0,
                  rebuild_single_part=False,
                  name_prefix=None,
                  width=DEFAULT_WIDTH,
                  height=DEFAULT_HEIGHT,
                  mpl_dpi=DEFAULT_MPL_DPI):
    """Render all figure materials for one sample."""
    loader = SampleLoader(cfg["paths"]["raw_data_dir"])
    sample = loader.load(sample_id)
    
    margin = cfg["repair"]["margin"]
    prox_thresh = cfg["repair"]["proximity_threshold"]
    
    sid = str(sample_id)
    prefix = _make_case_prefix(sample, sid, name_prefix=name_prefix)
    
    print(f"\n{'=' * 50}")
    print(f"Rendering sample: {sid}")
    print(f"Output prefix: {prefix}")
    print(f"{'=' * 50}")
    
    complete, damaged, removed, n_parts = _prepare_meshes_for_render(
        sample,
        component_index=component_index,
        rebuild_single_part=rebuild_single_part
    )
    
    print(f"Removed-part components: {n_parts}")
    print(f"rebuild_single_part: {rebuild_single_part}")

    loops = extract_boundary_loops(damaged)
    if len(loops) == 0:
        raise RuntimeError("新的 damaged_mesh 没有提取到 boundary loop，无法继续。")

    targets_rpa = select_target_loops_by_bbox(
        damaged, loops, removed, margin, prox_thresh
    )
    targets_lh = select_largest_loop(damaged, loops)

    # Repair
    res_cf = center_fan_repair(damaged, targets_rpa)
    res_planar = planar_triangulation_repair(damaged, targets_rpa)
    res_adv = advancing_front_repair(damaged, targets_rpa)
    res_lh = planar_triangulation_repair(damaged, targets_lh)

    if HAS_OPEN3D:
        print("  Using Open3D renderer")
        print(f"  Open3D resolution: {width} x {height}")

        camera_path = os.path.join(out_dir, f"{prefix}_camera.json")
        if reset_camera and os.path.exists(camera_path):
            os.remove(camera_path)

        # 第一张图：交互式设置视角
        first_save = os.path.join(out_dir, f"{prefix}_complete.png")
        if not os.path.exists(camera_path):
            print("  [1/10] Complete mesh... (首次交互设定视角)")
            render_o3d_interactive(
                [_trimesh_to_o3d(complete, C_MESH)],
                first_save,
                camera_path=camera_path,
                width=width,
                height=height
            )
        else:
            print("  [1/10] Complete mesh... (复用已保存视角)")
            render_o3d(
                [_trimesh_to_o3d(complete, C_MESH)],
                first_save,
                camera_path=camera_path,
                width=width,
                height=height
            )

        print("  [2/10] Damaged mesh (only one leg removed)...")
        render_o3d(
            [_trimesh_to_o3d(damaged, C_MESH)],
            os.path.join(out_dir, f"{prefix}_damaged.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [3/10] Removed part (single leg)...")
        render_o3d(
            [_trimesh_to_o3d(removed, C_REMOVED)],
            os.path.join(out_dir, f"{prefix}_removed_part.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [4/10] Repaired (center-fan)...")
        render_o3d_with_patch(
            res_cf["repaired_mesh"], res_cf["new_faces"],
            C_MESH, C_PATCH_CF,
            os.path.join(out_dir, f"{prefix}_repaired_cf.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [5/10] Repaired (planar+RPA)...")
        render_o3d_with_patch(
            res_planar["repaired_mesh"], res_planar["new_faces"],
            C_MESH, C_PATCH_PL,
            os.path.join(out_dir, f"{prefix}_repaired_planar.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [6/10] Repaired (Adv-front+RPA)...")
        render_o3d_with_patch(
            res_adv["repaired_mesh"], res_adv["new_faces"],
            C_MESH, C_PATCH_AF,
            os.path.join(out_dir, f"{prefix}_repaired_adv_front.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [7/10] Repaired (LH-only)...")
        render_o3d_with_patch(
            res_lh["repaired_mesh"], res_lh["new_faces"],
            C_MESH, C_PATCH_LH,
            os.path.join(out_dir, f"{prefix}_repaired_lh.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [8/10] Boundary loops...")
        loop_colors = [C_BOUNDARY] * len(loops)
        render_o3d_with_loops(
            damaged, loops, loop_colors,
            os.path.join(out_dir, f"{prefix}_boundary_loops.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [9/10] Target selection + bbox...")
        render_o3d_with_bbox(
            damaged, removed, targets_rpa,
            os.path.join(out_dir, f"{prefix}_target_loops.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )
        print("  [+] Failure case target visualization (RPA)...")
        render_o3d_with_loops(
            damaged,
            targets_rpa,
            [C_TARGET] * len(targets_rpa),
            os.path.join(out_dir, f"{prefix}_failure_rpa_target.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [+] Failure case target visualization (LH)...")
        render_o3d_with_loops(
            damaged,
            targets_lh,
            [C_TARGET_LH] * len(targets_lh),
            os.path.join(out_dir, f"{prefix}_failure_lh_target.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )
        print("  [+] Failure case patch visualization (RPA)...")
        render_o3d_with_patch(
            res_planar["repaired_mesh"], res_planar["new_faces"],
            C_MESH, C_PATCH_PL,
            os.path.join(out_dir, f"{prefix}_failure_rpa_patch.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

        print("  [+] Failure case patch visualization (LH)...")
        render_o3d_with_patch(
            res_lh["repaired_mesh"], res_lh["new_faces"],
            C_MESH, C_PATCH_LH,
            os.path.join(out_dir, f"{prefix}_failure_lh_patch.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )
        print("  [10/10] Patch overlay...")
        render_o3d_with_patch(
            res_planar["repaired_mesh"], res_planar["new_faces"],
            C_MESH, C_PATCH_PL,
            os.path.join(out_dir, f"{prefix}_patch_overlay.png"),
            camera_path=camera_path,
            width=width,
            height=height
        )

    else:
        print("  Using matplotlib fallback")
        print(f"  Matplotlib dpi: {mpl_dpi}")

        print("  [1/7] Complete mesh...")
        render_mpl(
            complete,
            os.path.join(out_dir, f"{prefix}_complete.png"),
            title="Complete mesh $M_{gt}$",
            dpi=mpl_dpi
        )

        print("  [2/7] Damaged mesh (only one leg removed)...")
        render_mpl(
            damaged,
            os.path.join(out_dir, f"{prefix}_damaged.png"),
            title="Damaged mesh $M_d$",
            dpi=mpl_dpi
        )

        print("  [3/7] Removed part (single leg)...")
        render_mpl(
            removed,
            os.path.join(out_dir, f"{prefix}_removed_part.png"),
            color=C_REMOVED,
            title="Removed part",
            dpi=mpl_dpi
        )

        print("  [4/7] Repaired (center-fan)...")
        render_mpl(
            res_cf["repaired_mesh"],
            os.path.join(out_dir, f"{prefix}_repaired_cf.png"),
            new_faces=res_cf["new_faces"],
            patch_color=C_PATCH_CF,
            title="Center-fan",
            dpi=mpl_dpi
        )

        print("  [5/7] Repaired (planar+RPA)...")
        render_mpl(
            res_planar["repaired_mesh"],
            os.path.join(out_dir, f"{prefix}_repaired_planar.png"),
            new_faces=res_planar["new_faces"],
            patch_color=C_PATCH_PL,
            title="Planar + RPA",
            dpi=mpl_dpi
        )

        print("  [6/7] Repaired (Adv-front+RPA)...")
        render_mpl(
            res_adv["repaired_mesh"],
            os.path.join(out_dir, f"{prefix}_repaired_adv_front.png"),
            new_faces=res_adv["new_faces"],
            patch_color=C_PATCH_AF,
            title="Adv-front + RPA",
            dpi=mpl_dpi
        )

        print("  [7/7] Repaired (LH-only)...")
        render_mpl(
            res_lh["repaired_mesh"],
            os.path.join(out_dir, f"{prefix}_repaired_lh.png"),
            new_faces=res_lh["new_faces"],
            patch_color=C_PATCH_LH,
            title="Planar + LH-only",
            dpi=mpl_dpi
        )

    print("  [+] Comparison strip (5 columns, saved with old filename for compatibility)...")
    render_mpl_comparison(
        meshes=[
            damaged,
            res_cf["repaired_mesh"],
            res_planar["repaired_mesh"],
            res_adv["repaired_mesh"],
            res_lh["repaired_mesh"]
        ],
        titles=[
            "$M_d$",
            "Center-fan",
            "Planar+RPA",
            "Adv-front+RPA",
            "Planar+LH-only"
        ],
        save_path=os.path.join(out_dir, f"{prefix}_comparison_4col.png"),
        patch_faces_list=[
            None,
            res_cf["new_faces"],
            res_planar["new_faces"],
            res_adv["new_faces"],
            res_lh["new_faces"]
        ],
        patch_colors_list=[
            None,
            C_PATCH_CF,
            C_PATCH_PL,
            C_PATCH_AF,
            C_PATCH_LH
        ],
        dpi=mpl_dpi
    )

    del res_cf, res_planar, res_adv, res_lh
    gc.collect()
    print(f"  Done! Files saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render paper figures")
    parser.add_argument("--sample_id", type=str, default=None,
                        help="Specific sample ID to render")
    parser.add_argument("--all_samples", type=int, default=0,
                        help="Render this many samples (0 = just one)")
    parser.add_argument("--config", type=str, default="configs/chair_leg.yaml",
                        help="Config file path")
    parser.add_argument("--reset_camera", action="store_true",
                        help="Delete saved camera and choose angle again")

    parser.add_argument("--component_index", type=int, default=0,
                    help="Which connected component inside removed_part_mesh to use (default: 0)")

    parser.add_argument("--leg_index", type=int, default=None,
                        help="Deprecated alias of --component_index")
    
    parser.add_argument("--rebuild_single_part", action="store_true",
                        help="Rebuild damaged mesh by removing only the selected component from complete mesh.")
    
    parser.add_argument("--name_prefix", type=str, default=None,
                        help="Prefix used in output filenames, e.g. chair / table / storagefurniture")

    # 分辨率
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help=f"Open3D output width (default: {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help=f"Open3D output height (default: {DEFAULT_HEIGHT})")
    parser.add_argument("--mpl_dpi", type=int, default=DEFAULT_MPL_DPI,
                        help=f"Matplotlib dpi (default: {DEFAULT_MPL_DPI})")
    parser.add_argument("--render_failure_triplet", action="store_true",
                    help="Render chair/table/storagefurniture failure cases in one run")
    args = parser.parse_args()
    if args.render_failure_triplet:
        print("Rendering predefined failure triplet cases...")
    
        for case in FAILURE_CASES:
            if str(case.get("sample_id", "")).startswith("PUT_"):
                print(f"[SKIP] Placeholder sample_id still present for {case['name_prefix']}: {case['sample_id']}")
                continue
            cfg = load_config(os.path.join(PROJECT_ROOT, case["config"]))
            ensure_dirs(cfg)
    
            out_dir = os.path.join(cfg["paths"]["figures_dir"], "renders")
            os.makedirs(out_dir, exist_ok=True)
    
            render_sample(
                case["sample_id"],
                cfg,
                out_dir,
                reset_camera=args.reset_camera,
                component_index=case["component_index"],
                rebuild_single_part=case["rebuild_single_part"],
                name_prefix=case["name_prefix"],
                width=args.width,
                height=args.height,
                mpl_dpi=args.mpl_dpi
            )
            gc.collect()
    
        print("All predefined failure cases rendered.")
        return
    component_index = args.component_index
    if args.leg_index is not None:
        component_index = args.leg_index

    cfg = load_config(os.path.join(PROJECT_ROOT, args.config))
    ensure_dirs(cfg)

    index = DatasetIndex(cfg["paths"]["raw_data_dir"])
    out_dir = os.path.join(cfg["paths"]["figures_dir"], "renders")
    os.makedirs(out_dir, exist_ok=True)

    if args.sample_id:
        sample_ids = [args.sample_id]
    elif args.all_samples > 0:
        sample_ids = index.sample_ids[:args.all_samples]
    else:
        sample_ids = [index.sample_ids[0]]

    print(f"Will render {len(sample_ids)} sample(s)")
    print(f"Output directory: {out_dir}")
    print(f"Renderer: {'Open3D' if HAS_OPEN3D else 'matplotlib (fallback)'}")
    print(f"Selected component index: {component_index}")
    print(f"Rebuild single part: {args.rebuild_single_part}")
    print(f"Name prefix: {args.name_prefix}")
    print(f"Open3D resolution: {args.width} x {args.height}")
    print(f"Matplotlib dpi: {args.mpl_dpi}")

    for sid in sample_ids:
        render_sample(
            sid, cfg, out_dir,
            reset_camera=args.reset_camera,
            component_index=component_index,
            rebuild_single_part=args.rebuild_single_part,
            name_prefix=args.name_prefix,
            width=args.width,
            height=args.height,
            mpl_dpi=args.mpl_dpi
        )
        gc.collect()

    print(f"\n{'=' * 50}")
    print(f"All done! {len(sample_ids)} sample(s) rendered.")
    print(f"Output: {out_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()