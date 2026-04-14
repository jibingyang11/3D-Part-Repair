"""Publication-quality mesh rendering via Open3D offscreen.

Generates Figure 1 / 3 / 5 style images:
- Colored mesh surfaces with highlighted patches
- Multi-view comparison panels
- Boundary loop overlays

Falls back to matplotlib if Open3D is unavailable.
"""

import os
import numpy as np
import trimesh
from typing import List, Optional, Tuple

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ------------------------------------------------------------------ #
#  Color palette
# ------------------------------------------------------------------ #
COLOR_MESH       = [0.85, 0.85, 0.85]   # light gray
COLOR_REMOVED    = [1.00, 0.60, 0.60]   # red-ish
COLOR_PATCH      = [1.00, 0.75, 0.30]   # orange
COLOR_PATCH_ALT  = [0.40, 0.80, 0.40]   # green
COLOR_BOUNDARY   = [0.20, 0.40, 0.90]   # blue


def render_single_mesh(mesh: trimesh.Trimesh,
                       save_path: str,
                       color: list = None,
                       highlight_faces: np.ndarray = None,
                       highlight_color: list = None,
                       title: str = "",
                       size: Tuple[int, int] = (800, 600)):
    """Render a single mesh and save to image file."""
    if HAS_OPEN3D:
        _render_o3d(mesh, save_path, color or COLOR_MESH,
                    highlight_faces, highlight_color or COLOR_PATCH, size)
    else:
        _render_mpl(mesh, save_path, color or COLOR_MESH,
                    highlight_faces, highlight_color or COLOR_PATCH, title)


def render_mesh_comparison(meshes: List[trimesh.Trimesh],
                           titles: List[str],
                           save_path: str,
                           patch_faces_list: List[Optional[np.ndarray]] = None,
                           boundary_loops_list: List[Optional[List[List[int]]]] = None,
                           figsize: Tuple[int, int] = None):
    """Render multiple meshes side-by-side for comparison figures.

    This is used for Fig. 3 (qualitative comparison) and Fig. 5 (failure cases).
    """
    n = len(meshes)
    if patch_faces_list is None:
        patch_faces_list = [None] * n
    if boundary_loops_list is None:
        boundary_loops_list = [None] * n

    if figsize is None:
        figsize = (5 * n, 5)

    fig = plt.figure(figsize=figsize, dpi=200)

    for i, (mesh, title) in enumerate(zip(meshes, titles)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")

        # Draw mesh wireframe (light)
        _draw_mesh_wireframe(ax, mesh, alpha=0.08)

        # Draw boundary loops
        if boundary_loops_list[i]:
            for loop in boundary_loops_list[i]:
                verts = mesh.vertices[loop]
                vc = np.vstack([verts, verts[0:1]])
                ax.plot(vc[:, 0], vc[:, 1], vc[:, 2],
                        color=COLOR_BOUNDARY, linewidth=1.5, alpha=0.9)

        # Draw patch faces
        if patch_faces_list[i] is not None and len(patch_faces_list[i]) > 0:
            polys = [mesh.vertices[f] for f in patch_faces_list[i]]
            pc = Poly3DCollection(polys, alpha=0.65,
                                  facecolor=COLOR_PATCH, edgecolor="black",
                                  linewidth=0.3)
            ax.add_collection3d(pc)

        ax.set_title(title, fontsize=10, pad=2)
        _set_equal_axes(ax, mesh.vertices)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Pipeline figure: Fig 2 style
# ------------------------------------------------------------------ #

def render_pipeline_figure(damaged_mesh: trimesh.Trimesh,
                           removed_mesh: trimesh.Trimesh,
                           repaired_mesh: trimesh.Trimesh,
                           target_loops: List[List[int]],
                           all_loops: List[List[int]],
                           patch_faces: np.ndarray,
                           save_path: str):
    """Render the 3-stage pipeline figure (Fig. 2)."""
    fig = plt.figure(figsize=(16, 5), dpi=200)

    # Panel 1: Damaged mesh with all boundary loops
    ax1 = fig.add_subplot(131, projection="3d")
    _draw_mesh_wireframe(ax1, damaged_mesh, alpha=0.06)
    for loop in all_loops:
        verts = damaged_mesh.vertices[loop]
        vc = np.vstack([verts, verts[0:1]])
        ax1.plot(vc[:, 0], vc[:, 1], vc[:, 2], linewidth=1.2, alpha=0.7)
    ax1.set_title("1. Boundary Extraction\n(Damaged Mesh $M_d$)", fontsize=10)
    _set_equal_axes(ax1, damaged_mesh.vertices)
    ax1.axis("off")

    # Panel 2: Target selection with removed part bbox
    ax2 = fig.add_subplot(132, projection="3d")
    _draw_mesh_wireframe(ax2, damaged_mesh, alpha=0.06)
    for loop in target_loops:
        verts = damaged_mesh.vertices[loop]
        vc = np.vstack([verts, verts[0:1]])
        ax2.plot(vc[:, 0], vc[:, 1], vc[:, 2],
                 color="green", linewidth=2.5, alpha=0.9)
    # Draw removed part bounding box
    _draw_bbox(ax2, removed_mesh)
    ax2.set_title("2. Target Selection\n(Removed-Part-Aware)", fontsize=10)
    _set_equal_axes(ax2, damaged_mesh.vertices)
    ax2.axis("off")

    # Panel 3: Repaired mesh with patch highlighted
    ax3 = fig.add_subplot(133, projection="3d")
    _draw_mesh_wireframe(ax3, repaired_mesh, alpha=0.06)
    if len(patch_faces) > 0:
        polys = [repaired_mesh.vertices[f] for f in patch_faces]
        pc = Poly3DCollection(polys, alpha=0.7,
                              facecolor=COLOR_PATCH, edgecolor="black",
                              linewidth=0.3)
        ax3.add_collection3d(pc)
    ax3.set_title("3. Local Repair\n(Repaired Mesh $M_r$)", fontsize=10)
    _set_equal_axes(ax3, repaired_mesh.vertices)
    ax3.axis("off")

    plt.tight_layout(pad=1.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

def _draw_mesh_wireframe(ax, mesh, alpha=0.1, color="gray", max_faces=5000):
    """Draw mesh as semi-transparent wireframe."""
    faces = mesh.faces
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    polys = [mesh.vertices[f] for f in faces]
    pc = Poly3DCollection(polys, alpha=alpha, facecolor=color,
                          edgecolor="gray", linewidth=0.05)
    ax.add_collection3d(pc)


def _draw_bbox(ax, mesh, color="red", linestyle="--", alpha=0.5):
    """Draw axis-aligned bounding box."""
    mn = mesh.vertices.min(axis=0)
    mx = mesh.vertices.max(axis=0)
    # 12 edges of a box
    edges = [
        [mn, [mx[0], mn[1], mn[2]]], [mn, [mn[0], mx[1], mn[2]]],
        [mn, [mn[0], mn[1], mx[2]]], [mx, [mn[0], mx[1], mx[2]]],
        [mx, [mx[0], mn[1], mx[2]]], [mx, [mx[0], mx[1], mn[2]]],
        [[mx[0], mn[1], mn[2]], [mx[0], mx[1], mn[2]]],
        [[mx[0], mn[1], mn[2]], [mx[0], mn[1], mx[2]]],
        [[mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]]],
        [[mn[0], mx[1], mn[2]], [mn[0], mx[1], mx[2]]],
        [[mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]]],
        [[mn[0], mn[1], mx[2]], [mn[0], mx[1], mx[2]]],
    ]
    for e in edges:
        pts = np.array(e)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, linestyle=linestyle, linewidth=1.0, alpha=alpha)


def _set_equal_axes(ax, vertices):
    """Set equal aspect ratio for 3D plot."""
    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# ------------------------------------------------------------------ #
#  Open3D offscreen renderer (if available)
# ------------------------------------------------------------------ #

def _render_o3d(mesh, save_path, base_color, highlight_faces,
                highlight_color, size):
    """Render using Open3D offscreen."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()

    # Color all faces
    face_colors = np.tile(base_color, (len(mesh.faces), 1))
    if highlight_faces is not None and len(highlight_faces) > 0:
        for f in highlight_faces:
            if f[0] < len(face_colors):
                face_idx = np.where((mesh.faces == f).all(axis=1))[0]
                for fi in face_idx:
                    face_colors[fi] = highlight_color

    # Map face colors to vertex colors (approximate)
    vertex_colors = np.tile(base_color, (len(mesh.vertices), 1))
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    # Render offscreen
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=size[0], height=size[1], visible=False)
    vis.add_geometry(o3d_mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.mesh_show_wireframe = False

    vis.poll_events()
    vis.update_renderer()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vis.capture_screen_image(save_path)
    vis.destroy_window()


def _render_mpl(mesh, save_path, base_color, highlight_faces,
                highlight_color, title):
    """Fallback matplotlib renderer."""
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    _draw_mesh_wireframe(ax, mesh, alpha=0.1, color=base_color)

    if highlight_faces is not None and len(highlight_faces) > 0:
        polys = [mesh.vertices[f] for f in highlight_faces]
        pc = Poly3DCollection(polys, alpha=0.7,
                              facecolor=highlight_color,
                              edgecolor="black", linewidth=0.3)
        ax.add_collection3d(pc)

    if title:
        ax.set_title(title)
    _set_equal_axes(ax, mesh.vertices)
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
