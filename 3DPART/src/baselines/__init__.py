from .largest_hole import largest_hole_baseline
from .trimesh_fill import trimesh_fill_all_holes, trimesh_fill_target_loops
from .advancing_front import advancing_front_repair

# Open3D baselines - optional dependency
try:
    from .open3d_fill import open3d_poisson_fill, open3d_ball_pivoting_fill
except ImportError:
    pass
