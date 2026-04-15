"""Registry of all repair methods for easy dispatch."""

from .center_fan import center_fan_repair
from .planar_patch import planar_triangulation_repair
from .minimal_area_patch import minimal_area_repair
from .mlp_patch import mlp_patch_repair
from ..baselines.advancing_front import advancing_front_repair
from ..baselines.trimesh_fill import trimesh_fill_all_holes, trimesh_fill_target_loops

REPAIR_METHODS = {
    "center_fan": center_fan_repair,
    "planar": planar_triangulation_repair,
    "planar_triangulation": planar_triangulation_repair,
    "minimal_area": minimal_area_repair,
    "ear_clipping": minimal_area_repair,
    "mlp_patch": mlp_patch_repair,
    "advancing_front": advancing_front_repair,
    "trimesh_fill_target": trimesh_fill_target_loops,
    "trimesh_fill_all": trimesh_fill_all_holes,
}

# Open3D methods (optional)
try:
    from ..baselines.open3d_fill import open3d_poisson_fill, open3d_ball_pivoting_fill
    REPAIR_METHODS["poisson"] = open3d_poisson_fill
    REPAIR_METHODS["ball_pivoting"] = open3d_ball_pivoting_fill
except ImportError:
    pass


def get_repair_method(name: str):
    """Get a repair method function by name."""
    if name not in REPAIR_METHODS:
        raise ValueError(f"Unknown repair method: {name}. "
                         f"Available: {list(REPAIR_METHODS.keys())}")
    return REPAIR_METHODS[name]
