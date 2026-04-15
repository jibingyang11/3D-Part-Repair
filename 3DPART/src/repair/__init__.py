from .center_fan import center_fan_repair
from .planar_patch import planar_triangulation_repair
from .minimal_area_patch import minimal_area_repair
from .mlp_patch import mlp_patch_repair, MLPPatchGenerator, train_mlp_patch_generator
from .registry import get_repair_method, REPAIR_METHODS
