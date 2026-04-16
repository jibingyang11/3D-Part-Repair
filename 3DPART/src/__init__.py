"""
3D Part Repair: Minimal-Change Local Mesh Repair after Semantic Part Removal

This package provides tools for:
- Building semantic part removal datasets from PartNet (Chair, Table, StorageFurniture)
- Extracting boundary loops from damaged meshes
- Removed-part-aware target boundary loop selection (geometric + learned)
- Geometric repair baselines (center-fan, planar triangulation)
- Lightweight learning-based patch generation (MLP Patch)
- SOTA comparison baselines (advancing-front, Poisson, trimesh fill-all)
- Evaluation metrics (closure, complexity, quality, locality, distance)
"""

__version__ = "2.0.0"
