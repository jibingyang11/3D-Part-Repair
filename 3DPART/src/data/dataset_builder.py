"""
Build semantic part removal dataset from PartNet raw mesh data.
Memory-safe version with garbage collection and robust error handling.

Supports multiple object categories (Chair, Table, StorageFurniture)
and semantic part types (leg, door, etc.), enabling cross-category
generalization studies. The multi-category design aligns with recent
work on hybrid representations for 3D model understanding (cf. Uwimana
et al., VR&IH 2025) and robust part-level processing across diverse
geometries (cf. Zhang et al., Joint-Learning, CAVW 2025).

PartNet directory structure (data_v0):
  <model_id>/
    result_after_merging.json   # hierarchical part annotations
    meta.json                   # category info
    objs/
      original-*.obj            # individual part meshes

This builder:
1. Scans PartNet for the target category (lightweight - JSON only)
2. For each selected sample:
   - Parses annotations to find target semantic parts
   - Loads OBJ files as raw numpy arrays (memory-efficient)
   - Merges remaining parts -> damaged mesh
   - Merges removed parts -> removed_part mesh
   - Merges all parts -> complete mesh
   - Saves and immediately frees memory
"""

import os
import gc
import sys
import json
import random
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm

from ..io.mesh_io import load_mesh_lightweight, merge_arrays, save_mesh
from ..io.meta_io import save_meta, create_sample_meta
from ..utils import setup_logger

# PartNet JSON can be deeply nested
sys.setrecursionlimit(5000)


class DatasetBuilder:
    """Build paired (complete, damaged) mesh dataset from PartNet."""

    def __init__(self, partnet_root: str, output_dir: str,
                 category: str = "Chair", semantic_label: str = "leg",
                 total_samples: int = 100, random_seed: int = 42):
        self.partnet_root = Path(partnet_root)
        self.output_dir = Path(output_dir)
        self.category = category
        self.semantic_label = semantic_label
        self.total_samples = total_samples
        self.random_seed = random_seed
        self.logger = setup_logger("DatasetBuilder")

        random.seed(random_seed)
        np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # Phase 1: Scan (lightweight, JSON-only, no mesh loading)
    # ------------------------------------------------------------------

    def _find_partnet_samples(self) -> List[str]:
        """Find all PartNet sample directories for the given category.

        This phase is lightweight: only reads JSON files, never loads meshes.
        """
        sample_dirs = []
        skipped = 0

        candidates = sorted(self.partnet_root.iterdir())
        for d in tqdm(candidates, desc="Scanning PartNet", leave=False):
            if not d.is_dir():
                continue

            # Quick category check via meta.json first (cheapest test)
            meta_file = d / "meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("model_cat", "") != self.category:
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue
            else:
                # No meta.json means we can't confirm category - skip
                skipped += 1
                continue

            # Check annotation file exists
            anno_file = d / "result_after_merging.json"
            if not anno_file.exists():
                anno_file = d / "result.json"
            if not anno_file.exists():
                skipped += 1
                continue

            # Check objs directory exists
            objs_dir = d / "objs"
            if not objs_dir.exists():
                skipped += 1
                continue

            # Check if annotation contains the semantic label
            try:
                with open(anno_file, "r", encoding="utf-8") as f:
                    anno = json.load(f)
                if self._has_semantic_part(anno, self.semantic_label):
                    sample_dirs.append(str(d))
            except Exception:
                skipped += 1
                continue

        self.logger.info(f"Scan complete: {len(sample_dirs)} valid, {skipped} skipped")
        return sample_dirs

    def _has_semantic_part(self, anno_tree, label: str) -> bool:
        """Recursively check if annotation tree contains the given semantic label."""
        if isinstance(anno_tree, list):
            for node in anno_tree:
                if self._has_semantic_part(node, label):
                    return True
            return False
        if isinstance(anno_tree, dict):
            name = anno_tree.get("name", "").lower()
            text = anno_tree.get("text", "").lower()
            if label.lower() in name or label.lower() in text:
                return True
            children = anno_tree.get("children", [])
            if children:
                return self._has_semantic_part(children, label)
        return False

    # ------------------------------------------------------------------
    # Phase 2: Build samples one at a time
    # ------------------------------------------------------------------

    def _find_part_objs(self, anno_tree, label: str) -> List[List[str]]:
        """Find all groups of OBJ file IDs for the given semantic label."""
        results = []
        if isinstance(anno_tree, list):
            for node in anno_tree:
                results.extend(self._find_part_objs(node, label))
        elif isinstance(anno_tree, dict):
            name = anno_tree.get("name", "").lower()
            text = anno_tree.get("text", "").lower()
            if label.lower() in name or label.lower() in text:
                obj_ids = self._collect_obj_ids(anno_tree)
                if obj_ids:
                    results.append(obj_ids)
            else:
                children = anno_tree.get("children", [])
                if children:
                    results.extend(self._find_part_objs(children, label))
        return results

    def _collect_obj_ids(self, node: dict) -> List[str]:
        """Recursively collect all OBJ IDs under a node."""
        ids = []
        objs = node.get("objs", [])
        if isinstance(objs, list):
            ids.extend(objs)
        elif isinstance(objs, str):
            ids.append(objs)
        for child in node.get("children", []):
            ids.extend(self._collect_obj_ids(child))
        return ids

    def _collect_all_obj_ids(self, anno_tree) -> List[str]:
        """Collect all OBJ IDs in the entire annotation tree."""
        ids = []
        if isinstance(anno_tree, list):
            for node in anno_tree:
                ids.extend(self._collect_all_obj_ids(node))
        elif isinstance(anno_tree, dict):
            objs = anno_tree.get("objs", [])
            if isinstance(objs, list):
                ids.extend(objs)
            elif isinstance(objs, str):
                ids.append(objs)
            for child in anno_tree.get("children", []):
                ids.extend(self._collect_all_obj_ids(child))
        return ids

    def build(self) -> str:
        """Build the dataset. Returns path to the output directory."""
        self.logger.info(f"Scanning PartNet for {self.category} "
                         f"with '{self.semantic_label}' parts...")
        sample_dirs = self._find_partnet_samples()

        if len(sample_dirs) == 0:
            raise RuntimeError(
                f"No {self.category} samples with '{self.semantic_label}' "
                f"found in {self.partnet_root}.\n"
                f"Check that:\n"
                f"  1) The path is correct\n"
                f"  2) Each subfolder has meta.json with model_cat=\"{self.category}\"\n"
                f"  3) result_after_merging.json contains \"{self.semantic_label}\""
            )

        # Shuffle and select
        random.shuffle(sample_dirs)
        selected = sample_dirs[:self.total_samples]
        self.logger.info(f"Selected {len(selected)} samples for dataset")

        os.makedirs(self.output_dir, exist_ok=True)
        success_count = 0
        index_entries = []

        for i, sample_dir in enumerate(tqdm(selected, desc="Building dataset")):
            sample_id = os.path.basename(sample_dir)
            try:
                entry = self._process_sample(sample_dir, sample_id)
                if entry is not None:
                    index_entries.append(entry)
                    success_count += 1
            except Exception as e:
                self.logger.warning(f"Failed {sample_id}: {e}")

            # Force garbage collection every 10 samples
            if (i + 1) % 10 == 0:
                gc.collect()

        # Save index
        index_path = self.output_dir / "dataset_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_entries, f, indent=2)

        self.logger.info(f"Dataset built: {success_count}/{len(selected)} "
                         f"samples saved to {self.output_dir}")
        return str(self.output_dir)

    def _process_sample(self, sample_dir: str, sample_id: str) -> Optional[dict]:
        """Process a single PartNet sample. Memory-safe."""
        sample_path = Path(sample_dir)
        objs_dir = sample_path / "objs"

        # --- 1. Parse annotation (lightweight) ---
        anno_file = sample_path / "result_after_merging.json"
        if not anno_file.exists():
            anno_file = sample_path / "result.json"
        with open(anno_file, "r", encoding="utf-8") as f:
            anno = json.load(f)

        part_groups = self._find_part_objs(anno, self.semantic_label)
        if not part_groups:
            return None

        remove_group = random.choice(part_groups)
        remove_set = set(remove_group)

        all_obj_ids = self._collect_all_obj_ids(anno)

        # --- 2. Classify OBJ files into removed vs remaining ---
        removed_paths = []
        remaining_paths = []

        for obj_id in all_obj_ids:
            obj_file = objs_dir / f"{obj_id}.obj"
            if not obj_file.exists():
                continue
            if obj_id in remove_set:
                removed_paths.append(str(obj_file))
            else:
                remaining_paths.append(str(obj_file))

        if not removed_paths or not remaining_paths:
            return None

        # --- 3. Load and merge meshes ONE GROUP AT A TIME ---
        # Load removed part
        removed_dicts = []
        for p in removed_paths:
            md = load_mesh_lightweight(p)
            if md is not None:
                removed_dicts.append(md)
        if not removed_dicts:
            return None
        removed_mesh = merge_arrays(removed_dicts)
        del removed_dicts
        # Don't gc here, we need removed_mesh

        # Load remaining parts -> damaged mesh
        remaining_dicts = []
        for p in remaining_paths:
            md = load_mesh_lightweight(p)
            if md is not None:
                remaining_dicts.append(md)
        if not remaining_dicts:
            del removed_mesh
            return None
        damaged_mesh = merge_arrays(remaining_dicts)
        del remaining_dicts

        # Complete mesh = damaged + removed (cheaper than re-loading all)
        complete_mesh = trimesh.Trimesh(
            vertices=np.vstack([damaged_mesh.vertices, removed_mesh.vertices]),
            faces=np.vstack([
                damaged_mesh.faces,
                removed_mesh.faces + len(damaged_mesh.vertices),
            ]),
            process=False,
        )

        # --- 4. Save ---
        out_dir = self.output_dir / sample_id
        os.makedirs(out_dir, exist_ok=True)

        save_mesh(complete_mesh, str(out_dir / "complete.obj"))
        n_verts_before = len(complete_mesh.vertices)
        n_faces_before = len(complete_mesh.faces)
        del complete_mesh  # free immediately

        save_mesh(damaged_mesh, str(out_dir / "damaged.obj"))
        n_verts_after = len(damaged_mesh.vertices)
        n_faces_after = len(damaged_mesh.faces)
        del damaged_mesh

        save_mesh(removed_mesh, str(out_dir / "removed_part.obj"))
        del removed_mesh

        # --- 5. Save metadata ---
        meta = create_sample_meta(
            sample_id=sample_id,
            original_dir=str(sample_dir),
            category=self.category,
            removed_part_name=self.semantic_label,
            removed_obj_files=[os.path.basename(f) for f in removed_paths],
            n_verts_before=n_verts_before,
            n_faces_before=n_faces_before,
            n_verts_after=n_verts_after,
            n_faces_after=n_faces_after,
        )
        save_meta(meta, str(out_dir / "meta.json"))

        return {
            "sample_id": sample_id,
            "dir": str(out_dir),
            "category": self.category,
            "removed_part": self.semantic_label,
        }
