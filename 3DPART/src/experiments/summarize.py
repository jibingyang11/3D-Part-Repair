"""Summarize experiment results and generate all paper tables."""

import os
import numpy as np
import pandas as pd
from typing import Dict, List

from ..utils import save_json


# All methods in display order
ALL_METHODS = {
    "Center-fan + RPA": "center_fan",
    "Planar + RPA": "planar_removed_part_aware",
    "MLP Patch + RPA": "mlp_patch_rpa",
    "Planar + LH-only": "planar_largest_hole_only",
    "Trimesh fill-all": "trimesh_fill_all",
    "Adv-front + RPA": "advancing_front_rpa",
    "Poisson recon": "open3d_poisson",
}

OURS_METHODS = {
    "Center-fan + RPA": "center_fan",
    "Planar + RPA": "planar_removed_part_aware",
    "MLP Patch + RPA": "mlp_patch_rpa",
    "Planar + LH-only": "planar_largest_hole_only",
}


def _safe_mean(df, col):
    if col in df.columns:
        return df[col].dropna().mean()
    return np.nan


def summarize_results(df: pd.DataFrame, output_dir: str) -> Dict[str, pd.DataFrame]:
    """Generate all summary tables.

    Key change:
    - table6_sota_comparison.csv is now a FULL table for the paper's main
      quantitative comparison, including:
        Residual, Improvement, New Vtx, New Faces, Avg Quality, Locality, CD, HD
    - table7_distance.csv is still kept as a separate distance-only table.
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    tables = {}

    # Table 2: Overall quantitative comparison (ours)
    tables["table2_quantitative"] = _build_quantitative_table(df, OURS_METHODS)
    tables["table2_quantitative"].to_csv(
        os.path.join(tables_dir, "table2_quantitative.csv"),
        index=False
    )

    # Table 3: Locality analysis (ours)
    tables["table3_locality"] = _build_locality_table(df, ALL_METHODS)
    tables["table3_locality"].to_csv(
        os.path.join(tables_dir, "table3_locality.csv"),
        index=False
    )

    # Table 4: Failure cases
    tables["table4_failures"] = _build_failure_table(df)
    tables["table4_failures"].to_csv(
        os.path.join(tables_dir, "table4_failures.csv"),
        index=False
    )

    # Table 5: Target selection comparison
    tables["table5_target_selection"] = _build_target_selection_table(df)
    tables["table5_target_selection"].to_csv(
        os.path.join(tables_dir, "table5_target_selection.csv"),
        index=False
    )

    # Detect available methods
    available = {}
    for display_name, prefix in ALL_METHODS.items():
        probe_cols = [
            f"{prefix}/target_loop_length_after",
            f"{prefix}/n_new_faces",
            f"{prefix}/avg_new_face_quality",
            f"{prefix}/locality_ratio",
            f"{prefix}/chamfer_distance",
        ]
        if any(col in df.columns for col in probe_cols):
            available[display_name] = prefix

    # Table 6: Full SOTA comparison for paper main table
    # Columns match the LaTeX Table 2 you want to fill directly.
    if len(available) > 0:
        rows = []
        for display_name, prefix in available.items():
            row = {
                "Method": display_name,
                "Residual ↓": _safe_mean(df, f"{prefix}/target_loop_length_after"),
                "Improvement ↑": _safe_mean(df, f"{prefix}/improvement"),
                "New Vtx ↓": _safe_mean(df, f"{prefix}/n_new_vertices"),
                "New Faces ↓": _safe_mean(df, f"{prefix}/n_new_faces"),
                "Avg Quality ↑": _safe_mean(df, f"{prefix}/avg_new_face_quality"),
                "Locality ↑": _safe_mean(df, f"{prefix}/locality_ratio"),
                "CD ↓": _safe_mean(df, f"{prefix}/chamfer_distance"),
                "HD ↓": _safe_mean(df, f"{prefix}/hausdorff_distance"),
            }
            rows.append(row)

        tables["table6_sota_comparison"] = pd.DataFrame(rows)

        # Optional: reorder rows to exactly follow ALL_METHODS order
        method_order = list(ALL_METHODS.keys())
        tables["table6_sota_comparison"]["_order"] = tables["table6_sota_comparison"]["Method"].map(
            {m: i for i, m in enumerate(method_order)}
        )
        tables["table6_sota_comparison"] = (
            tables["table6_sota_comparison"]
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )

        tables["table6_sota_comparison"].to_csv(
            os.path.join(tables_dir, "table6_sota_comparison.csv"),
            index=False
        )

    # Table 7: Distance metrics only
    dist_col = "planar_removed_part_aware/chamfer_distance"
    if dist_col in df.columns and len(available) > 0:
        tables["table7_distance"] = _build_distance_table(df, available)
        tables["table7_distance"].to_csv(
            os.path.join(tables_dir, "table7_distance.csv"),
            index=False
        )

    # Table 8: MLP Patch comparison (new)
    mlp_methods = {
        "Center-fan + RPA": "center_fan",
        "MLP Patch + RPA": "mlp_patch_rpa",
        "Planar + RPA": "planar_removed_part_aware",
    }
    mlp_available = {k: v for k, v in mlp_methods.items()
                     if any(f"{v}/{c}" in df.columns
                            for c in ["target_loop_length_after", "n_new_faces"])}
    if len(mlp_available) > 1:
        tables["table8_mlp_comparison"] = _build_quantitative_table(df, mlp_available)
        tables["table8_mlp_comparison"].to_csv(
            os.path.join(tables_dir, "table8_mlp_comparison.csv"),
            index=False
        )

    return tables


def _build_quantitative_table(df, methods):
    rows = []
    for display_name, prefix in methods.items():
        row = {"Method": display_name}
        col_map = {
            "Residual ↓": f"{prefix}/target_loop_length_after",
            "Improvement ↑": f"{prefix}/improvement",
            "New Vtx ↓": f"{prefix}/n_new_vertices",
            "New Faces ↓": f"{prefix}/n_new_faces",
            "Avg Quality ↑": f"{prefix}/avg_new_face_quality",
            "Min Quality ↑": f"{prefix}/min_new_face_quality",
        }
        for metric_name, col_name in col_map.items():
            row[metric_name] = _safe_mean(df, col_name)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_locality_table(df, methods):
    rows = []
    for display_name, prefix in methods.items():
        row = {"Method": display_name}
        col_map = {
            "Inside ↑": f"{prefix}/n_faces_inside",
            "Outside ↓": f"{prefix}/n_faces_outside",
            "Locality ↑": f"{prefix}/locality_ratio",
        }
        for metric_name, col_name in col_map.items():
            row[metric_name] = _safe_mean(df, col_name)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_failure_table(df, top_n=10):
    rpa_col = "planar_removed_part_aware/target_loop_length_after"
    lh_col = "planar_largest_hole_only/target_loop_length_after"
    lh_outside = "planar_largest_hole_only/n_faces_outside"
    rpa_improv = "planar_removed_part_aware/improvement"
    lh_improv = "planar_largest_hole_only/improvement"

    required = [rpa_col, lh_col]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df2 = df.copy()
    df2["gap"] = df2[lh_col] - df2[rpa_col]
    df2 = df2.sort_values("gap", ascending=False).head(top_n)

    rows = []
    for _, r in df2.iterrows():
        rows.append({
            "ID": r["sample_id"],
            "RP Residual": r.get(rpa_col, np.nan),
            "LH Residual": r.get(lh_col, np.nan),
            "LH Outside": r.get(lh_outside, np.nan),
            "RP Improv": r.get(rpa_improv, np.nan),
            "LH Improv": r.get(lh_improv, np.nan),
            "Gap": r.get("gap", np.nan),
        })
    return pd.DataFrame(rows)


def _build_target_selection_table(df):
    methods = {
        "Largest-hole-only": "planar_largest_hole_only",
        "Removed-part-aware": "planar_removed_part_aware",
    }
    rows = []
    for display_name, prefix in methods.items():
        row = {"Method": display_name}
        col_map = {
            "Residual ↓": f"{prefix}/target_loop_length_after",
            "Improvement ↑": f"{prefix}/improvement",
            "Outside ↓": f"{prefix}/n_faces_outside",
            "Locality ↑": f"{prefix}/locality_ratio",
            "Quality ↑": f"{prefix}/avg_new_face_quality",
        }
        for metric_name, col_name in col_map.items():
            if col_name in df.columns:
                vals = df[col_name].dropna()
                row[metric_name] = vals.mean()
                if "Residual" in metric_name:
                    row["Median ↓"] = vals.median()
            else:
                row[metric_name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_sota_table(df, methods):
    """Table 6: Full comparison including SOTA methods."""
    rows = []
    for display_name, prefix in methods.items():
        row = {"Method": display_name}
        col_map = {
            "Residual ↓": f"{prefix}/target_loop_length_after",
            "New Faces ↓": f"{prefix}/n_new_faces",
            "Quality ↑": f"{prefix}/avg_new_face_quality",
            "Locality ↑": f"{prefix}/locality_ratio",
            "Closure ↑": f"{prefix}/closure_ratio",
        }
        for metric_name, col_name in col_map.items():
            row[metric_name] = _safe_mean(df, col_name)
        rows.append(row)
    return pd.DataFrame(rows)


def _build_distance_table(df, methods):
    """Table 7: Distance metrics comparison."""
    rows = []
    for display_name, prefix in methods.items():
        row = {"Method": display_name}
        col_map = {
            "Chamfer Dist ↓": f"{prefix}/chamfer_distance",
            "Hausdorff ↓": f"{prefix}/hausdorff_distance",
            "Dev Mean ↓": f"{prefix}/dev_mean",
            "Dev Max ↓": f"{prefix}/dev_max",
        }
        for metric_name, col_name in col_map.items():
            row[metric_name] = _safe_mean(df, col_name)
        rows.append(row)
    return pd.DataFrame(rows)
