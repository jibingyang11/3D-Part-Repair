"""Publication-quality plotting utilities for paper figures."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------ #
#  Paper style
# ------------------------------------------------------------------ #

def set_paper_style():
    """Configure matplotlib for Springer / Visual Computer style."""
    matplotlib.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


COLORS = {
    "center_fan":                  "#5B9BD5",
    "planar_removed_part_aware":   "#70AD47",
    "planar_largest_hole_only":    "#FFC000",
    "trimesh_fill_all":            "#ED7D31",
    "advancing_front_rpa":         "#A855F7",
    "open3d_poisson":              "#FF6384",
}

METHOD_LABELS = {
    "center_fan":                  "Center-fan",
    "planar_removed_part_aware":   "Planar+RPA",
    "planar_largest_hole_only":    "Planar+LH-only",
    "trimesh_fill_all":            "Trimesh fill-all",
    "advancing_front_rpa":         "Adv-front+RPA",
    "open3d_poisson":              "Poisson recon",
}


# ------------------------------------------------------------------ #
#  Figure 4: Quantitative summary bar chart (3 panels)
# ------------------------------------------------------------------ #

def plot_quantitative_summary(df: pd.DataFrame, save_path: str,
                              methods=None):
    """Generate Figure 4: Closure / Complexity-Quality / Locality."""
    set_paper_style()

    if methods is None:
        methods = ["center_fan", "planar_removed_part_aware",
                   "planar_largest_hole_only","trimesh_fill_all","advancing_front_rpa","open3d_poisson"]

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [COLORS.get(m, "#999999") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Closure
    ax = axes[0]
    vals = [df[f"{m}/target_loop_length_after"].mean() for m in methods]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean residual target loop length")
    ax.set_title("(a) Closure")
    ax.tick_params(axis="x", rotation=15)

    # (b) Patch complexity / quality
    ax = axes[1]
    x = np.arange(len(methods))
    w = 0.35
    faces = [df[f"{m}/n_new_faces"].mean() for m in methods]
    quals = [df[f"{m}/avg_new_face_quality"].mean() for m in methods]
    ax.bar(x - w / 2, faces, w, color=colors, edgecolor="black", linewidth=0.5)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, quals, w, color=colors, edgecolor="black",
            linewidth=0.5, alpha=0.45, hatch="//")
    for i, (fv, qv) in enumerate(zip(faces, quals)):
        ax.text(i - w / 2, fv + 0.5, f"{fv:.1f}", ha="center", fontsize=7)
        ax2.text(i + w / 2, qv + 0.005, f"Q={qv:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean added faces")
    ax2.set_ylabel("Mean quality")
    ax.set_title("(b) Patch complexity / quality")

    # (c) Locality
    ax = axes[2]
    inside = [df[f"{m}/n_faces_inside"].mean() for m in methods]
    loc_r  = [df[f"{m}/locality_ratio"].mean() for m in methods]
    bars = ax.bar(labels, inside, color=colors, edgecolor="black", linewidth=0.5)
    for bar, iv, lr in zip(bars, inside, loc_r):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{iv:.1f}\nR={lr:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Mean added faces inside zone")
    ax.set_title("(c) Locality")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Ablation boxplots
# ------------------------------------------------------------------ #

def plot_ablation_boxplots(df_rpa: pd.DataFrame, df_lh: pd.DataFrame,
                           save_path: str):
    """Boxplot comparison of RPA vs LH-only."""
    set_paper_style()

    metrics = [
        ("target_loop_length_after", "Residual Target\nLoop Length", True),
        ("improvement", "Improvement", False),
        ("locality_ratio", "Locality Ratio", False),
        ("avg_new_face_quality", "Avg Triangle\nQuality", False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))

    for ax, (col, title, _) in zip(axes, metrics):
        data = [df_rpa[col].dropna().values, df_lh[col].dropna().values]
        bp = ax.boxplot(data, labels=["RPA", "LH-Only"], patch_artist=True,
                        widths=0.5)
        bp["boxes"][0].set_facecolor(COLORS["planar_removed_part_aware"])
        bp["boxes"][1].set_facecolor(COLORS["planar_largest_hole_only"])
        ax.set_title(title, fontsize=10)

    plt.suptitle("Ablation: Removed-Part-Aware vs Largest-Hole-Only",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  SOTA comparison table + bar chart
# ------------------------------------------------------------------ #

def plot_sota_comparison(df: pd.DataFrame, methods: list, save_path: str):
    """Bar chart comparing all methods including SOTA baselines."""
    set_paper_style()

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [COLORS.get(m, "#999999") for m in methods]

    metric_cols = [
        ("target_loop_length_after", "Residual"),
        ("locality_ratio", "Locality"),
        ("avg_new_face_quality", "Quality"),
        ("chamfer_distance", "Chamfer Dist."),
    ]

    available = [(col, name) for col, name in metric_cols
                 if f"{methods[0]}/{col}" in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, name) in zip(axes, available):
        vals = []
        for m in methods:
            c = f"{m}/{col}"
            vals.append(df[c].mean() if c in df.columns else 0)
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)


# ------------------------------------------------------------------ #
#  Failure case panel
# ------------------------------------------------------------------ #

def plot_failure_cases(df: pd.DataFrame, save_path: str, top_n: int = 5):
    """Bar chart of top failure cases."""
    set_paper_style()

    rpa_col = "planar_removed_part_aware/target_loop_length_after"
    lh_col  = "planar_largest_hole_only/target_loop_length_after"

    if rpa_col not in df.columns or lh_col not in df.columns:
        return

    df2 = df.copy()
    df2["gap"] = df2[lh_col] - df2[rpa_col]
    top = df2.nlargest(top_n, "gap")

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(top))
    w = 0.35
    ax.bar(x - w / 2, top[rpa_col].values, w, label="RPA",
           color=COLORS["planar_removed_part_aware"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, top[lh_col].values, w, label="LH-Only",
           color=COLORS["planar_largest_hole_only"], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in top["sample_id"].values], rotation=30)
    ax.set_ylabel("Residual target loop length")
    ax.set_title("Top Failure Cases: LH-Only vs RPA")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
