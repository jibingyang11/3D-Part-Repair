"""Configuration loading and management."""

import os
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file, merging with defaults."""
    project_root = Path(__file__).parent.parent
    default_path = project_root / "configs" / "default.yaml"

    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        cfg = _deep_merge(cfg, override)

    # Resolve relative paths
    for key, val in cfg.get("paths", {}).items():
        if key != "partnet_root" and isinstance(val, str) and not os.path.isabs(val):
            cfg["paths"][key] = str(project_root / val)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def ensure_dirs(cfg: dict):
    """Create all output directories."""
    for key in ["raw_data_dir", "processed_data_dir", "splits_dir",
                "figures_dir", "tables_dir", "metrics_dir",
                "repaired_meshes_dir", "logs_dir"]:
        path = cfg["paths"].get(key)
        if path:
            os.makedirs(path, exist_ok=True)
